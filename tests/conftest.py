# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration and the fake MVS library.

Hardware tests are opt-in: they need a real camera and an installed SDK, so they are
skipped unless ``--hardware`` is passed. Everything under ``tests/unit`` must pass on a
machine with no SDK, no CUDA and no camera.

The fake sits exactly where the real ``CDLL`` would, so the code under test still packs
real ctypes structures, still has its ``argtypes`` applied, and still runs every status
code through the real exception mapping. A fake one level up - a mock ``Camera`` - would
exercise none of that.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any

import pytest

import hikrobot._ctypes_defs
import hikrobot._loader
from hikrobot._ctypes_defs import (
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_GIGE_DEVICE,
    MV_MATCH_INFO_NET_DETECT,
    MV_USB_DEVICE,
    PixelType_Gvsp_Mono8,
)
from hikrobot._errors import MV_E_NODATA, MV_OK

#: What a real camera answers for a missing node or a wrong-typed access - the SDK does
#: not distinguish the two.
MV_E_GC_GENERIC = 0x80000100
MV_E_GC_RANGE = 0x80000102
MV_E_PARAMETER = 0x80000004
MV_E_HANDLE = 0x80000000
MV_E_CALLORDER = 0x80000003
MV_E_BUF_INVALID = 0x80000011


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="run tests that require a real Hikrobot camera and an installed MVS SDK",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--hardware"):
        return
    skip = pytest.mark.skip(reason="needs a real camera; pass --hardware to run")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_leaked_library() -> object:
    """Keep a library loaded by one test from leaking into the next."""
    hikrobot._loader.reset()
    hikrobot._ctypes_defs.reset()
    yield
    hikrobot._loader.reset()
    hikrobot._ctypes_defs.reset()


# --------------------------------------------------------------------------------------
# The fake library
# --------------------------------------------------------------------------------------


def _fill(field: Any, text: str) -> None:
    """Write a NUL-padded string into one of the vendor's ``unsigned char`` arrays."""
    encoded = text.encode("utf-8")[: len(field) - 1]
    for index, byte in enumerate(encoded):
        field[index] = byte
    field[len(encoded)] = 0


def gige_device(
    model: str = "MV-CA050-10GM",
    serial: str = "GIGE0001",
    ip: tuple[int, int, int, int] = (192, 168, 1, 10),
    user_defined_name: str = "",
) -> MV_CC_DEVICE_INFO:
    """Build a plausible GigE entry, the way the SDK would fill one in."""
    info = MV_CC_DEVICE_INFO()
    info.nTLayerType = MV_GIGE_DEVICE
    info.nMacAddrHigh = 0x0000_00AB
    info.nMacAddrLow = 0xCDEF_0123
    gige = info.SpecialInfo.stGigEInfo
    _fill(gige.chModelName, model)
    _fill(gige.chSerialNumber, serial)
    _fill(gige.chManufacturerName, "Hikrobot")
    _fill(gige.chUserDefinedName, user_defined_name)
    _fill(gige.chDeviceVersion, "V1.2.3")
    gige.nCurrentIp = (ip[0] << 24) | (ip[1] << 16) | (ip[2] << 8) | ip[3]
    return info


def usb_device(model: str = "MV-CU060-10UM", serial: str = "USB0001") -> MV_CC_DEVICE_INFO:
    """Build a plausible USB3 entry."""
    info = MV_CC_DEVICE_INFO()
    info.nTLayerType = MV_USB_DEVICE
    usb = info.SpecialInfo.stUsb3VInfo
    _fill(usb.chModelName, model)
    _fill(usb.chSerialNumber, serial)
    _fill(usb.chManufacturerName, "Hikrobot")
    _fill(usb.chUserDefinedName, "")
    _fill(usb.chDeviceVersion, "V4.5.6")
    return info


class FakeEntryPoint:
    """One exported symbol: settable ``argtypes``/``restype``, dispatching to a handler."""

    def __init__(self, name: str, handler: Callable[..., int]) -> None:
        self.__name__ = name
        self._handler = handler
        self.argtypes: Any = None
        self.restype: Any = None
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any) -> int:
        self.calls.append(args)
        return self._handler(*args)


def _deref(argument: Any) -> Any:
    """Recover the object behind a ``byref`` wrapper or a pointer."""
    target = getattr(argument, "_obj", None)
    if target is not None:
        return target
    return argument.contents


class FakeMvsLibrary:
    """A stand-in for ``MvCameraControl`` at the CDLL boundary.

    Holds real ``MV_CC_DEVICE_INFO`` structures so that the device list it fills in points
    at memory that stays alive, and hands out opaque handles the way the SDK does.

    Attributes:
        devices: The device records enumeration will report. Assign to change the answer.
        statuses: Per-symbol status overrides, e.g. ``{"MV_CC_OpenDevice": MV_E_ACCESS_DENIED}``.
        open_handles: Handle values currently created but not destroyed.
        connected: What ``MV_CC_IsDeviceConnected`` answers.
    """

    def __init__(self, devices: list[MV_CC_DEVICE_INFO] | None = None) -> None:
        self.devices: list[MV_CC_DEVICE_INFO] = [] if devices is None else devices
        self.statuses: dict[str, int] = {}
        self.open_handles: set[int] = set()
        self.opened_handles: set[int] = set()
        self.connected = True
        self.enumerated_masks: list[int] = []
        self.null_entries = 0
        self.open_arguments: list[tuple[int, int]] = []
        self._next_handle = 0x1000
        self._entry_points: dict[str, FakeEntryPoint] = {}

        # Node map, seeded with what a real MV-CS023-10GM reports.
        # Integers are (value, min, max, inc); floats (value, min, max).
        self.int_nodes: dict[str, tuple[int, int, int, int]] = {
            "Width": (1920, 32, 1920, 2),
            "Height": (1200, 4, 1200, 2),
            "OffsetX": (0, 0, 0, 2),
            "OffsetY": (0, 0, 0, 2),
            "PayloadSize": (2304000, 0, 0xFFFFFFFF, 1),
            "GevSCPSPacketSize": (1500, 576, 9000, 2),
            "GevSCPD": (400, 0, 0xFFFFFFFF, 1),
        }
        self.float_nodes: dict[str, tuple[float, float, float]] = {
            "ExposureTime": (5000.0, 15.0, 9999451.0),
            "Gain": (0.0, 0.0, 23.981),
            "AcquisitionFrameRate": (41.0, 0.09, 100000.0),
        }
        self.bool_nodes: dict[str, bool] = {"ReverseX": False, "ReverseY": False}
        self.enum_nodes: dict[str, tuple[str, list[str]]] = {
            "PixelFormat": (
                "Mono8",
                ["Mono8", "Mono10", "Mono10Packed", "Mono12", "Mono12Packed"],
            ),
            "TriggerMode": ("Off", ["Off", "On"]),
        }
        self.string_nodes: dict[str, str] = {
            "DeviceUserID": "7",
            "DeviceModelName": "MV-CS023-10GM",
        }
        self.commands: set[str] = {"TriggerSoftware"}
        self.executed: list[str] = []

        # Streaming. Small frames keep the tests fast; the pool is fixed and recycled the
        # way the SDK's is, which is what makes the lifetime tests meaningful.
        self.grabbing = False
        self.no_frames = False
        self.node_count = 3
        self.frame_width = 16
        self.frame_height = 8
        self.frame_bytes_per_pixel = 1
        self.frame_pixel_type = PixelType_Gvsp_Mono8
        self.live_nodes: set[int] = set()
        self.freed: list[int] = []
        self._pool: list[Any] = []
        self._frames_served = 0

        # GigE transport.
        self.optimal_packet_size = 8164
        self.resend_settings: tuple[bool, int, int] | None = None
        self.stats_received_bytes = 0
        self.stats_lost_packets = 0
        self.stats_lost_frames = 0

    # -- plumbing ----------------------------------------------------------------------

    def __getattr__(self, name: str) -> FakeEntryPoint:
        # __getattr__ only fires for names not found normally, so the attributes set in
        # __init__ are untouched. The handler is looked up on the class, not the instance,
        # so a missing one cannot recurse back into here.
        if not name.startswith("MV_"):
            raise AttributeError(name)
        handler = getattr(type(self), f"_call_{name}", None)
        if handler is None:
            raise AttributeError(f"{name} is not exported by the fake library")
        entry_points: dict[str, FakeEntryPoint] = self.__dict__["_entry_points"]
        if name not in entry_points:
            entry_points[name] = FakeEntryPoint(name, handler.__get__(self))
        return entry_points[name]

    def entry_point(self, name: str) -> FakeEntryPoint:
        """Return the recorded entry point, for asserting on calls."""
        return getattr(self, name)  # type: ignore[no-any-return]

    def _status(self, symbol: str) -> int:
        return self.statuses.get(symbol, MV_OK)

    # -- entry points ------------------------------------------------------------------

    def _call_MV_CC_GetSDKVersion(self) -> int:
        return 0x04040103

    def _call_MV_CC_EnumDevices(self, mask: int, out: Any) -> int:
        self.enumerated_masks.append(mask)
        status = self._status("MV_CC_EnumDevices")
        device_list: MV_CC_DEVICE_INFO_LIST = _deref(out)
        if status != MV_OK:
            return status
        # nDeviceNum may exceed the pointers actually written; `null_entries` reproduces
        # that so the reader's guard is exercised.
        device_list.nDeviceNum = len(self.devices) + self.null_entries
        for index, device in enumerate(self.devices):
            device_list.pDeviceInfo[index] = ctypes.pointer(device)
        return MV_OK

    def _call_MV_CC_CreateHandle(self, out: Any, device: Any) -> int:
        status = self._status("MV_CC_CreateHandle")
        if status != MV_OK:
            return status
        # Touch the device record so that a wrong pointer type fails here.
        _ = _deref(device).nTLayerType
        self._next_handle += 0x10
        _deref(out).value = self._next_handle
        self.open_handles.add(self._next_handle)
        return MV_OK

    def _call_MV_CC_DestroyHandle(self, handle: Any) -> int:
        status = self._status("MV_CC_DestroyHandle")
        if status != MV_OK:
            return status
        self.open_handles.discard(_value(handle))
        return MV_OK

    def _call_MV_CC_OpenDevice(self, handle: Any, mode: int, key: int) -> int:
        status = self._status("MV_CC_OpenDevice")
        if status != MV_OK:
            return status
        self.open_arguments.append((mode, key))
        self.opened_handles.add(_value(handle))
        return MV_OK

    def _call_MV_CC_CloseDevice(self, handle: Any) -> int:
        status = self._status("MV_CC_CloseDevice")
        if status != MV_OK:
            return status
        self.opened_handles.discard(_value(handle))
        return MV_OK

    def _call_MV_CC_IsDeviceConnected(self, handle: Any) -> bool:
        return self.connected and _value(handle) in self.opened_handles

    def _call_MV_CC_SetImageNodeNum(self, handle: Any, count: int) -> int:
        status = self._status("MV_CC_SetImageNodeNum")
        if status != MV_OK:
            return status
        self.node_count = count
        self._pool = []
        return MV_OK

    def _call_MV_CC_StartGrabbing(self, handle: Any) -> int:
        status = self._status("MV_CC_StartGrabbing")
        if status != MV_OK:
            return status
        self.grabbing = True
        # The real counters restart with each acquisition.
        self._frames_served = 0
        return MV_OK

    def _call_MV_CC_StopGrabbing(self, handle: Any) -> int:
        status = self._status("MV_CC_StopGrabbing")
        if status != MV_OK:
            return status
        self.grabbing = False
        return MV_OK

    def _node_buffer(self) -> Any:
        """Hand out the next node of a fixed pool, the way the SDK recycles its own."""
        size = self.frame_width * self.frame_height * self.frame_bytes_per_pixel
        while len(self._pool) < self.node_count:
            self._pool.append((ctypes.c_ubyte * size)())
        buffer = self._pool[self._frames_served % self.node_count]
        if len(buffer) != size:
            self._pool = [(ctypes.c_ubyte * size)() for _ in range(self.node_count)]
            buffer = self._pool[self._frames_served % self.node_count]
        return buffer

    def _call_MV_CC_GetImageBuffer(self, handle: Any, frame: Any, timeout: int) -> int:
        status = self._status("MV_CC_GetImageBuffer")
        if status != MV_OK:
            return status
        if self.no_frames:
            return MV_E_NODATA

        buffer = self._node_buffer()
        # A recognisable payload: every byte is the frame number, so a stale view is
        # visibly a different frame rather than plausible-looking noise.
        ctypes.memset(buffer, self._frames_served % 251, len(buffer))

        out = _deref(frame)
        out.pBufAddr = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        info = out.stFrameInfo
        info.nWidth = self.frame_width
        info.nHeight = self.frame_height
        info.nExtendWidth = self.frame_width
        info.nExtendHeight = self.frame_height
        info.enPixelType = self.frame_pixel_type
        info.nFrameLen = len(buffer)
        info.nFrameLenEx = len(buffer)
        info.nFrameNum = self._frames_served
        info.nHostTimeStamp = 1785174061741 + self._frames_served
        info.nDevTimeStampHigh = 0
        info.nDevTimeStampLow = self._frames_served * 24643042
        info.fExposureTime = 5000.0
        info.fGain = 0.0
        info.nLostPacket = 0

        self._frames_served += 1
        self.live_nodes.add(ctypes.addressof(buffer))
        return MV_OK

    def _call_MV_CC_FreeImageBuffer(self, handle: Any, frame: Any) -> int:
        status = self._status("MV_CC_FreeImageBuffer")
        if status != MV_OK:
            return status
        out = _deref(frame)
        address = ctypes.cast(out.pBufAddr, ctypes.c_void_p).value
        if address is None:
            return MV_E_PARAMETER
        if address not in self.live_nodes:
            # A real double free is undefined behaviour; the fake reports it instead.
            return MV_E_BUF_INVALID
        self.live_nodes.discard(address)
        self.freed.append(address)
        # The driver reuses the node immediately: scribble on it so that a view kept past
        # the release is demonstrably looking at something else.
        size = out.stFrameInfo.nFrameLenEx or out.stFrameInfo.nFrameLen
        ctypes.memset(address, 0xEE, size)
        return MV_OK

    # -- node map ----------------------------------------------------------------------
    #
    # A node that is absent, or reached through the wrong accessor, answers
    # MV_E_GC_GENERIC - that is what a real MV-CS023-10GM does, and the SDK gives no way
    # to tell the two apart.

    def _node(self, table: dict[str, Any], key: Any) -> tuple[str | None, int]:
        name = key.decode("ascii")
        if name not in table:
            return None, MV_E_GC_GENERIC
        return name, MV_OK

    def _call_MV_CC_GetIntValueEx(self, handle: Any, key: Any, out: Any) -> int:
        status = self._status("MV_CC_GetIntValueEx")
        if status != MV_OK:
            return status
        name, status = self._node(self.int_nodes, key)
        if name is None:
            return status
        node = self.int_nodes[name]
        target = _deref(out)
        target.nCurValue, target.nMin, target.nMax, target.nInc = node
        return MV_OK

    def _call_MV_CC_SetIntValueEx(self, handle: Any, key: Any, value: int) -> int:
        name, status = self._node(self.int_nodes, key)
        if name is None:
            return status
        _current, low, high, inc = self.int_nodes[name]
        if not low <= value <= high or (value - low) % inc:
            return MV_E_GC_RANGE
        self.int_nodes[name] = (value, low, high, inc)
        return MV_OK

    def _call_MV_CC_GetFloatValue(self, handle: Any, key: Any, out: Any) -> int:
        name, status = self._node(self.float_nodes, key)
        if name is None:
            return status
        target = _deref(out)
        target.fCurValue, target.fMin, target.fMax = self.float_nodes[name]
        return MV_OK

    def _call_MV_CC_SetFloatValue(self, handle: Any, key: Any, value: float) -> int:
        name, status = self._node(self.float_nodes, key)
        if name is None:
            return status
        _current, low, high = self.float_nodes[name]
        if not low <= value <= high:
            return MV_E_GC_RANGE
        self.float_nodes[name] = (value, low, high)
        return MV_OK

    def _call_MV_CC_GetBoolValue(self, handle: Any, key: Any, out: Any) -> int:
        name, status = self._node(self.bool_nodes, key)
        if name is None:
            return status
        _deref(out).value = self.bool_nodes[name]
        return MV_OK

    def _call_MV_CC_SetBoolValue(self, handle: Any, key: Any, value: bool) -> int:
        name, status = self._node(self.bool_nodes, key)
        if name is None:
            return status
        self.bool_nodes[name] = bool(value)
        return MV_OK

    def _call_MV_CC_GetEnumValueEx(self, handle: Any, key: Any, out: Any) -> int:
        name, status = self._node(self.enum_nodes, key)
        if name is None:
            return status
        current, entries = self.enum_nodes[name]
        target = _deref(out)
        target.nCurValue = entries.index(current)
        target.nSupportedNum = len(entries)
        for index in range(len(entries)):
            target.nSupportValue[index] = index
        return MV_OK

    def _call_MV_CC_SetEnumValue(self, handle: Any, key: Any, value: int) -> int:
        name, status = self._node(self.enum_nodes, key)
        if name is None:
            return status
        _current, entries = self.enum_nodes[name]
        if not 0 <= value < len(entries):
            return MV_E_GC_GENERIC
        self.enum_nodes[name] = (entries[value], entries)
        return MV_OK

    def _call_MV_CC_SetEnumValueByString(self, handle: Any, key: Any, value: Any) -> int:
        name, status = self._node(self.enum_nodes, key)
        if name is None:
            return status
        _current, entries = self.enum_nodes[name]
        symbol = value.decode("ascii")
        if symbol not in entries:
            return MV_E_GC_GENERIC
        self.enum_nodes[name] = (symbol, entries)
        return MV_OK

    def _call_MV_CC_GetEnumEntrySymbolic(self, handle: Any, key: Any, out: Any) -> int:
        name, status = self._node(self.enum_nodes, key)
        if name is None:
            return status
        _current, entries = self.enum_nodes[name]
        entry = _deref(out)
        if not 0 <= entry.nValue < len(entries):
            return MV_E_GC_GENERIC
        entry.chSymbolic = entries[entry.nValue].encode("ascii")
        return MV_OK

    def _call_MV_CC_GetStringValue(self, handle: Any, key: Any, out: Any) -> int:
        name, status = self._node(self.string_nodes, key)
        if name is None:
            return status
        target = _deref(out)
        target.chCurValue = self.string_nodes[name].encode("utf-8")
        target.nMaxLength = 256
        return MV_OK

    def _call_MV_CC_SetStringValue(self, handle: Any, key: Any, value: Any) -> int:
        name, status = self._node(self.string_nodes, key)
        if name is None:
            return status
        self.string_nodes[name] = value.decode("utf-8")
        return MV_OK

    # -- GigE transport ----------------------------------------------------------------

    def _call_MV_CC_GetOptimalPacketSize(self, handle: Any) -> int:
        # This entry point returns the size itself, and reports failure by returning an
        # MV_E_* code - a real SDK answers MV_E_HANDLE to a null handle.
        status = self._status("MV_CC_GetOptimalPacketSize")
        if status != MV_OK:
            return status
        if _value(handle) == 0:
            return MV_E_HANDLE
        return self.optimal_packet_size

    def _call_MV_GIGE_SetResend(self, handle: Any, enable: int, percent: int, timeout: int) -> int:
        status = self._status("MV_GIGE_SetResend")
        if status != MV_OK:
            return status
        self.resend_settings = (bool(enable), percent, timeout)
        return MV_OK

    def _call_MV_CC_GetAllMatchInfo(self, handle: Any, info: Any) -> int:
        status = self._status("MV_CC_GetAllMatchInfo")
        if status != MV_OK:
            return status
        if not self.grabbing:
            # Measured on an MV-CS023-10GM: the counters exist only between start and
            # stop, and answer MV_E_CALLORDER outside that window.
            return MV_E_CALLORDER
        request = _deref(info)
        detect = ctypes.cast(request.pInfo, ctypes.POINTER(MV_MATCH_INFO_NET_DETECT)).contents
        detect.nReceiveDataSize = self.stats_received_bytes
        detect.nLostPacketCount = self.stats_lost_packets
        detect.nLostFrameCount = self.stats_lost_frames
        detect.nNetRecvFrameCount = self._frames_served
        detect.nRequestResendPacketCount = 0
        detect.nResendPacketCount = 0
        return MV_OK

    def _call_MV_CC_SetCommandValue(self, handle: Any, key: Any) -> int:
        name = key.decode("ascii")
        if name not in self.commands:
            return MV_E_GC_GENERIC
        self.executed.append(name)
        return MV_OK


def _value(handle: Any) -> int:
    """Read a handle argument, which arrives as a ``c_void_p`` or a plain integer."""
    raw = getattr(handle, "value", handle)
    return 0 if raw is None else int(raw)


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> FakeMvsLibrary:
    """Install a fake MVS library and return it for configuration and assertions."""
    library = FakeMvsLibrary()
    monkeypatch.setattr(hikrobot._loader, "_load", lambda: library)
    hikrobot._loader.reset()
    hikrobot._ctypes_defs.reset()
    return library
