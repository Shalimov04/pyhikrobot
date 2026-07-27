# SPDX-License-Identifier: Apache-2.0
"""Camera handle lifecycle.

A :class:`Camera` owns one SDK handle. :meth:`Camera.open` creates it and opens the
device; :meth:`Camera.close` closes and destroys it, leaving the object reusable. Nothing
here streams yet.

There is no ``__del__``: destroying an SDK handle from the garbage collector, possibly
during interpreter shutdown, is a reliable way to crash inside the vendor library. Use the
context manager, or call :meth:`Camera.close` yourself.
"""

from __future__ import annotations

import contextlib
import ctypes
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from ._ctypes_defs import (
    MV_ALL_MATCH_INFO,
    MV_FRAME_OUT,
    MV_MATCH_INFO_NET_DETECT,
    MV_MATCH_TYPE_NET_DETECT,
    MV_ACCESS_Control,
    MV_ACCESS_ControlSwitchEnable,
    MV_ACCESS_ControlSwitchEnableWithKey,
    MV_ACCESS_ControlWithSwitch,
    MV_ACCESS_Exclusive,
    MV_ACCESS_ExclusiveWithSwitch,
    MV_ACCESS_Monitor,
    sdk,
)
from ._errors import (
    MV_E_GC_TIMEOUT,
    MV_E_NODATA,
    MV_OK,
    CameraStateError,
    HikrobotError,
    NoDataError,
    call,
    check,
)
from .device import DeviceInfo
from .frame import Frame
from .nodes import FloatRange, NodeMap

__all__ = ["Camera", "TransportStats"]


@dataclass(frozen=True)
class TransportStats:
    """GigE traffic counters, accumulated between start and stop of acquisition.

    Attributes:
        received_bytes: Bytes received on the stream channel. This is wire traffic, not
            image payload - on an MV-CS023-10GM three 2304000-byte frames counted
            7148538 bytes, the extra 3.4% being GVSP packet headers.
        lost_packets: Packets the host never saw, after any resend.
        lost_frames: Frames dropped because of that loss.
        received_frames: Frames delivered whole.
        requested_resend_packets: Packets the host asked the camera to send again.
        resent_packets: Packets the camera actually sent again.

    A run with ``lost_packets == 0`` is what a correctly sized packet and an adequate link
    look like. Loss that resends fix shows up as a non-zero resend count with
    ``lost_frames == 0``; loss they cannot fix shows up as dropped frames.
    """

    received_bytes: int
    lost_packets: int
    lost_frames: int
    received_frames: int
    requested_resend_packets: int
    resent_packets: int


#: Statuses that mean "no frame within the timeout" rather than a failure. An
#: MV-CS023-10GM answers MV_E_NODATA; the header does not say whether every transport
#: agrees, so the GenICam timeout is accepted as well.
_NO_FRAME = (MV_E_NODATA, MV_E_GC_TIMEOUT)


def _timeout_error(timeout_ms: int) -> NoDataError:
    return NoDataError(
        MV_E_NODATA,
        "MV_CC_GetImageBuffer",
        "MV_E_NODATA",
        f"no frame arrived within {timeout_ms} ms",
    )


#: Public access-mode name -> ``MV_ACCESS_*``. ``"exclusive"`` is the SDK's own default:
#: other applications may read the control register but cannot take the camera.
_ACCESS_MODES = {
    "exclusive": MV_ACCESS_Exclusive,
    "exclusive-with-switch": MV_ACCESS_ExclusiveWithSwitch,
    "control": MV_ACCESS_Control,
    "control-with-switch": MV_ACCESS_ControlWithSwitch,
    "control-switch-enable": MV_ACCESS_ControlSwitchEnable,
    "control-switch-enable-with-key": MV_ACCESS_ControlSwitchEnableWithKey,
    "monitor": MV_ACCESS_Monitor,
}


class Camera:
    """One camera, identified by the :class:`~hikrobot.device.DeviceInfo` that found it.

    Construction is free and touches no SDK state; everything happens in :meth:`open`.

    Example:
        >>> from hikrobot import Camera, enumerate_devices
        >>> with Camera(enumerate_devices()[0]) as cam:  # doctest: +SKIP
        ...     print(cam.is_connected)
    """

    def __init__(self, device: DeviceInfo) -> None:
        self._device = device
        self._handle: ctypes.c_void_p | None = None
        self._nodes = NodeMap(self)
        self._grabbing = False
        # Frames handed out and not yet released. Weak, so a caller dropping a frame
        # without releasing it does not keep the object alive here.
        self._live_frames: weakref.WeakSet[Frame] = weakref.WeakSet()

    @property
    def info(self) -> DeviceInfo:
        """The device record this camera was constructed from."""
        return self._device

    @property
    def handle(self) -> ctypes.c_void_p:
        """The raw SDK handle.

        Mostly for the node map and, later, the streaming layer. The handle belongs to
        this object: do not close or destroy it.

        Raises:
            CameraStateError: The camera is not open.
        """
        if self._handle is None:
            raise CameraStateError(f"{self!r} is not open")
        return self._handle

    @property
    def sdk_library(self) -> Any:
        """The configured MVS library. Used by :class:`~hikrobot.frame.Frame`."""
        return sdk()

    @property
    def nodes(self) -> NodeMap:
        """Generic access to the GenICam node map.

        Everything the named properties below do goes through here, and anything they do
        not cover is reachable the same way.
        """
        return self._nodes

    @property
    def is_open(self) -> bool:
        """True between a successful :meth:`open` and the next :meth:`close`."""
        return self._handle is not None

    @property
    def is_connected(self) -> bool:
        """Ask the SDK whether the device is still reachable.

        False for a camera that is not open. Unlike every other entry point this one
        returns a plain boolean rather than a status code, so there is nothing to check.
        """
        if self._handle is None:
            return False
        return bool(sdk().MV_CC_IsDeviceConnected(self._handle))

    def open(self, access: str = "exclusive", switchover_key: int = 0) -> None:
        """Create the SDK handle and open the device.

        Args:
            access: One of the keys of the access-mode table - ``"exclusive"``,
                ``"control"``, ``"monitor"`` and the switch-over variants.
            switchover_key: Only meaningful for the ``*-with-key`` access modes.

        Raises:
            ValueError: Unknown access mode.
            CameraStateError: This camera is already open.
            AccessDeniedError: Another process holds the camera exclusively.
            StatusError: Any other refusal from the SDK.
        """
        try:
            mode = _ACCESS_MODES[access]
        except KeyError:
            known = ", ".join(sorted(_ACCESS_MODES))
            raise ValueError(f"unknown access mode {access!r}; known modes: {known}") from None

        if self._handle is not None:
            raise CameraStateError(f"{self!r} is already open")

        lib = sdk()
        handle = ctypes.c_void_p()
        call(lib.MV_CC_CreateHandle, ctypes.byref(handle), ctypes.byref(self._device.raw))
        try:
            call(lib.MV_CC_OpenDevice, handle, mode, switchover_key)
        except HikrobotError:
            # The handle exists even though the device did not open; leaking it would hold
            # SDK resources for the life of the process. A failure to destroy it is not
            # worth reporting over the reason the open failed.
            with contextlib.suppress(HikrobotError):
                call(lib.MV_CC_DestroyHandle, handle)
            raise
        self._handle = handle

    def close(self) -> None:
        """Close the device and destroy the handle.

        Idempotent: closing a camera that is not open does nothing. The handle is always
        destroyed, including when closing the device fails.

        Raises:
            StatusError: The SDK refused to close the device, or refused to destroy the
                handle after a successful close.
        """
        if self._handle is not None and self._grabbing:
            # Give the nodes back and shut the engine down before the handle goes; a
            # failure here must not stop the close.
            with contextlib.suppress(HikrobotError):
                self.stop_grabbing()

        handle, self._handle = self._handle, None
        if handle is None:
            return

        self._grabbing = False
        self._live_frames.clear()
        lib = sdk()
        try:
            call(lib.MV_CC_CloseDevice, handle)
        except HikrobotError:
            with contextlib.suppress(HikrobotError):
                call(lib.MV_CC_DestroyHandle, handle)
            raise
        call(lib.MV_CC_DestroyHandle, handle)

    # -- streaming ---------------------------------------------------------------------

    @property
    def is_grabbing(self) -> bool:
        """True between :meth:`start_grabbing` and :meth:`stop_grabbing`."""
        return self._grabbing

    def start_grabbing(self, node_count: int | None = None) -> None:
        """Start the acquisition engine.

        Args:
            node_count: Size of the SDK's buffer pool. More nodes absorb a slower consumer
                without dropping frames upstream, at the cost of one payload each. The
                pool is fixed once acquisition starts, so this has to be set here.

        Raises:
            CameraStateError: The camera is not open, or is already grabbing.
            StatusError: The SDK refused to start.
        """
        handle = self.handle
        if self._grabbing:
            raise CameraStateError(f"{self!r} is already grabbing")

        lib = sdk()
        if node_count is not None:
            call(lib.MV_CC_SetImageNodeNum, handle, node_count)
        call(lib.MV_CC_StartGrabbing, handle)
        self._grabbing = True

    def stop_grabbing(self) -> None:
        """Stop the acquisition engine, releasing any frame still outstanding.

        Idempotent.

        Stopping tears down the buffer pool, which makes every node handed out so far
        invalid - the SDK answers ``MV_E_CALLORDER`` to a release that arrives afterwards,
        and a NumPy view onto one is then pointing at memory the driver has reclaimed. So
        outstanding frames are returned to the driver here, before the stop, and are marked
        released; touching their pixel data afterwards raises
        :class:`~hikrobot.BufferReleasedError` rather than reading freed memory.

        Raises:
            CameraStateError: The camera is not open.
            StatusError: The SDK refused to stop.
        """
        if not self._grabbing:
            return
        self._grabbing = False
        for frame in list(self._live_frames):
            # A frame the caller never released. Returning it now is the only chance;
            # after the stop the SDK will not take it.
            with contextlib.suppress(HikrobotError):
                frame.release()
            frame._mark_released()
        self._live_frames.clear()
        call(sdk().MV_CC_StopGrabbing, self.handle)

    def _forget_frame(self, frame: Frame) -> None:
        """Stop tracking a frame whose node has gone back to the driver."""
        self._live_frames.discard(frame)

    def _grab(self, timeout_ms: int) -> Frame | None:
        """Take one node from the pool, or ``None`` if none arrived in time.

        The building block under both iterators. Measured on an MV-CS023-10GM, an expired
        timeout comes back as ``MV_E_NODATA``; ``MV_E_GC_TIMEOUT`` is accepted too because
        the header promises nothing about which one a given transport uses.
        """
        out = MV_FRAME_OUT()
        status = call(
            sdk().MV_CC_GetImageBuffer,
            self.handle,
            ctypes.byref(out),
            timeout_ms,
            allow=_NO_FRAME,
        )
        if status != MV_OK:
            return None
        frame = Frame(self, out)
        self._live_frames.add(frame)
        return frame

    @contextlib.contextmanager
    def _acquisition(self, node_count: int | None) -> Iterator[None]:
        """Start grabbing for the duration of an iterator, unless it was already running."""
        started_here = not self._grabbing
        if started_here:
            self.start_grabbing(node_count)
        try:
            yield
        finally:
            if started_here:
                self.stop_grabbing()

    def frames(self, timeout_ms: int = 1000, node_count: int | None = None) -> Iterator[Frame]:
        """Iterate frames, releasing each one when the loop body ends.

        The safe path. ``frame.data`` is valid inside the body and invalid outside it, on
        every route out including ``break``, ``return`` and exceptions. Use
        :meth:`Frame.copy` for anything that has to survive.

        Acquisition starts on the first frame and stops when the iterator is closed,
        unless :meth:`start_grabbing` was already called - in that case the caller keeps
        control of it.

        Args:
            timeout_ms: How long to wait for each frame.
            node_count: Buffer pool size, applied only if this call starts acquisition.

        Yields:
            One frame at a time. The frame is released as soon as the body finishes.

        Raises:
            CameraStateError: The camera is not open.
            NoDataError: No frame arrived within ``timeout_ms``. A camera waiting on a
                trigger does this; catch it and continue, or pass a longer timeout.
            StatusError: Any other refusal from the SDK.
        """
        with self._acquisition(node_count):
            while True:
                frame = self._grab(timeout_ms)
                if frame is None:
                    raise _timeout_error(timeout_ms)
                try:
                    yield frame
                finally:
                    frame.release()

    def frames_raw(self, timeout_ms: int = 1000, node_count: int | None = None) -> Iterator[Frame]:
        """Iterate frames, leaving the release to the caller.

        For consumers that outlive the loop body - the CUDA layer holds a node until its
        kernel has finished reading it. Every frame yielded here **must** get a
        :meth:`Frame.release`, or the pool drains and acquisition stalls: the pool holds
        exactly ``node_count`` nodes and the SDK's default is **one**, so holding a second
        frame without raising it fails with :class:`~hikrobot.InsufficientBufferError`.

        Two rules follow from acquisition ownership. If this call starts acquisition, then
        leaving the loop stops it, and stopping releases every frame still outstanding -
        so frames cannot be carried past the loop. To keep them, call
        :meth:`start_grabbing` first; the iterator then leaves acquisition alone and the
        frames stay yours until you release them.

        Args:
            timeout_ms: How long to wait for each frame.
            node_count: Buffer pool size, applied only if this call starts acquisition.

        Yields:
            One frame at a time, still holding its node.

        Raises:
            CameraStateError: The camera is not open.
            NoDataError: No frame arrived within ``timeout_ms``.
            StatusError: Any other refusal from the SDK.
        """
        with self._acquisition(node_count):
            while True:
                frame = self._grab(timeout_ms)
                if frame is None:
                    raise _timeout_error(timeout_ms)
                yield frame

    # -- GigE transport ----------------------------------------------------------------
    #
    # The knobs that decide whether streaming works at all on a GigE link. Everything here
    # is GigE-only per the header; on another transport the SDK refuses and that refusal
    # is what surfaces, rather than a guess made here.

    @property
    def packet_size(self) -> int:
        """Stream-channel packet size in bytes (``GevSCPSPacketSize``).

        Too large for the path and frames arrive incomplete or not at all - the classic
        symptom of an MTU that the switch, the NIC or a driver setting does not actually
        carry. :meth:`tune_packet_size` picks a size that works.
        """
        return self._nodes.get_int("GevSCPSPacketSize")

    @packet_size.setter
    def packet_size(self, value: int) -> None:
        self._nodes.set_int("GevSCPSPacketSize", value)

    @property
    def packet_delay(self) -> int:
        """Inter-packet delay in device timestamp ticks (``GevSCPD``).

        Spacing packets out keeps a burst from overrunning a switch buffer or a slower
        host, at the cost of frame rate. Ticks, not microseconds: divide by the camera's
        ``GevTimestampTickFrequency`` node.
        """
        return self._nodes.get_int("GevSCPD")

    @packet_delay.setter
    def packet_delay(self, value: int) -> None:
        self._nodes.set_int("GevSCPD", value)

    @property
    def optimal_packet_size(self) -> int:
        """Ask the SDK for the largest packet size this network path carries.

        Must be called with the camera open and acquisition **not** started; the SDK
        probes the path to answer.

        This entry point is the odd one out: it returns the packet size itself rather than
        a status, and reports failure by returning an ``MV_E_*`` code instead. Values with
        the high bit set are therefore treated as errors - confirmed by passing a null
        handle, which returns ``MV_E_HANDLE``.

        Returns:
            The packet size in bytes. On a path without jumbo frames this is 1500.

        Raises:
            CameraStateError: The camera is not open.
            StatusError: The SDK refused, e.g. on a non-GigE device.
        """
        returned: int = sdk().MV_CC_GetOptimalPacketSize(self.handle)
        status = returned & 0xFFFFFFFF
        if status & 0x80000000:
            return check(status, "MV_CC_GetOptimalPacketSize")
        return status

    def tune_packet_size(self) -> int:
        """Probe the path and apply the packet size it supports.

        The first thing to try when frames arrive incomplete. Acquisition must not be
        running: the probe needs the stream channel idle and the size is fixed once
        streaming starts.

        Returns:
            The packet size that was applied.

        Raises:
            CameraStateError: The camera is not open, or is grabbing.
            StatusError: The SDK refused the probe or the write.
        """
        if self._grabbing:
            raise CameraStateError(
                f"{self!r} is grabbing; the packet size can only be probed while idle"
            )
        size = self.optimal_packet_size
        self.packet_size = size
        return size

    def enable_resend(
        self,
        enabled: bool = True,
        max_resend_percent: int = 10,
        timeout_ms: int = 50,
    ) -> None:
        """Ask the camera to retransmit packets the host missed.

        The remedy for sporadic loss on a link that is otherwise fine. It cannot rescue a
        link that is simply too slow - there, lower the frame rate or the resolution.

        Args:
            enabled: Whether to request resends at all.
            max_resend_percent: Ceiling on how much of a frame may be re-requested.
            timeout_ms: How long to wait for a resent packet, 0 to 10000 per the header.

        Raises:
            CameraStateError: The camera is not open.
            StatusError: The SDK refused, e.g. on a non-GigE device.
        """
        call(
            sdk().MV_GIGE_SetResend,
            self.handle,
            1 if enabled else 0,
            max_resend_percent,
            timeout_ms,
        )

    @property
    def statistics(self) -> TransportStats:
        """Traffic and loss counters for the acquisition currently running.

        GigE only. The counters exist **only while acquisition is running** and reset to
        zero on every :meth:`start_grabbing` - measured on an MV-CS023-10GM, which answers
        ``MV_E_CALLORDER`` both before the first start and after any stop. So a run has to
        be measured from inside it: read this before leaving the ``frames()`` loop, or
        call :meth:`start_grabbing` yourself so that the iterator does not stop it.

        Raises:
            CameraStateError: The camera is not open.
            CallOrderError: Acquisition is not running.
            StatusError: The SDK refused, e.g. on a non-GigE device.
        """
        detect = MV_MATCH_INFO_NET_DETECT()
        request = MV_ALL_MATCH_INFO()
        request.nType = MV_MATCH_TYPE_NET_DETECT
        request.pInfo = ctypes.cast(ctypes.byref(detect), ctypes.c_void_p)
        request.nInfoSize = ctypes.sizeof(detect)
        call(sdk().MV_CC_GetAllMatchInfo, self.handle, ctypes.byref(request))
        return TransportStats(
            received_bytes=detect.nReceiveDataSize,
            lost_packets=detect.nLostPacketCount,
            lost_frames=detect.nLostFrameCount,
            received_frames=detect.nNetRecvFrameCount,
            requested_resend_packets=detect.nRequestResendPacketCount,
            resent_packets=detect.nResendPacketCount,
        )

    # -- typed properties --------------------------------------------------------------
    #
    # Thin named wrappers over the node map. The node names are GenICam SFNC standard
    # features and were confirmed present on an MV-CS023-10GM; a camera that spells one
    # differently raises GenICamError, and `nodes` remains available for it.

    @property
    def width(self) -> int:
        """Image width in pixels. Steps by the node's increment, usually 2."""
        return self._nodes.get_int("Width")

    @width.setter
    def width(self, value: int) -> None:
        self._nodes.set_int("Width", value)

    @property
    def height(self) -> int:
        """Image height in pixels. Steps by the node's increment, usually 2."""
        return self._nodes.get_int("Height")

    @height.setter
    def height(self, value: int) -> None:
        self._nodes.set_int("Height", value)

    @property
    def offset_x(self) -> int:
        """Horizontal ROI origin in pixels."""
        return self._nodes.get_int("OffsetX")

    @offset_x.setter
    def offset_x(self, value: int) -> None:
        self._nodes.set_int("OffsetX", value)

    @property
    def offset_y(self) -> int:
        """Vertical ROI origin in pixels."""
        return self._nodes.get_int("OffsetY")

    @offset_y.setter
    def offset_y(self, value: int) -> None:
        self._nodes.set_int("OffsetY", value)

    @property
    def payload_size(self) -> int:
        """Bytes in one frame at the current geometry and pixel format. Read-only."""
        return self._nodes.get_int("PayloadSize")

    @property
    def exposure_us(self) -> float:
        """Exposure time in microseconds.

        The unit is the SFNC one and matches what the camera reports: an MV-CS023-10GM
        allows 15 to 9999451, which is only sensible as microseconds.

        Writing has no effect while ``ExposureAuto`` is not ``"Off"``.
        """
        return self._nodes.get_float("ExposureTime")

    @exposure_us.setter
    def exposure_us(self, value: float) -> None:
        self._nodes.set_float("ExposureTime", value)

    @property
    def exposure_range_us(self) -> FloatRange:
        """Current exposure and the bounds the camera accepts, in microseconds."""
        return self._nodes.float_range("ExposureTime")

    @property
    def gain_db(self) -> float:
        """Analogue gain in decibels.

        Cameras quantise gain to a hardware step that the node map does not expose, so
        reading back after a write returns a nearby value rather than the exact one - an
        MV-CS023-10GM turns ``1.0`` into ``1.0052``. Read the property back if the precise
        value matters.

        Writing has no effect while ``GainAuto`` is not ``"Off"``.
        """
        return self._nodes.get_float("Gain")

    @gain_db.setter
    def gain_db(self, value: float) -> None:
        self._nodes.set_float("Gain", value)

    @property
    def gain_range_db(self) -> FloatRange:
        """Current gain and the bounds the camera accepts, in decibels."""
        return self._nodes.float_range("Gain")

    @property
    def frame_rate(self) -> float:
        """Configured acquisition frame rate in hertz.

        This is the requested rate, not the achieved one - exposure time, bandwidth and
        ``AcquisitionFrameRateEnable`` all cap it. The camera reports what it can actually
        deliver through the ``ResultingFrameRate`` node.
        """
        return self._nodes.get_float("AcquisitionFrameRate")

    @frame_rate.setter
    def frame_rate(self, value: float) -> None:
        self._nodes.set_float("AcquisitionFrameRate", value)

    @property
    def pixel_format(self) -> str:
        """Current pixel format as its GenICam name, e.g. ``"Mono8"``."""
        return self._nodes.get_enum("PixelFormat")

    @pixel_format.setter
    def pixel_format(self, value: str) -> None:
        self._nodes.set_enum("PixelFormat", value)

    @property
    def pixel_formats(self) -> list[str]:
        """Pixel formats the camera accepts in its current state."""
        return self._nodes.enum_entries("PixelFormat")

    def __enter__(self) -> Camera:
        """Open the camera, unless it is already open."""
        if not self.is_open:
            self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return f"<Camera {self._device.name!r} serial={self._device.serial_number!r} {state}>"
