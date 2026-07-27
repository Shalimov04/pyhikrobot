# SPDX-License-Identifier: Apache-2.0
"""Device discovery.

:func:`enumerate_devices` asks the SDK what is reachable and returns plain Python objects.
The SDK owns the structures it fills in and reuses them on the next enumeration, so every
:class:`DeviceInfo` carries its own byte-for-byte copy of the device record. That copy is
what :class:`~hikrobot.camera.Camera` later hands to ``MV_CC_CreateHandle``, which is why a
``DeviceInfo`` stays usable after a second enumeration has run.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._ctypes_defs import (
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_GIGE_DEVICE,
    MV_MAX_DEVICE_NUM,
    MV_USB_DEVICE,
    MV_VIR_GIGE_DEVICE,
    MV_VIR_USB_DEVICE,
    sdk,
)
from ._errors import call

__all__ = ["DeviceInfo", "enumerate_devices"]

#: Public transport name -> the transport-layer bits ``MV_CC_EnumDevices`` accepts.
_TRANSPORT_BITS = {
    "gige": MV_GIGE_DEVICE | MV_VIR_GIGE_DEVICE,
    "usb": MV_USB_DEVICE | MV_VIR_USB_DEVICE,
}

_VIRTUAL_BITS = MV_VIR_GIGE_DEVICE | MV_VIR_USB_DEVICE


def _text(raw: Any) -> str:
    """Decode one of the vendor's fixed-size ``unsigned char`` name fields.

    The fields are NUL-padded and their encoding is not documented; a user-defined name
    typed on the camera's web page can be anything. Undecodable bytes are replaced rather
    than raising, because a name is never worth failing an enumeration over.
    """
    data: bytes = bytes(raw)
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _ipv4(value: int) -> str:
    """Format a GigE address field as dotted quad, most significant octet first.

    The header does not state the byte order, but the device record carries the subnet
    mask in the same encoding, and only this order yields a valid mask: an MV-CS023-10GM
    reported ``0xFFFFFF00``, which is ``255.255.255.0`` read this way and the impossible
    ``0.255.255.255`` read the other. The gateway landing in the same subnet as the
    address agrees.
    """
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def _mac(high: int, low: int) -> str:
    """Format the split MAC address fields as six colon-separated octets.

    The header does not spell out where the 48 bits are cut. On real hardware
    ``nMacAddrHigh`` came back as ``0x000034BD`` - the upper 16 bits are unused, so the
    high word carries exactly two octets and the low word the remaining four.
    """
    octets = (
        (high >> 8) & 0xFF,
        high & 0xFF,
        (low >> 24) & 0xFF,
        (low >> 16) & 0xFF,
        (low >> 8) & 0xFF,
        low & 0xFF,
    )
    return ":".join(f"{octet:02x}" for octet in octets)


@dataclass(frozen=True)
class DeviceInfo:
    """One device reported by :func:`enumerate_devices`.

    Attributes:
        transport: ``"gige"``, ``"usb"``, or ``"unknown"`` for a transport layer this
            package does not decode yet.
        is_virtual: True for the SDK's simulated cameras.
        model_name: Model as reported by the device.
        serial_number: Serial number; the usual way to pin a camera in a multi-camera rig.
        manufacturer_name: Manufacturer as reported by the device.
        user_defined_name: The name configured on the device, often empty.
        device_version: Firmware version string.
        ip_address: Current address, GigE only, otherwise ``None``.
        mac_address: Hardware address, GigE only, otherwise ``None``.
    """

    transport: str
    is_virtual: bool
    model_name: str
    serial_number: str
    manufacturer_name: str
    user_defined_name: str
    device_version: str
    ip_address: str | None = None
    mac_address: str | None = None
    #: Private copy of the vendor record, kept for ``MV_CC_CreateHandle``. The union holds
    #: only scalars and byte arrays, so a bitwise copy is self-contained.
    raw: MV_CC_DEVICE_INFO = field(repr=False, compare=False, default_factory=MV_CC_DEVICE_INFO)

    @property
    def name(self) -> str:
        """A short label for logs: the user-defined name if set, otherwise the model."""
        return self.user_defined_name or self.model_name


def _from_raw(raw: MV_CC_DEVICE_INFO) -> DeviceInfo:
    layer = raw.nTLayerType
    is_virtual = bool(layer & _VIRTUAL_BITS)
    copy = MV_CC_DEVICE_INFO.from_buffer_copy(raw)

    if layer & _TRANSPORT_BITS["gige"]:
        gige = raw.SpecialInfo.stGigEInfo
        return DeviceInfo(
            transport="gige",
            is_virtual=is_virtual,
            model_name=_text(gige.chModelName),
            serial_number=_text(gige.chSerialNumber),
            manufacturer_name=_text(gige.chManufacturerName),
            user_defined_name=_text(gige.chUserDefinedName),
            device_version=_text(gige.chDeviceVersion),
            ip_address=_ipv4(gige.nCurrentIp),
            mac_address=_mac(raw.nMacAddrHigh, raw.nMacAddrLow),
            raw=copy,
        )

    if layer & _TRANSPORT_BITS["usb"]:
        usb = raw.SpecialInfo.stUsb3VInfo
        return DeviceInfo(
            transport="usb",
            is_virtual=is_virtual,
            model_name=_text(usb.chModelName),
            serial_number=_text(usb.chSerialNumber),
            manufacturer_name=_text(usb.chManufacturerName),
            user_defined_name=_text(usb.chUserDefinedName),
            device_version=_text(usb.chDeviceVersion),
            raw=copy,
        )

    # TODO(verify): the CameraLink and GenTL transports use their own union members. They
    # are reported rather than dropped, but their name fields are not decoded until there
    # is hardware to check the layout against.
    return DeviceInfo(
        transport="unknown",
        is_virtual=is_virtual,
        model_name="",
        serial_number="",
        manufacturer_name="",
        user_defined_name="",
        device_version="",
        raw=copy,
    )


def _transport_mask(transports: Iterable[str]) -> int:
    mask = 0
    for name in transports:
        try:
            mask |= _TRANSPORT_BITS[name]
        except KeyError:
            known = ", ".join(sorted(_TRANSPORT_BITS))
            raise ValueError(f"unknown transport {name!r}; known transports: {known}") from None
    if mask == 0:
        raise ValueError("at least one transport must be given")
    return mask


def enumerate_devices(transports: Sequence[str] = ("gige", "usb")) -> list[DeviceInfo]:
    """Return the devices the SDK can currently reach.

    This is the first call that touches the SDK, so it is also where a missing
    installation surfaces.

    Args:
        transports: Transport layers to scan; ``"gige"`` and ``"usb"`` each cover both the
            physical and the SDK's virtual devices.

    Returns:
        One entry per device, in the order the SDK reported them. Empty if nothing is
        reachable - that is a normal result, not an error.

    Raises:
        ValueError: An unknown transport name was given.
        SDKNotFoundError: The MVS SDK is not installed.
        StatusError: The SDK refused the enumeration.
    """
    mask = _transport_mask(transports)
    lib = sdk()

    device_list = MV_CC_DEVICE_INFO_LIST()
    call(lib.MV_CC_EnumDevices, mask, ctypes.byref(device_list))

    # The SDK fills a fixed-size array; trust nDeviceNum but never index past the array.
    count = min(device_list.nDeviceNum, MV_MAX_DEVICE_NUM)
    devices = []
    for index in range(count):
        pointer = device_list.pDeviceInfo[index]
        if not pointer:
            continue
        devices.append(_from_raw(pointer.contents))
    return devices
