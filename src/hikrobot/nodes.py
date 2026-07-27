# SPDX-License-Identifier: Apache-2.0
"""Typed access to the GenICam node map.

Every camera feature - exposure, ROI, pixel format, trigger wiring - is a named node in
the device's GenICam description. :class:`NodeMap` is the generic way to reach one;
:class:`~hikrobot.camera.Camera` puts named properties on top of the handful that most
callers need.

One measured fact shapes how failures read here. On an MV-CS023-10GM the SDK answers
``MV_E_GC_GENERIC`` for *both* a node that does not exist and a node accessed through the
wrong type - ``get_int("ExposureTime")`` on a float node fails exactly like
``get_int("NoSuchNode")``. The two are indistinguishable from the status alone, so this
module does not pretend to tell them apart: both raise :class:`~hikrobot.GenICamError`.
When one shows up, check the spelling *and* the type. Out-of-range writes are the
exception - those come back as ``MV_E_GC_RANGE`` and raise
:class:`~hikrobot.ValueOutOfRangeError`.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._ctypes_defs import (
    MVCC_ENUMENTRY,
    MVCC_ENUMVALUE_EX,
    MVCC_FLOATVALUE,
    MVCC_INTVALUE_EX,
    MVCC_STRINGVALUE,
    sdk,
)
from ._errors import call

if TYPE_CHECKING:
    from .camera import Camera

__all__ = ["FloatRange", "IntRange", "NodeMap"]


@dataclass(frozen=True)
class IntRange:
    """Current value and bounds of an integer node.

    Attributes:
        value: The value the node holds now.
        min: Smallest accepted value.
        max: Largest accepted value.
        inc: Granularity - accepted values are ``min + k * inc``. Width and height
            typically step by 2.

    Read-only nodes report the full 64-bit range rather than a meaningful one; treat
    ``min``/``max`` as advisory unless the node is writable.
    """

    value: int
    min: int
    max: int
    inc: int


@dataclass(frozen=True)
class FloatRange:
    """Current value and bounds of a float node. Float nodes carry no increment."""

    value: float
    min: float
    max: float


def _key(name: str) -> bytes:
    """Encode a node name the way the SDK expects it.

    Raises:
        ValueError: The name is not ASCII, so it cannot be a GenICam node name.
    """
    try:
        return name.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"node name must be ASCII, got {name!r}") from None


class NodeMap:
    """Generic access to one camera's GenICam nodes.

    Obtained from :attr:`hikrobot.Camera.nodes`; not constructed directly. Every method
    needs the camera to be open.

    Example:
        >>> cam.nodes.get_float("ExposureTime")  # doctest: +SKIP
        5000.0
        >>> cam.nodes.enum_entries("PixelFormat")  # doctest: +SKIP
        ['Mono8', 'Mono10', 'Mono10Packed', 'Mono12', 'Mono12Packed']
    """

    def __init__(self, camera: Camera) -> None:
        self._camera = camera

    def _handle(self) -> ctypes.c_void_p:
        return self._camera.handle

    # -- integers ----------------------------------------------------------------------

    def int_range(self, name: str) -> IntRange:
        """Read an integer node together with its bounds.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, or it is not an integer node.
        """
        out = MVCC_INTVALUE_EX()
        call(sdk().MV_CC_GetIntValueEx, self._handle(), _key(name), ctypes.byref(out))
        return IntRange(value=out.nCurValue, min=out.nMin, max=out.nMax, inc=out.nInc)

    def get_int(self, name: str) -> int:
        """Read an integer node.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, or it is not an integer node.
        """
        return self.int_range(name).value

    def set_int(self, name: str, value: int) -> None:
        """Write an integer node.

        Raises:
            CameraStateError: The camera is not open.
            ValueOutOfRangeError: Outside the node's bounds, or off its increment.
            GenICamError: No such node, it is not an integer node, or it is not writable
                in the camera's current state.
        """
        call(sdk().MV_CC_SetIntValueEx, self._handle(), _key(name), value)

    # -- floats ------------------------------------------------------------------------

    def float_range(self, name: str) -> FloatRange:
        """Read a float node together with its bounds.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, or it is not a float node.
        """
        out = MVCC_FLOATVALUE()
        call(sdk().MV_CC_GetFloatValue, self._handle(), _key(name), ctypes.byref(out))
        return FloatRange(value=out.fCurValue, min=out.fMin, max=out.fMax)

    def get_float(self, name: str) -> float:
        """Read a float node.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, or it is not a float node.
        """
        return self.float_range(name).value

    def set_float(self, name: str, value: float) -> None:
        """Write a float node.

        Reading a float node back does not have to return what was written. Two things
        get in the way: the SDK takes a 32-bit float, and the camera may quantise the
        value to a step of its own. ``MVCC_FLOATVALUE`` carries no increment field, so
        that step is not discoverable - writing ``Gain = 1.0`` to an MV-CS023-10GM reads
        back as ``1.0052``. Write, then read, and use what the camera reports; that second
        value is a fixed point.

        Raises:
            CameraStateError: The camera is not open.
            ValueOutOfRangeError: Outside the node's bounds.
            GenICamError: No such node, it is not a float node, or it is not writable in
                the camera's current state.
        """
        call(sdk().MV_CC_SetFloatValue, self._handle(), _key(name), value)

    # -- booleans ----------------------------------------------------------------------

    def get_bool(self, name: str) -> bool:
        """Read a boolean node.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, or it is not a boolean node.
        """
        out = ctypes.c_bool()
        call(sdk().MV_CC_GetBoolValue, self._handle(), _key(name), ctypes.byref(out))
        return bool(out.value)

    def set_bool(self, name: str, value: bool) -> None:
        """Write a boolean node.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, it is not a boolean node, or it is not writable in
                the camera's current state.
        """
        call(sdk().MV_CC_SetBoolValue, self._handle(), _key(name), value)

    # -- enumerations ------------------------------------------------------------------

    def get_enum(self, name: str) -> str:
        """Read an enumeration node as its symbolic name, e.g. ``"Mono8"``.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, or it is not an enumeration node.
        """
        out = MVCC_ENUMVALUE_EX()
        call(sdk().MV_CC_GetEnumValueEx, self._handle(), _key(name), ctypes.byref(out))
        return self._symbol(name, out.nCurValue)

    def set_enum(self, name: str, value: str) -> None:
        """Write an enumeration node by symbolic name.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, no such entry, or the entry is not selectable in
                the camera's current state.
        """
        call(sdk().MV_CC_SetEnumValueByString, self._handle(), _key(name), _key(value))

    def enum_entries(self, name: str) -> list[str]:
        """List the symbolic names an enumeration node currently accepts.

        The list reflects the camera's present state: entries that a different pixel
        format or trigger configuration would unlock may be missing.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, or it is not an enumeration node.
        """
        out = MVCC_ENUMVALUE_EX()
        call(sdk().MV_CC_GetEnumValueEx, self._handle(), _key(name), ctypes.byref(out))
        count = min(out.nSupportedNum, len(out.nSupportValue))
        return [self._symbol(name, out.nSupportValue[index]) for index in range(count)]

    def _symbol(self, name: str, value: int) -> str:
        entry = MVCC_ENUMENTRY()
        entry.nValue = value
        call(sdk().MV_CC_GetEnumEntrySymbolic, self._handle(), _key(name), ctypes.byref(entry))
        symbol: bytes = entry.chSymbolic
        return symbol.decode("ascii", errors="replace")

    # -- strings and commands ----------------------------------------------------------

    def get_string(self, name: str) -> str:
        """Read a string node.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, or it is not a string node.
        """
        out = MVCC_STRINGVALUE()
        call(sdk().MV_CC_GetStringValue, self._handle(), _key(name), ctypes.byref(out))
        value: bytes = out.chCurValue
        return value.decode("utf-8", errors="replace")

    def set_string(self, name: str, value: str) -> None:
        """Write a string node.

        Raises:
            CameraStateError: The camera is not open.
            ValueError: The value is not ASCII.
            GenICamError: No such node, it is not a string node, or the value is too long.
        """
        call(sdk().MV_CC_SetStringValue, self._handle(), _key(name), _key(value))

    def execute(self, name: str) -> None:
        """Run a command node, e.g. ``"TriggerSoftware"``.

        Raises:
            CameraStateError: The camera is not open.
            GenICamError: No such node, it is not a command node, or it is not executable
                in the camera's current state.
        """
        call(sdk().MV_CC_SetCommandValue, self._handle(), _key(name))
