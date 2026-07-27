# SPDX-License-Identifier: Apache-2.0
"""One grabbed frame and the lifetime rules around its pixel data.

A :class:`Frame` is a *view* onto a node of the SDK's fixed buffer pool, not an array this
process owns. The node goes back to the driver on :meth:`Frame.release`, and the driver
then writes the next frame into the same address - measured on an MV-CS023-10GM, eight
consecutive frames came back on seven distinct addresses, so reuse starts almost
immediately. An array held past the release keeps working and silently returns a mixture
of two frames, which is why :attr:`Frame.data` refuses to hand one out afterwards.

Metadata is snapshotted when the frame is constructed, so width, timestamps and the frame
number stay readable after the buffer is gone. Only the pixel data has a lifetime.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._ctypes_defs import (
    MV_FRAME_OUT,
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerBG10,
    PixelType_Gvsp_BayerBG12,
    PixelType_Gvsp_BayerBG16,
    PixelType_Gvsp_BayerGB8,
    PixelType_Gvsp_BayerGB10,
    PixelType_Gvsp_BayerGB12,
    PixelType_Gvsp_BayerGB16,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerGR10,
    PixelType_Gvsp_BayerGR12,
    PixelType_Gvsp_BayerGR16,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BayerRG10,
    PixelType_Gvsp_BayerRG12,
    PixelType_Gvsp_BayerRG16,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_BGRA8_Packed,
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_Mono10,
    PixelType_Gvsp_Mono10_Packed,
    PixelType_Gvsp_Mono12,
    PixelType_Gvsp_Mono12_Packed,
    PixelType_Gvsp_Mono14,
    PixelType_Gvsp_Mono16,
    PixelType_Gvsp_RGB8_Packed,
    PixelType_Gvsp_RGBA8_Packed,
    pixel_bit_depth,
)
from ._errors import BufferReleasedError, UnsupportedPixelFormatError, call

if TYPE_CHECKING:
    from .camera import Camera

__all__ = ["Frame"]


class _Layout:
    """How one pixel format maps onto a NumPy array."""

    __slots__ = ("channels", "dtype", "name")

    def __init__(self, name: str, dtype: str, channels: int) -> None:
        self.name = name
        self.dtype = np.dtype(dtype)
        self.channels = channels


#: Pixel type -> array layout. Bit depth is computable from the type, but the channel
#: layout is not, so the mapping is spelled out. Formats absent here still deliver raw
#: bytes; only the shaped array is refused.
_LAYOUTS: dict[int, _Layout] = {
    PixelType_Gvsp_Mono8: _Layout("Mono8", "uint8", 1),
    PixelType_Gvsp_Mono10: _Layout("Mono10", "uint16", 1),
    PixelType_Gvsp_Mono12: _Layout("Mono12", "uint16", 1),
    PixelType_Gvsp_Mono14: _Layout("Mono14", "uint16", 1),
    PixelType_Gvsp_Mono16: _Layout("Mono16", "uint16", 1),
    PixelType_Gvsp_BayerGR8: _Layout("BayerGR8", "uint8", 1),
    PixelType_Gvsp_BayerRG8: _Layout("BayerRG8", "uint8", 1),
    PixelType_Gvsp_BayerGB8: _Layout("BayerGB8", "uint8", 1),
    PixelType_Gvsp_BayerBG8: _Layout("BayerBG8", "uint8", 1),
    PixelType_Gvsp_BayerGR10: _Layout("BayerGR10", "uint16", 1),
    PixelType_Gvsp_BayerRG10: _Layout("BayerRG10", "uint16", 1),
    PixelType_Gvsp_BayerGB10: _Layout("BayerGB10", "uint16", 1),
    PixelType_Gvsp_BayerBG10: _Layout("BayerBG10", "uint16", 1),
    PixelType_Gvsp_BayerGR12: _Layout("BayerGR12", "uint16", 1),
    PixelType_Gvsp_BayerRG12: _Layout("BayerRG12", "uint16", 1),
    PixelType_Gvsp_BayerGB12: _Layout("BayerGB12", "uint16", 1),
    PixelType_Gvsp_BayerBG12: _Layout("BayerBG12", "uint16", 1),
    PixelType_Gvsp_BayerGR16: _Layout("BayerGR16", "uint16", 1),
    PixelType_Gvsp_BayerRG16: _Layout("BayerRG16", "uint16", 1),
    PixelType_Gvsp_BayerGB16: _Layout("BayerGB16", "uint16", 1),
    PixelType_Gvsp_BayerBG16: _Layout("BayerBG16", "uint16", 1),
    PixelType_Gvsp_RGB8_Packed: _Layout("RGB8", "uint8", 3),
    PixelType_Gvsp_BGR8_Packed: _Layout("BGR8", "uint8", 3),
    PixelType_Gvsp_RGBA8_Packed: _Layout("RGBA8", "uint8", 4),
    PixelType_Gvsp_BGRA8_Packed: _Layout("BGRA8", "uint8", 4),
}

#: Formats whose bits do not align to whole array elements. Named so that the error can
#: say which one it is instead of printing a bare hex value.
_PACKED_NAMES = {
    PixelType_Gvsp_Mono10_Packed: "Mono10Packed",
    PixelType_Gvsp_Mono12_Packed: "Mono12Packed",
}


class Frame:
    """A frame held in one node of the driver's buffer pool.

    Obtained from :meth:`hikrobot.Camera.frames` or
    :meth:`hikrobot.Camera.frames_raw`; not constructed directly.

    The pixel data is a view into memory owned by the SDK. It is valid only until the node
    is released - by the iteration body ending in ``frames()``, or by an explicit
    :meth:`release` in ``frames_raw()``. Anything that must outlive the frame goes through
    :meth:`copy`.

    Example:
        >>> for frame in camera.frames(timeout_ms=1000):  # doctest: +SKIP
        ...     total = frame.data.sum()          # fine, inside the body
        ...     keep = frame.copy()               # fine, owns its memory
        ...     leaked = frame.data               # the array dies with the body
    """

    __slots__ = (
        # The camera tracks outstanding frames weakly, which a slotted class only allows
        # with this entry.
        "__weakref__",
        "_camera",
        "_raw",
        "_released",
        "device_timestamp_ticks",
        "exposure_us",
        "frame_number",
        "gain_db",
        "height",
        "host_timestamp_ms",
        "lost_packets",
        "offset_x",
        "offset_y",
        "pixel_type",
        "size_bytes",
        "width",
    )

    def __init__(self, camera: Camera, raw: MV_FRAME_OUT) -> None:
        self._camera = camera
        self._raw = raw
        self._released = False

        info = raw.stFrameInfo
        # The 16-bit width/height saturate at 65535 and the extended fields carry the real
        # value; the same holds for the frame length. Prefer the wide field when the
        # camera fills it in, which the header says it does above the 16-bit limit.
        self.width: int = info.nExtendWidth or info.nWidth
        self.height: int = info.nExtendHeight or info.nHeight
        self.size_bytes: int = info.nFrameLenEx or info.nFrameLen
        self.pixel_type: int = info.enPixelType
        self.frame_number: int = info.nFrameNum
        self.lost_packets: int = info.nLostPacket
        self.offset_x: int = info.nOffsetX
        self.offset_y: int = info.nOffsetY
        self.exposure_us: float = info.fExposureTime
        self.gain_db: float = info.fGain
        #: Milliseconds since the Unix epoch, stamped by the host when the frame arrived.
        #: Confirmed against ``time.time()`` on an MV-CS023-10GM.
        self.host_timestamp_ms: int = info.nHostTimeStamp
        #: Device counter in its own ticks, *not* nanoseconds. Divide by the camera's
        #: ``GevTimestampTickFrequency`` node to get seconds - it read 100 MHz on an
        #: MV-CS023-10GM, but it is device-specific and must not be assumed.
        self.device_timestamp_ticks: int = (info.nDevTimeStampHigh << 32) | info.nDevTimeStampLow

    @property
    def pixel_format(self) -> str:
        """Pixel format name, e.g. ``"Mono8"``, or ``"0x…"`` for one not in the table."""
        layout = _LAYOUTS.get(self.pixel_type)
        if layout is not None:
            return layout.name
        packed = _PACKED_NAMES.get(self.pixel_type)
        return packed if packed is not None else f"0x{self.pixel_type:08X}"

    @property
    def bit_depth(self) -> int:
        """Effective bits per pixel, padding included, as the SDK encodes it."""
        return pixel_bit_depth(self.pixel_type)

    @property
    def is_released(self) -> bool:
        """True once the node has gone back to the driver."""
        return self._released

    @property
    def raw_bytes(self) -> npt.NDArray[Any]:
        """The frame's bytes as a flat read-only ``uint8`` view.

        Available for every pixel format, including the packed ones that :attr:`data`
        refuses. The lifetime caveat is the same: the view dies with the node.

        Raises:
            BufferReleasedError: The node has already gone back to the driver.
        """
        return self._view(np.dtype("uint8"), (self.size_bytes,))

    @property
    def data(self) -> npt.NDArray[Any]:
        """The pixels as a read-only NumPy view, shaped ``(height, width[, channels])``.

        Zero-copy: this is the driver's memory, not a copy, and it is read-only because
        writing into a node that the SDK is about to reuse corrupts the next frame. Use
        :meth:`copy` for an array you own and can modify.

        The view is invalid the moment the node is released. In ``frames()`` that is when
        the iteration body ends.

        Raises:
            BufferReleasedError: The node has already gone back to the driver.
            UnsupportedPixelFormatError: The format is bit-packed, so no array shape
                describes it; use :attr:`raw_bytes` instead.
        """
        layout = _LAYOUTS.get(self.pixel_type)
        if layout is None:
            raise UnsupportedPixelFormatError(
                f"pixel format {self.pixel_format} has no direct NumPy layout; "
                f"read Frame.raw_bytes and unpack it, or ask the camera for an "
                f"unpacked format"
            )
        shape = (
            (self.height, self.width)
            if layout.channels == 1
            else (self.height, self.width, layout.channels)
        )
        return self._view(layout.dtype, shape)

    def copy(self) -> npt.NDArray[Any]:
        """Return a writable array that owns its memory and outlives the frame.

        The explicit way to keep a frame. There is deliberately no "hold this one for me"
        shortcut: a copy of a 2 MB frame is visible in a profile, an accidentally retained
        view is not.

        Raises:
            BufferReleasedError: The node has already gone back to the driver.
            UnsupportedPixelFormatError: The format is bit-packed; copy
                :attr:`raw_bytes` instead.
        """
        return np.array(self.data, copy=True)

    def release(self) -> None:
        """Return the node to the driver.

        Idempotent, so a double release is harmless rather than a double free. After this
        the pixel data is gone; the metadata attributes remain readable.

        Raises:
            StatusError: The SDK refused to take the node back.
        """
        if self._released:
            return
        self._released = True
        self._camera._forget_frame(self)
        call(
            self._camera.sdk_library.MV_CC_FreeImageBuffer,
            self._camera.handle,
            ctypes.byref(self._raw),
        )

    def _mark_released(self) -> None:
        """Declare the buffer gone without asking the SDK to take it back.

        Used when acquisition stops: the pool is torn down as a whole, so there is nothing
        left to hand over, but the pixel data must still stop being readable.
        """
        self._released = True

    def _view(self, dtype: np.dtype[Any], shape: tuple[int, ...]) -> npt.NDArray[Any]:
        if self._released:
            raise BufferReleasedError(
                f"the buffer of {self!r} went back to the driver; the pixels there now "
                f"belong to a later frame. Use Frame.copy() to keep data past the "
                f"release."
            )

        needed = int(np.prod(shape)) * dtype.itemsize
        if needed > self.size_bytes:
            raise ValueError(
                f"{self.pixel_format} {self.width}x{self.height} needs {needed} bytes but "
                f"the frame carries {self.size_bytes}"
            )

        address = ctypes.cast(self._raw.pBufAddr, ctypes.c_void_p).value
        if address is None:
            raise BufferReleasedError(f"{self!r} carries no buffer address")

        buffer = (ctypes.c_ubyte * needed).from_address(address)
        array: npt.NDArray[Any] = np.frombuffer(memoryview(buffer), dtype=dtype).reshape(shape)
        array.flags.writeable = False
        return array

    def __repr__(self) -> str:
        state = "released" if self._released else "live"
        return (
            f"<Frame #{self.frame_number} {self.width}x{self.height} {self.pixel_format} {state}>"
        )
