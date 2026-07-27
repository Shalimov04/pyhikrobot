# SPDX-License-Identifier: Apache-2.0
"""Exception hierarchy and the checked-call helper.

Every error raised by this package derives from :class:`HikrobotError`, so callers can
catch exactly one thing.

The status codes are a hand transcription of ``MvErrorDefine.h`` (MVS SDK 4.4.1). The SDK
has no error-string entry point, so the descriptions below are ours. The vendor groups the
codes into five ranges, and this module mirrors that grouping in the exception hierarchy:
a code with no named class of its own still raises the class of its range, carrying
:attr:`StatusError.status` and :attr:`StatusError.name`. Adding a named class later is
therefore backwards compatible - an existing ``except GigEError`` keeps catching it.

The ``MV_ALG_*`` codes of ``MvISPErrorDefine.h`` (range ``0x1000xxxx``) belong to the ISP
algorithm library, which this package does not wrap; they are deliberately not transcribed.
If one ever reaches us it surfaces as a plain :class:`StatusError` with its hex value
intact rather than being mistaken for a camera error.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, NamedTuple

__all__ = [
    "AccessDeniedError",
    "BufferInUseError",
    "BufferReleasedError",
    "CallOrderError",
    "CameraStateError",
    "CameraTimeoutError",
    "DeviceBusyError",
    "GenICamError",
    "GeneralError",
    "GigEError",
    "HikrobotError",
    "IPConflictError",
    "IncompleteImageError",
    "InsufficientBufferError",
    "InvalidHandleError",
    "NetworkError",
    "NoDataError",
    "NodeAccessError",
    "NotImplementedByDeviceError",
    "NotSupportedError",
    "PacketError",
    "ParameterError",
    "PreconditionError",
    "SDKLoadError",
    "SDKNotFoundError",
    "StatusError",
    "USBBandwidthError",
    "USBError",
    "UnsupportedPixelFormatError",
    "UnsupportedPlatformError",
    "UpgradeError",
    "ValueOutOfRangeError",
]

#: Success. Every SDK entry point returns this or one of the codes below.
MV_OK = 0x00000000

# The two codes the package itself branches on rather than merely reporting. Both are in
# the table below; a test asserts they agree, so these cannot drift.
MV_E_NODATA = 0x80000007
MV_E_GC_TIMEOUT = 0x80000107

_UINT32 = 0xFFFFFFFF


# --------------------------------------------------------------------------------------
# Hierarchy
# --------------------------------------------------------------------------------------


class HikrobotError(Exception):
    """Base class for every error raised by this package."""


class SDKNotFoundError(HikrobotError):
    """The MVS shared library could not be located.

    Raised on first real use, never at import time. The message lists the locations that
    were searched.
    """


class SDKLoadError(HikrobotError):
    """The MVS shared library was found but could not be opened.

    Usually an architecture mismatch (32-bit library, 64-bit interpreter or vice versa)
    or a missing dependency of the vendor library itself.
    """


class UnsupportedPlatformError(HikrobotError):
    """This OS/CPU combination has no known MVS SDK layout.

    Raised instead of guessing a directory name, because a wrong guess degrades into a
    confusing "not found" much later.
    """


class BufferReleasedError(HikrobotError):
    """A frame's pixel data was touched after its buffer went back to the driver.

    The array is a view onto a node from the SDK's pool, not memory this process owns.
    Once the node is released the driver writes the next frame into the same address, so
    the array keeps working and silently yields a mixture of two frames. This exception is
    what turns that into an error you can see. Use ``Frame.copy()`` for data that has to
    outlive the frame.
    """


class UnsupportedPixelFormatError(HikrobotError):
    """The frame's pixel format has no direct NumPy layout.

    Raised for bit-packed formats such as ``Mono12Packed``, where three bytes carry two
    pixels and no array shape describes the buffer. The raw bytes are still reachable.
    """


class CameraStateError(HikrobotError):
    """The camera is not in the state this operation needs.

    Raised by the wrapper before it reaches the SDK - opening an already open camera, for
    instance. A state mismatch the SDK catches instead surfaces as :class:`CallOrderError`.
    """


class StatusError(HikrobotError):
    """An SDK entry point returned a status other than ``MV_OK``.

    Also the class raised directly for a status outside every documented range.

    Attributes:
        status: The status as an unsigned 32-bit value, normalised from the signed ``int``
            the SDK actually returns.
        name: The vendor's symbolic name, e.g. ``"MV_E_ACCESS_DENIED"``. Empty for a
            status that is not in the transcribed table.
        description: Short human-readable cause. Empty for an untranscribed status.
        operation: The SDK entry point that failed, e.g. ``"MV_CC_OpenDevice"``.
    """

    def __init__(
        self,
        status: int,
        operation: str,
        name: str = "",
        description: str = "",
    ) -> None:
        self.status = status & _UINT32
        self.operation = operation
        self.name = name
        self.description = description
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if not self.name:
            return f"{self.operation} failed: unknown status 0x{self.status:08X}"
        return f"{self.operation} failed: {self.name} (0x{self.status:08X}) - {self.description}"

    def __reduce__(self) -> tuple[Any, tuple[int, str, str, str]]:
        # Default exception pickling replays type(self)(*self.args), and self.args holds
        # the formatted message, not this constructor's arguments. Multi-camera pipelines
        # do move errors across process boundaries, so spell the round trip out.
        return (self.__class__, (self.status, self.operation, self.name, self.description))


class GeneralError(StatusError):
    """A general SDK error, range ``0x80000000``-``0x800000FF``."""


class GenICamError(StatusError):
    """A GenICam-layer error, range ``0x80000100``-``0x800001FF``."""


class GigEError(StatusError):
    """A GigE Vision transport error, range ``0x80000200``-``0x800002FF``."""


class USBError(StatusError):
    """A USB3 Vision transport error, range ``0x80000300``-``0x800003FF``."""


class UpgradeError(StatusError):
    """A firmware-upgrade error, range ``0x80000400``-``0x800004FF``."""


class InvalidHandleError(GeneralError):
    """``MV_E_HANDLE`` - the camera handle is invalid or already destroyed."""


class NotSupportedError(GeneralError):
    """``MV_E_SUPPORT`` - this device or transport layer does not implement the call."""


class CallOrderError(GeneralError):
    """``MV_E_CALLORDER`` - the SDK functions were called in the wrong order.

    Typically streaming was started before the device was opened, or a setting that is
    only writable while idle was written during acquisition.
    """


class ParameterError(GeneralError):
    """``MV_E_PARAMETER`` - one of the arguments was rejected by the SDK."""


class PreconditionError(GeneralError):
    """``MV_E_PRECONDITION`` - a precondition failed, or the environment changed."""


class NoDataError(GeneralError):
    """``MV_E_NODATA`` - the SDK had no data to return.

    This is what an expired ``MV_CC_GetImageBuffer`` timeout reports - measured on an
    MV-CS023-10GM over GigE, where a camera held on an unfired trigger answers ``NODATA``
    rather than ``MV_E_GC_TIMEOUT``. The header documents only "return error code", so the
    acquisition path still accepts both as "no frame yet"; only GigE has been checked.
    """


class IncompleteImageError(GeneralError):
    """``MV_E_ABNORMAL_IMAGE`` - the frame is incomplete, usually from packet loss.

    On GigE this is the symptom of an undersized packet size, a too-small inter-packet
    delay, or an MTU that the path does not actually carry.

    TODO(verify): the header does not say whether the frame struct was populated before
    this status was returned. If it was, the node still has to go back to the driver, and
    raising between grab and free would leak one node out of a fixed pool. Until this is
    confirmed on hardware the acquisition path must release in a ``finally`` on every
    non-``MV_OK`` path rather than assuming there is nothing to release.
    """


class InsufficientBufferError(GeneralError):
    """Not enough buffer to complete the call.

    Covers ``MV_E_NOENOUGH_BUF``, ``MV_E_NOOUTBUF`` and ``MV_E_NOENOUGH_BUF_NUM``.
    """


class BufferInUseError(GeneralError):
    """``MV_E_BUF_IN_USE`` - the buffer is still owned by the driver.

    Returned when a node is handed back to the SDK while it is still in flight; see the
    buffer-lifetime rules in the package documentation.
    """


class ValueOutOfRangeError(GenICamError):
    """``MV_E_GC_RANGE`` - the written value is outside the node's allowed range."""


class NodeAccessError(GenICamError):
    """``MV_E_GC_ACCESS`` - the node is not readable or writable in the current state."""


class CameraTimeoutError(GenICamError):
    """``MV_E_GC_TIMEOUT`` - the device did not answer in time.

    Named with the ``Camera`` prefix on purpose: shadowing the builtin ``TimeoutError``
    in user code would be worse than the extra six characters.
    """


class NotImplementedByDeviceError(GigEError):
    """``MV_E_NOT_IMPLEMENTED`` - the device does not implement this command."""


class AccessDeniedError(GigEError):
    """``MV_E_ACCESS_DENIED`` - no permission to access the device.

    The usual cause is that the camera is already open exclusively somewhere else, for
    example in the vendor's MVS viewer.
    """


class DeviceBusyError(GigEError):
    """``MV_E_BUSY`` - the device is busy, or the network link dropped."""


class PacketError(GigEError):
    """``MV_E_PACKET`` - a network data packet error."""


class NetworkError(GigEError):
    """``MV_E_NETER`` - a network-level failure."""


class IPConflictError(GigEError):
    """``MV_E_IP_CONFLICT`` - two devices claim the same IP address."""


class USBBandwidthError(USBError):
    """``MV_E_USB_BANDWIDTH`` - insufficient USB bandwidth for the current settings."""


# --------------------------------------------------------------------------------------
# Transcription of MvErrorDefine.h (MVS SDK 4.4.1)
# --------------------------------------------------------------------------------------


class _Entry(NamedTuple):
    code: int
    name: str
    description: str


_CODES: tuple[_Entry, ...] = (
    # General error codes, 0x80000000 - 0x800000FF.
    _Entry(0x80000000, "MV_E_HANDLE", "error or invalid handle"),
    _Entry(0x80000001, "MV_E_SUPPORT", "function not supported"),
    _Entry(0x80000002, "MV_E_BUFOVER", "buffer overflow"),
    _Entry(0x80000003, "MV_E_CALLORDER", "functions called in the wrong order"),
    _Entry(0x80000004, "MV_E_PARAMETER", "incorrect parameter"),
    _Entry(0x80000006, "MV_E_RESOURCE", "failed to acquire a resource"),
    _Entry(0x80000007, "MV_E_NODATA", "no data"),
    _Entry(0x80000008, "MV_E_PRECONDITION", "precondition failed, or the environment changed"),
    _Entry(0x80000009, "MV_E_VERSION", "version mismatch"),
    _Entry(0x8000000A, "MV_E_NOENOUGH_BUF", "insufficient memory"),
    _Entry(
        0x8000000B, "MV_E_ABNORMAL_IMAGE", "abnormal image, possibly incomplete from packet loss"
    ),
    _Entry(0x8000000C, "MV_E_LOAD_LIBRARY", "failed to load a dynamic library"),
    _Entry(0x8000000D, "MV_E_NOOUTBUF", "no output buffer available"),
    _Entry(0x8000000E, "MV_E_ENCRYPT", "encryption error"),
    _Entry(0x8000000F, "MV_E_OPENFILE", "failed to open a file"),
    _Entry(0x80000010, "MV_E_BUF_IN_USE", "buffer already in use"),
    _Entry(0x80000011, "MV_E_BUF_INVALID", "invalid buffer address"),
    _Entry(0x80000012, "MV_E_NOALIGN_BUF", "buffer alignment error"),
    _Entry(0x80000013, "MV_E_NOENOUGH_BUF_NUM", "insufficient number of cache buffers"),
    _Entry(0x80000014, "MV_E_PORT_IN_USE", "port already in use"),
    _Entry(0x80000015, "MV_E_IMAGE_DECODEC", "decoding error, the SDK image check failed"),
    # TODO(verify): the header comments this code in Chinese only - it is the single entry
    # with no English text. The description below is our reading of it; confirm the
    # semantics before any code branches on this status.
    _Entry(0x80000016, "MV_E_UINT32_LIMIT", "image size exceeds the unsigned int limit"),
    _Entry(0x800000FF, "MV_E_UNKNOW", "unknown error"),
    # GenICam error codes, 0x80000100 - 0x800001FF.
    _Entry(0x80000100, "MV_E_GC_GENERIC", "general GenICam error"),
    _Entry(0x80000101, "MV_E_GC_ARGUMENT", "illegal GenICam argument"),
    _Entry(0x80000102, "MV_E_GC_RANGE", "the value is out of range"),
    _Entry(0x80000103, "MV_E_GC_PROPERTY", "property error"),
    _Entry(0x80000104, "MV_E_GC_RUNTIME", "running environment error"),
    _Entry(0x80000105, "MV_E_GC_LOGICAL", "logical error"),
    _Entry(0x80000106, "MV_E_GC_ACCESS", "node access condition error"),
    _Entry(0x80000107, "MV_E_GC_TIMEOUT", "timeout"),
    _Entry(0x80000108, "MV_E_GC_DYNAMICCAST", "transformation exception"),
    _Entry(0x800001FF, "MV_E_GC_UNKNOW", "unknown GenICam error"),
    # GigE Vision status codes, 0x80000200 - 0x800002FF.
    _Entry(0x80000200, "MV_E_NOT_IMPLEMENTED", "the command is not supported by the device"),
    _Entry(0x80000201, "MV_E_INVALID_ADDRESS", "the target address does not exist"),
    _Entry(0x80000202, "MV_E_WRITE_PROTECT", "the target address is not writable"),
    _Entry(0x80000203, "MV_E_ACCESS_DENIED", "no permission to access the device"),
    _Entry(0x80000204, "MV_E_BUSY", "the device is busy, or the network is disconnected"),
    _Entry(0x80000205, "MV_E_PACKET", "network data packet error"),
    _Entry(0x80000206, "MV_E_NETER", "network error"),
    _Entry(
        0x8000020E,
        "MV_E_SUPPORT_MODIFY_DEVICE_IP",
        "the current mode does not support modifying the device IP",
    ),
    _Entry(0x8000020F, "MV_E_KEY_VERIFICATION", "switchover key verification failed"),
    _Entry(0x80000221, "MV_E_IP_CONFLICT", "device IP conflict"),
    # USB3 Vision status codes, 0x80000300 - 0x800003FF.
    _Entry(0x80000300, "MV_E_USB_READ", "USB read error"),
    _Entry(0x80000301, "MV_E_USB_WRITE", "USB write error"),
    _Entry(0x80000302, "MV_E_USB_DEVICE", "USB device exception"),
    _Entry(0x80000303, "MV_E_USB_GENICAM", "GenICam error on the USB transport"),
    _Entry(0x80000304, "MV_E_USB_BANDWIDTH", "insufficient bandwidth"),
    _Entry(0x80000305, "MV_E_USB_DRIVER", "driver mismatch, or the driver is not installed"),
    _Entry(0x800003FF, "MV_E_USB_UNKNOW", "unknown USB error"),
    # Firmware upgrade codes, 0x80000400 - 0x800004FF. MV_E_UPG_LANGUSGE_MISMATCH is
    # spelled that way in the vendor header; the name is a fact about the SDK.
    _Entry(0x80000400, "MV_E_UPG_FILE_MISMATCH", "the firmware does not match the device"),
    _Entry(0x80000401, "MV_E_UPG_LANGUSGE_MISMATCH", "the firmware language does not match"),
    _Entry(0x80000402, "MV_E_UPG_CONFLICT", "upgrade conflict, the device is already upgrading"),
    _Entry(0x80000403, "MV_E_UPG_INNER_ERR", "internal camera error during upgrade"),
    _Entry(0x800004FF, "MV_E_UPG_UNKNOW", "unknown error during upgrade"),
)

#: Inclusive range -> the exception class every code in it falls back to.
_GROUPS: tuple[tuple[int, int, type[StatusError]], ...] = (
    (0x80000000, 0x800000FF, GeneralError),
    (0x80000100, 0x800001FF, GenICamError),
    (0x80000200, 0x800002FF, GigEError),
    (0x80000300, 0x800003FF, USBError),
    (0x80000400, 0x800004FF, UpgradeError),
)

#: Codes that earn a class of their own. Everything else uses its group class.
_NAMED: dict[int, type[StatusError]] = {
    0x80000000: InvalidHandleError,
    0x80000001: NotSupportedError,
    0x80000003: CallOrderError,
    0x80000004: ParameterError,
    0x80000007: NoDataError,
    0x80000008: PreconditionError,
    0x8000000A: InsufficientBufferError,
    0x8000000B: IncompleteImageError,
    0x8000000D: InsufficientBufferError,
    0x80000010: BufferInUseError,
    0x80000013: InsufficientBufferError,
    0x80000102: ValueOutOfRangeError,
    0x80000106: NodeAccessError,
    0x80000107: CameraTimeoutError,
    0x80000200: NotImplementedByDeviceError,
    0x80000203: AccessDeniedError,
    0x80000204: DeviceBusyError,
    0x80000205: PacketError,
    0x80000206: NetworkError,
    0x80000221: IPConflictError,
    0x80000304: USBBandwidthError,
}

_BY_CODE: dict[int, _Entry] = {entry.code: entry for entry in _CODES}


# --------------------------------------------------------------------------------------
# Checked calls
# --------------------------------------------------------------------------------------


def _exception_class(status: int) -> type[StatusError]:
    named = _NAMED.get(status)
    if named is not None:
        return named
    for low, high, group in _GROUPS:
        if low <= status <= high:
            return group
    return StatusError


def _exception_for(status: int, operation: str) -> StatusError:
    entry = _BY_CODE.get(status)
    name = "" if entry is None else entry.name
    description = "" if entry is None else entry.description
    return _exception_class(status)(status, operation, name, description)


def check(status: int, operation: str) -> int:
    """Raise the mapped exception unless ``status`` is ``MV_OK``.

    Args:
        status: The value an SDK entry point returned. The SDK declares its return type as
            a signed ``int`` while writing the constants as ``0x8000xxxx``, so this is
            normalised to unsigned before it is matched.
        operation: Name of the entry point, used in the message, e.g. ``"MV_CC_OpenDevice"``.

    Returns:
        ``MV_OK``. The return value exists so that call sites can be written as an
        expression.

    Raises:
        StatusError: Or one of its subclasses, chosen by the status code.
    """
    normalised = status & _UINT32
    if normalised == MV_OK:
        return MV_OK
    raise _exception_for(normalised, operation)


def call(func: Any, *args: Any, allow: Collection[int] = ()) -> int:
    """Invoke an SDK entry point and turn a bad status into a typed exception.

    This is the only way SDK functions should be called: there must be no bare
    ``if ret != 0`` anywhere in the package.

    Args:
        func: A ctypes function pointer, or anything callable returning a status.
        *args: Passed through to ``func`` unchanged.
        allow: Status codes that are expected rather than exceptional at this call site,
            for example a timeout while polling for a frame. Passing an empty collection
            means every nonzero status raises.

    Returns:
        The normalised status: ``MV_OK``, or one of the codes in ``allow``.

    Raises:
        StatusError: Or one of its subclasses, chosen by the status code.
    """
    operation = getattr(func, "__name__", None) or repr(func)
    returned: int = func(*args)
    status = returned & _UINT32
    if status == MV_OK or status in allow:
        return status
    raise _exception_for(status, operation)
