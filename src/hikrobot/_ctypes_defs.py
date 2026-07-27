# SPDX-License-Identifier: Apache-2.0
"""ctypes transcription of the MVS structures needed to enumerate, open and grab.

Hand-written from the vendor C headers of MVS SDK 4.4.1 - ``CameraParams.h`` for the
structures and constants, ``MvCameraControl.h`` for the entry-point signatures. Only what
the enumerate/open/grab path needs is here; everything else is transcribed on demand.

Field names keep the vendor's spelling (``nWidth``, ``chSerialNumber``) so that a reader
can diff this file against the header line by line. Turning them into Python names is the
job of the public layer.

Three details that a careless transcription gets wrong:

* ``MvGvspPixelType`` contains ``PixelType_Gvsp_Undefined = 0xFFFFFFFF`` and a custom-format
  bit of ``0x80000000``, so its values do not fit in a signed ``int``. The C enum is
  therefore unsigned, and ``enPixelType`` must be read as :class:`ctypes.c_uint` - as a
  signed field every custom 3D format would come back negative.
* ``MV_CC_DEVICE_INFO`` ends in a union of six transport-specific structures. Only the
  GigE and USB3 members are ever read here, but every member has to be transcribed or the
  union - and with it the whole device list - comes out the wrong size.
* The structures are naturally aligned, not packed. ``MV_FRAME_OUT_INFO_EX`` even carries
  an explicit ``nReserved0`` whose header comment says it is there to 8-byte-align the
  ``int64_t`` that follows, which confirms the vendor compiles with default alignment.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    Union,
    c_bool,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_int64,
    c_ubyte,
    c_uint,
    c_uint64,
    c_ushort,
    c_void_p,
)
from typing import Any

from . import _loader

# --------------------------------------------------------------------------------------
# Constants (CameraParams.h)
# --------------------------------------------------------------------------------------

INFO_MAX_BUFFER_SIZE = 64
MV_MAX_DEVICE_NUM = 256

# Device transport layer protocol type, passed to MV_CC_EnumDevices as a bit mask.
MV_UNKNOW_DEVICE = 0x00000000
MV_GIGE_DEVICE = 0x00000001
MV_1394_DEVICE = 0x00000002
MV_USB_DEVICE = 0x00000004
MV_CAMERALINK_DEVICE = 0x00000008
MV_VIR_GIGE_DEVICE = 0x00000010
MV_VIR_USB_DEVICE = 0x00000020
MV_GENTL_GIGE_DEVICE = 0x00000040
MV_GENTL_CAMERALINK_DEVICE = 0x00000080
MV_GENTL_CXP_DEVICE = 0x00000100
MV_GENTL_XOF_DEVICE = 0x00000200

# Device access mode, passed to MV_CC_OpenDevice.
MV_ACCESS_Exclusive = 1
MV_ACCESS_ExclusiveWithSwitch = 2
MV_ACCESS_Control = 3
MV_ACCESS_ControlWithSwitch = 4
MV_ACCESS_ControlSwitchEnable = 5
MV_ACCESS_ControlSwitchEnableWithKey = 6
MV_ACCESS_Monitor = 7


# --------------------------------------------------------------------------------------
# Device information (CameraParams.h)
# --------------------------------------------------------------------------------------


class MV_GIGE_DEVICE_INFO(Structure):
    """GigE Vision device information."""

    _fields_ = [
        ("nIpCfgOption", c_uint),
        ("nIpCfgCurrent", c_uint),
        ("nCurrentIp", c_uint),
        ("nCurrentSubNetMask", c_uint),
        ("nDefultGateWay", c_uint),
        ("chManufacturerName", c_ubyte * 32),
        ("chModelName", c_ubyte * 32),
        ("chDeviceVersion", c_ubyte * 32),
        ("chManufacturerSpecificInfo", c_ubyte * 48),
        ("chSerialNumber", c_ubyte * 16),
        ("chUserDefinedName", c_ubyte * 16),
        ("nNetExport", c_uint),
        ("nReserved", c_uint * 4),
    ]


class MV_USB3_DEVICE_INFO(Structure):
    """USB3 Vision device information."""

    _fields_ = [
        ("CrtlInEndPoint", c_ubyte),
        ("CrtlOutEndPoint", c_ubyte),
        ("StreamEndPoint", c_ubyte),
        ("EventEndPoint", c_ubyte),
        ("idVendor", c_ushort),
        ("idProduct", c_ushort),
        ("nDeviceNumber", c_uint),
        ("chDeviceGUID", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chFamilyName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nbcdUSB", c_uint),
        ("nDeviceAddress", c_uint),
        ("nReserved", c_uint * 2),
    ]


class MV_CamL_DEV_INFO(Structure):
    """Camera Link device information."""

    _fields_ = [
        ("chPortID", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chFamilyName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nReserved", c_uint * 38),
    ]


class MV_CXP_DEVICE_INFO(Structure):
    """CoaXPress device information, as reported by a frame grabber."""

    _fields_ = [
        ("chInterfaceID", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerInfo", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceID", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nReserved", c_uint * 7),
    ]


class MV_CML_DEVICE_INFO(Structure):
    """Camera Link device information, as reported by a frame grabber."""

    _fields_ = [
        ("chInterfaceID", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerInfo", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceID", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nReserved", c_uint * 7),
    ]


class MV_XOF_DEVICE_INFO(Structure):
    """XoFLink device information, as reported by a frame grabber."""

    _fields_ = [
        ("chInterfaceID", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chVendorName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chModelName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chManufacturerInfo", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceVersion", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chSerialNumber", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chUserDefinedName", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("chDeviceID", c_ubyte * INFO_MAX_BUFFER_SIZE),
        ("nReserved", c_uint * 7),
    ]


class _SpecialInfo(Union):
    _fields_ = [
        ("stGigEInfo", MV_GIGE_DEVICE_INFO),
        ("stUsb3VInfo", MV_USB3_DEVICE_INFO),
        ("stCamLInfo", MV_CamL_DEV_INFO),
        ("stCMLInfo", MV_CML_DEVICE_INFO),
        ("stCXPInfo", MV_CXP_DEVICE_INFO),
        ("stXoFInfo", MV_XOF_DEVICE_INFO),
    ]


class MV_CC_DEVICE_INFO(Structure):
    """One enumerated device.

    ``nTLayerType`` selects which member of ``SpecialInfo`` is the valid one; reading the
    wrong member yields plausible-looking garbage rather than an error.
    """

    _fields_ = [
        ("nMajorVer", c_ushort),
        ("nMinorVer", c_ushort),
        ("nMacAddrHigh", c_uint),
        ("nMacAddrLow", c_uint),
        ("nTLayerType", c_uint),
        ("nDevTypeInfo", c_uint),
        ("nReserved", c_uint * 3),
        ("SpecialInfo", _SpecialInfo),
    ]


class MV_CC_DEVICE_INFO_LIST(Structure):
    """Result of ``MV_CC_EnumDevices``.

    The SDK owns the pointed-to device structures. They stay valid only until the next
    enumeration, so anything worth keeping must be copied out.
    """

    _fields_ = [
        ("nDeviceNum", c_uint),
        ("pDeviceInfo", POINTER(MV_CC_DEVICE_INFO) * MV_MAX_DEVICE_NUM),
    ]


# --------------------------------------------------------------------------------------
# Frames (CameraParams.h)
# --------------------------------------------------------------------------------------


class _UnparsedChunkList(Union):
    _fields_ = [
        # MV_CHUNK_DATA_CONTENT*. Kept as an opaque pointer: chunk data is not parsed yet,
        # and only the width of the member matters for the layout.
        ("pUnparsedChunkContent", c_void_p),
        ("nAligning", c_int64),
    ]


class MV_FRAME_OUT_INFO_EX(Structure):
    """Metadata of one grabbed frame.

    ``nWidth`` / ``nHeight`` saturate at 65535; the header directs callers to
    ``nExtendWidth`` / ``nExtendHeight`` above that. Likewise ``nFrameLen`` saturates at
    4 GiB and ``nFrameLenEx`` carries the real length.
    """

    _fields_ = [
        ("nWidth", c_ushort),
        ("nHeight", c_ushort),
        # enum MvGvspPixelType - unsigned, see the module docstring.
        ("enPixelType", c_uint),
        ("nFrameNum", c_uint),
        ("nDevTimeStampHigh", c_uint),
        ("nDevTimeStampLow", c_uint),
        ("nReserved0", c_uint),
        ("nHostTimeStamp", c_int64),
        ("nFrameLen", c_uint),
        ("nSecondCount", c_uint),
        ("nCycleCount", c_uint),
        ("nCycleOffset", c_uint),
        ("fGain", c_float),
        ("fExposureTime", c_float),
        ("nAverageBrightness", c_uint),
        ("nRed", c_uint),
        ("nGreen", c_uint),
        ("nBlue", c_uint),
        ("nFrameCounter", c_uint),
        ("nTriggerIndex", c_uint),
        ("nInput", c_uint),
        ("nOutput", c_uint),
        ("nOffsetX", c_ushort),
        ("nOffsetY", c_ushort),
        ("nChunkWidth", c_ushort),
        ("nChunkHeight", c_ushort),
        ("nLostPacket", c_uint),
        ("nUnparsedChunkNum", c_uint),
        ("UnparsedChunkList", _UnparsedChunkList),
        ("nExtendWidth", c_uint),
        ("nExtendHeight", c_uint),
        ("nFrameLenEx", c_uint64),
        ("nReserved", c_uint * 32),
    ]


class MV_FRAME_OUT(Structure):
    """A grabbed frame: a pointer into driver memory plus its metadata.

    ``pBufAddr`` is a node from the SDK's fixed pool, not memory this process owns. It is
    valid only until the node goes back with ``MV_CC_FreeImageBuffer``; after that the
    driver writes the next frame to the same address.
    """

    _fields_ = [
        ("pBufAddr", POINTER(c_ubyte)),
        ("stFrameInfo", MV_FRAME_OUT_INFO_EX),
        ("nRes", c_uint * 16),
    ]


# --------------------------------------------------------------------------------------
# Pixel formats (PixelType.h)
# --------------------------------------------------------------------------------------
#
# A pixel type packs three fields: bits 24-31 say monochrome/colour/custom, bits 16-23 the
# effective bits per pixel including padding, bits 0-15 an arbitrary id. Only the last is
# opaque, so bit depth is computable while the channel layout is not - hence the explicit
# table in `frame.py`.

MV_GVSP_PIX_MONO = 0x01000000
MV_GVSP_PIX_COLOR = 0x02000000
MV_GVSP_PIX_CUSTOM = 0x80000000
MV_GVSP_PIX_COLOR_MASK = 0xFF000000
MV_GVSP_PIX_EFFECTIVE_PIXEL_SIZE_MASK = 0x00FF0000
MV_GVSP_PIX_EFFECTIVE_PIXEL_SIZE_SHIFT = 16
MV_GVSP_PIX_ID_MASK = 0x0000FFFF


def _pixel(color: int, bits: int, identifier: int) -> int:
    """Compose a pixel type the way ``PixelType.h`` does."""
    return color | (bits << MV_GVSP_PIX_EFFECTIVE_PIXEL_SIZE_SHIFT) | identifier


def pixel_bit_depth(pixel_type: int) -> int:
    """Effective bits per pixel of a pixel type, padding included."""
    return (pixel_type & MV_GVSP_PIX_EFFECTIVE_PIXEL_SIZE_MASK) >> (
        MV_GVSP_PIX_EFFECTIVE_PIXEL_SIZE_SHIFT
    )


PixelType_Gvsp_Undefined = 0xFFFFFFFF

PixelType_Gvsp_Mono8 = _pixel(MV_GVSP_PIX_MONO, 8, 0x0001)
PixelType_Gvsp_Mono10 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0003)
PixelType_Gvsp_Mono10_Packed = _pixel(MV_GVSP_PIX_MONO, 12, 0x0004)
PixelType_Gvsp_Mono12 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0005)
PixelType_Gvsp_Mono12_Packed = _pixel(MV_GVSP_PIX_MONO, 12, 0x0006)
PixelType_Gvsp_Mono14 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0025)
PixelType_Gvsp_Mono16 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0007)

# Bayer data is undemosaiced, so the vendor tags it monochrome.
PixelType_Gvsp_BayerGR8 = _pixel(MV_GVSP_PIX_MONO, 8, 0x0008)
PixelType_Gvsp_BayerRG8 = _pixel(MV_GVSP_PIX_MONO, 8, 0x0009)
PixelType_Gvsp_BayerGB8 = _pixel(MV_GVSP_PIX_MONO, 8, 0x000A)
PixelType_Gvsp_BayerBG8 = _pixel(MV_GVSP_PIX_MONO, 8, 0x000B)
PixelType_Gvsp_BayerGR10 = _pixel(MV_GVSP_PIX_MONO, 16, 0x000C)
PixelType_Gvsp_BayerRG10 = _pixel(MV_GVSP_PIX_MONO, 16, 0x000D)
PixelType_Gvsp_BayerGB10 = _pixel(MV_GVSP_PIX_MONO, 16, 0x000E)
PixelType_Gvsp_BayerBG10 = _pixel(MV_GVSP_PIX_MONO, 16, 0x000F)
PixelType_Gvsp_BayerGR12 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0010)
PixelType_Gvsp_BayerRG12 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0011)
PixelType_Gvsp_BayerGB12 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0012)
PixelType_Gvsp_BayerBG12 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0013)
PixelType_Gvsp_BayerGR16 = _pixel(MV_GVSP_PIX_MONO, 16, 0x002E)
PixelType_Gvsp_BayerRG16 = _pixel(MV_GVSP_PIX_MONO, 16, 0x002F)
PixelType_Gvsp_BayerGB16 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0030)
PixelType_Gvsp_BayerBG16 = _pixel(MV_GVSP_PIX_MONO, 16, 0x0031)

PixelType_Gvsp_RGB8_Packed = _pixel(MV_GVSP_PIX_COLOR, 24, 0x0014)
PixelType_Gvsp_BGR8_Packed = _pixel(MV_GVSP_PIX_COLOR, 24, 0x0015)
PixelType_Gvsp_RGBA8_Packed = _pixel(MV_GVSP_PIX_COLOR, 32, 0x0016)
PixelType_Gvsp_BGRA8_Packed = _pixel(MV_GVSP_PIX_COLOR, 32, 0x0017)


# --------------------------------------------------------------------------------------
# Node map values (CameraParams.h)
# --------------------------------------------------------------------------------------

MV_MAX_ENUM_SYMBOLIC_NUM = 256
MV_MAX_SYMBOLIC_LEN = 64


class MVCC_INTVALUE_EX(Structure):
    """Current value and range of an integer node.

    The non-``Ex`` form of this structure is 32-bit; GenICam integer nodes are 64-bit, so
    only this one can carry values such as a full sensor timestamp.
    """

    _fields_ = [
        ("nCurValue", c_int64),
        ("nMax", c_int64),
        ("nMin", c_int64),
        ("nInc", c_int64),
        ("nReserved", c_uint * 16),
    ]


class MVCC_FLOATVALUE(Structure):
    """Current value and range of a float node. There is no increment for these."""

    _fields_ = [
        ("fCurValue", c_float),
        ("fMax", c_float),
        ("fMin", c_float),
        ("nReserved", c_uint * 4),
    ]


class MVCC_ENUMVALUE_EX(Structure):
    """Current value of an enumeration node and the raw values it accepts.

    The entries are integers; ``MV_CC_GetEnumEntrySymbolic`` turns one into its name. The
    non-``Ex`` form caps the list at 64, which some cameras' ``PixelFormat`` exceeds.
    """

    _fields_ = [
        ("nCurValue", c_uint),
        ("nSupportedNum", c_uint),
        ("nSupportValue", c_uint * MV_MAX_ENUM_SYMBOLIC_NUM),
        ("nReserved", c_uint * 4),
    ]


class MVCC_ENUMENTRY(Structure):
    """One entry of an enumeration: ``nValue`` goes in, ``chSymbolic`` comes back."""

    _fields_ = [
        ("nValue", c_uint),
        ("chSymbolic", c_char * MV_MAX_SYMBOLIC_LEN),
        ("nReserved", c_uint * 4),
    ]


class MVCC_STRINGVALUE(Structure):
    """Current value of a string node, with the length the node will accept."""

    _fields_ = [
        ("chCurValue", c_char * 256),
        ("nMaxLength", c_int64),
        ("nReserved", c_uint * 2),
    ]


# --------------------------------------------------------------------------------------
# Transport statistics (CameraParams.h)
# --------------------------------------------------------------------------------------

MV_MATCH_TYPE_NET_DETECT = 0x00000001
MV_MATCH_TYPE_USB_DETECT = 0x00000002


class MV_MATCH_INFO_NET_DETECT(Structure):
    """GigE traffic and packet-loss counters, accumulated between start and stop."""

    _fields_ = [
        ("nReceiveDataSize", c_int64),
        ("nLostPacketCount", c_int64),
        ("nLostFrameCount", c_uint),
        ("nNetRecvFrameCount", c_uint),
        ("nRequestResendPacketCount", c_int64),
        ("nResendPacketCount", c_int64),
    ]


class MV_MATCH_INFO_USB_DETECT(Structure):
    """USB3 traffic counters, accumulated between open and close."""

    _fields_ = [
        ("nReceiveDataSize", c_int64),
        ("nReceivedFrameCount", c_uint),
        ("nErrorFrameCount", c_uint),
        ("nReserved", c_uint * 2),
    ]


class MV_ALL_MATCH_INFO(Structure):
    """Request wrapper: ``nType`` selects the statistics, the caller owns the buffer."""

    _fields_ = [
        ("nType", c_uint),
        ("pInfo", c_void_p),
        ("nInfoSize", c_uint),
    ]


# --------------------------------------------------------------------------------------
# Entry points (MvCameraControl.h)
# --------------------------------------------------------------------------------------

#: ``(symbol, argtypes, restype)``. Every entry point is declared before first use: an
#: unset ``argtypes`` happens to work on x86_64 Linux and corrupts the stack on Windows.
#: ``restype`` follows the header, which declares a signed ``int``; the checked-call helper
#: normalises it to unsigned before matching against the ``MV_E_*`` constants.
PROTOTYPES: tuple[tuple[str, list[Any], Any], ...] = (
    ("MV_CC_GetSDKVersion", [], c_uint),
    ("MV_CC_EnumDevices", [c_uint, POINTER(MV_CC_DEVICE_INFO_LIST)], c_int),
    ("MV_CC_CreateHandle", [POINTER(c_void_p), POINTER(MV_CC_DEVICE_INFO)], c_int),
    ("MV_CC_DestroyHandle", [c_void_p], c_int),
    ("MV_CC_OpenDevice", [c_void_p, c_uint, c_ushort], c_int),
    ("MV_CC_CloseDevice", [c_void_p], c_int),
    ("MV_CC_IsDeviceConnected", [c_void_p], c_bool),
    ("MV_CC_SetImageNodeNum", [c_void_p, c_uint], c_int),
    ("MV_CC_StartGrabbing", [c_void_p], c_int),
    ("MV_CC_StopGrabbing", [c_void_p], c_int),
    ("MV_CC_GetImageBuffer", [c_void_p, POINTER(MV_FRAME_OUT), c_uint], c_int),
    ("MV_CC_FreeImageBuffer", [c_void_p, POINTER(MV_FRAME_OUT)], c_int),
    # Node map. Keys are GenICam node names, ASCII, e.g. b"ExposureTime".
    ("MV_CC_GetIntValueEx", [c_void_p, c_char_p, POINTER(MVCC_INTVALUE_EX)], c_int),
    ("MV_CC_SetIntValueEx", [c_void_p, c_char_p, c_int64], c_int),
    ("MV_CC_GetFloatValue", [c_void_p, c_char_p, POINTER(MVCC_FLOATVALUE)], c_int),
    ("MV_CC_SetFloatValue", [c_void_p, c_char_p, c_float], c_int),
    ("MV_CC_GetBoolValue", [c_void_p, c_char_p, POINTER(c_bool)], c_int),
    ("MV_CC_SetBoolValue", [c_void_p, c_char_p, c_bool], c_int),
    ("MV_CC_GetEnumValueEx", [c_void_p, c_char_p, POINTER(MVCC_ENUMVALUE_EX)], c_int),
    ("MV_CC_SetEnumValue", [c_void_p, c_char_p, c_uint], c_int),
    ("MV_CC_SetEnumValueByString", [c_void_p, c_char_p, c_char_p], c_int),
    ("MV_CC_GetEnumEntrySymbolic", [c_void_p, c_char_p, POINTER(MVCC_ENUMENTRY)], c_int),
    ("MV_CC_GetStringValue", [c_void_p, c_char_p, POINTER(MVCC_STRINGVALUE)], c_int),
    ("MV_CC_SetStringValue", [c_void_p, c_char_p, c_char_p], c_int),
    ("MV_CC_SetCommandValue", [c_void_p, c_char_p], c_int),
    # Transport tuning and statistics.
    ("MV_CC_GetAllMatchInfo", [c_void_p, POINTER(MV_ALL_MATCH_INFO)], c_int),
    # Returns the packet size itself, not a status - see `Camera.optimal_packet_size`.
    ("MV_CC_GetOptimalPacketSize", [c_void_p], c_int),
    ("MV_GIGE_SetResend", [c_void_p, c_uint, c_uint, c_uint], c_int),
)

_sdk: ctypes.CDLL | None = None


def apply_prototypes(lib: Any) -> None:
    """Set ``argtypes`` and ``restype`` on every entry point this package calls.

    Args:
        lib: The loaded MVS library, or a test double standing in for it.

    Raises:
        AttributeError: The library does not export one of the expected symbols, which
            means it is not the MVS library or is too old.
    """
    for symbol, argtypes, restype in PROTOTYPES:
        func = getattr(lib, symbol)
        func.argtypes = argtypes
        func.restype = restype


def sdk() -> Any:
    """Return the MVS library with every prototype declared, loading it on first call.

    Raises:
        SDKNotFoundError: The SDK is not installed or is installed somewhere unexpected.
        SDKLoadError: The library was found but could not be opened.
        UnsupportedPlatformError: The OS/CPU combination has no known SDK layout.
    """
    global _sdk
    if _sdk is None:
        lib = _loader.load()
        apply_prototypes(lib)
        _sdk = lib
    return _sdk


def reset() -> None:
    """Drop the cached library handle.

    For tests only; the prototypes are reapplied on the next :func:`sdk` call.
    """
    global _sdk
    _sdk = None
