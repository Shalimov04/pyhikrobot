# SPDX-License-Identifier: Apache-2.0
"""Thin, zero-copy Python bindings for Hikrobot machine-vision cameras (MVS SDK).

Importing this package never loads the vendor SDK: the shared library is located and
opened lazily on first real use, so ``import hikrobot`` succeeds on machines with no
SDK and no camera attached. A missing SDK surfaces as :class:`SDKNotFoundError` at
call time.

Every failure derives from :class:`HikrobotError`. A status returned by the SDK becomes a
:class:`StatusError` subclass chosen by the code; the raw ``MV_E_*`` values stay internal
and are reachable through :attr:`StatusError.status` and :attr:`StatusError.name`.
"""

__version__ = "0.0.1.dev0"

__all__ = [
    "AccessDeniedError",
    "BufferInUseError",
    "BufferReleasedError",
    "CallOrderError",
    "Camera",
    "CameraStateError",
    "CameraTimeoutError",
    "DeviceBusyError",
    "DeviceInfo",
    "FloatRange",
    "Frame",
    "GenICamError",
    "GeneralError",
    "GigEError",
    "HikrobotError",
    "IPConflictError",
    "IncompleteImageError",
    "InsufficientBufferError",
    "IntRange",
    "InvalidHandleError",
    "NetworkError",
    "NoDataError",
    "NodeAccessError",
    "NodeMap",
    "NotImplementedByDeviceError",
    "NotSupportedError",
    "PacketError",
    "ParameterError",
    "PreconditionError",
    "SDKLoadError",
    "SDKNotFoundError",
    "StatusError",
    "TransportStats",
    "USBBandwidthError",
    "USBError",
    "UnsupportedPixelFormatError",
    "UnsupportedPlatformError",
    "UpgradeError",
    "ValueOutOfRangeError",
    "__version__",
    "enumerate_devices",
]

from ._errors import (
    AccessDeniedError,
    BufferInUseError,
    BufferReleasedError,
    CallOrderError,
    CameraStateError,
    CameraTimeoutError,
    DeviceBusyError,
    GeneralError,
    GenICamError,
    GigEError,
    HikrobotError,
    IncompleteImageError,
    InsufficientBufferError,
    InvalidHandleError,
    IPConflictError,
    NetworkError,
    NoDataError,
    NodeAccessError,
    NotImplementedByDeviceError,
    NotSupportedError,
    PacketError,
    ParameterError,
    PreconditionError,
    SDKLoadError,
    SDKNotFoundError,
    StatusError,
    UnsupportedPixelFormatError,
    UnsupportedPlatformError,
    UpgradeError,
    USBBandwidthError,
    USBError,
    ValueOutOfRangeError,
)
from .camera import Camera, TransportStats
from .device import DeviceInfo, enumerate_devices
from .frame import Frame
from .nodes import FloatRange, IntRange, NodeMap
