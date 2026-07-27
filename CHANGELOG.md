# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-27

First release. Enumeration, open/close, the GenICam node map, streaming with enforced buffer
lifetime, and GigE transport tuning — all exercised against an MV-CS023-10GM. Action commands
and the CUDA layer are not implemented yet, and the public API may still change.

### Added

- README with the public API, the buffer-lifetime rules and the GigE transport notes. Every
  example is verified against a real camera; `ruff format` also parses the Python blocks, so a
  broken snippet fails lint.
- CI: a `package` job that builds the wheel, installs it into a clean environment and imports it
  from `site-packages` — the check the `src/` layout exists to make possible.

- Repository skeleton: `src/` layout, hatchling build, ruff/mypy/pytest configuration, CI matrix
  (ubuntu-latest + windows-latest, unit tests only).
- `_loader`: lazy MVS library discovery and loading across Linux and Windows, with the
  `MVCAM_SDK_PATH` / `MVCAM_COMMON_RUNENV` / installer-default search order, the
  architecture-directory map, `WinDLL` on Windows and `RTLD_GLOBAL` on Linux.
- `_errors`: `HikrobotError` base with `SDKNotFoundError`, `SDKLoadError` and
  `UnsupportedPlatformError`.
- `_errors`: all 55 `MV_E_*` status codes of `MvErrorDefine.h` transcribed by hand, mapped onto a
  `StatusError` hierarchy — one class per vendor range (`GeneralError`, `GenICamError`, `GigEError`,
  `USBError`, `UpgradeError`) plus named classes for the codes callers act on. Codes are normalised
  from the signed `int` the SDK returns, so `restype` may be `c_int` or `c_uint`.
- `_errors`: the `check` / `call` helpers, the only sanctioned way to invoke an SDK entry point.
  `call(..., allow=...)` covers statuses that are expected at a given call site, which keeps bare
  `if ret != 0` out of the package.
- `_ctypes_defs`: the structures the enumerate/open/grab path needs — `MV_CC_DEVICE_INFO` with its
  six-member transport union, `MV_CC_DEVICE_INFO_LIST`, `MV_FRAME_OUT_INFO_EX`, `MV_FRAME_OUT` —
  transcribed by hand from `CameraParams.h`, with the transport-layer and access-mode constants.
- `_ctypes_defs`: `argtypes`/`restype` for the twelve entry points used so far, applied through
  `sdk()`, which loads the library once and declares every prototype before first use.
- `device`: `enumerate_devices()` and the `DeviceInfo` record. Each record holds its own copy of
  the vendor structure, so it stays valid after a later enumeration has reused the SDK's own.
- `camera`: `Camera` with `open()` / `close()`, context-manager support, `is_open`,
  `is_connected` and the access-mode table. A failed open destroys the handle it had already
  created; a failed close still destroys it and still forgets it.
- `CameraStateError` for state mismatches the wrapper catches before the SDK does.
- Tests: a fake MVS library at the CDLL boundary, so struct packing, `argtypes` and the
  status-to-exception mapping stay under test without hardware.
- `nodes`: `NodeMap` with typed access to integer, float, boolean, enumeration, string and
  command nodes, plus the `IntRange` / `FloatRange` records. Enumerations go through the `Ex`
  entry point, whose 256-entry list is not capped at 64 like the older one.
- `frame`: `Frame` — a zero-copy, read-only NumPy view onto a driver node, with metadata
  snapshotted so it outlives the buffer. `data` and `copy()` raise `BufferReleasedError` once the
  node is gone; bit-packed formats raise `UnsupportedPixelFormatError` from `data` and stay
  reachable through `raw_bytes`.
- `camera`: `start_grabbing()` / `stop_grabbing()` / `is_grabbing`, and the `frames()` and
  `frames_raw()` iterators. `frames()` releases each node in a `finally`; `frames_raw()` hands the
  release to the caller. Stopping acquisition reclaims any frame still outstanding, because the
  pool is torn down with it.
- `BufferReleasedError` and `UnsupportedPixelFormatError`.
- `camera`: GigE transport tuning — `packet_size`, `packet_delay`, `optimal_packet_size`,
  `tune_packet_size()`, `enable_resend()` and the `statistics` counters (`TransportStats`).
  `MV_CC_GetOptimalPacketSize` returns the size rather than a status and reports failure by
  returning an `MV_E_*` code, which the wrapper unpicks.
- `camera`: `Camera.nodes` and `Camera.handle`, plus named properties over the SFNC features -
  `width`, `height`, `offset_x`, `offset_y`, `payload_size`, `exposure_us`, `gain_db`,
  `frame_rate`, `pixel_format`, `pixel_formats`, and the matching range accessors.
