# pyhikrobot

Thin, zero-copy Python bindings for Hikrobot machine-vision cameras over the MVS SDK.
NumPy views without the copy, a cross-platform loader, and errors you can catch.

> **Status: alpha.** Enumeration, open/close, the GenICam node map, streaming and GigE
> transport tuning are implemented and tested against real hardware. Action commands and
> the CUDA layer are not written yet. The public API may still change.

---

## Requirements

| | |
|---|---|
| Python | 3.9+ |
| Runtime dependency | `numpy` — nothing else, ever, in the core package |
| SDK | Hikrobot MVS 4.x, installed separately by you |
| Platforms | Linux `x86_64` / `aarch64` / `armv7l`, Windows x64 |

This package **does not ship any Hikrobot code**. It opens an SDK you installed yourself;
get it from the vendor under their terms.

## Install

Not published to PyPI yet — install from the repository:

```bash
pip install git+https://github.com/Shalimov04/pyhikrobot
pip install "pyhikrobot[cuda] @ git+https://github.com/Shalimov04/pyhikrobot"
```

The `cuda` extra pulls in CuPy for the device-side layer, which is not written yet.

The SDK is found through `MVCAM_SDK_PATH`, falling back to `/opt/MVS` on Linux and the
installer's `Common Files\MVS\Runtime` on Windows. Nothing is loaded at import time, so
`import hikrobot` works on a machine with no SDK and no camera — a missing SDK raises
`SDKNotFoundError` on first real use.

## Quick start

```python
import hikrobot

devices = hikrobot.enumerate_devices()
for device in devices:
    print(device.transport, device.model_name, device.serial_number, device.ip_address)

with hikrobot.Camera(devices[0]) as camera:
    camera.exposure_us = 2500.0
    camera.gain_db = 0.0

    for frame in camera.frames(timeout_ms=1000):
        print(frame.frame_number, frame.data.shape, frame.data.mean())
        break
```

`Camera()` touches no SDK state; the context manager opens and closes the device.

## Buffer lifetime

**This is the one rule that matters.** `frame.data` is a read-only NumPy view onto a node
of the driver's buffer pool — not memory you own. The pool is fixed and recycled, so once
the node goes back the driver writes the *next* frame into the same address.

```python
for frame in camera.frames(timeout_ms=1000):
    total = frame.data.sum()  # fine — inside the body
    keep = frame.copy()  # fine — owns its memory, writable
    leaked = frame.data  # a view; its node goes back at the end of the body
    break

frame.data  # BufferReleasedError — the usual mistake, caught
leaked.mean()  # no error, and no longer this frame's pixels
```

Note the asymmetry. Reaching for `frame.data` after the release raises, which catches the
common mistake in development instead of in the field. An array you already took out
cannot be caught — NumPy has no idea the memory changed hands, so it keeps working and
quietly returns whatever the driver wrote there next. That is the failure this API is
shaped to avoid, and `.copy()` is the only thing that avoids it.

`frames()` releases the node in a `finally`, on every route out including `break`,
`return` and exceptions. There is deliberately no "hold this one for me" shortcut: a
`.copy()` of a 2 MB frame is visible in a profile, an accidentally retained view is not.

For consumers that outlive the loop body, `frames_raw()` hands the release over:

```python
camera.start_grabbing(node_count=8)  # own acquisition, or leaving the loop stops it
try:
    for frame in camera.frames_raw(timeout_ms=1000):
        queue.put(frame)  # released later, by whoever drains the queue
finally:
    camera.stop_grabbing()  # reclaims anything still outstanding
```

The pool holds exactly `node_count` nodes and **the SDK's default is one**, so holding a
second frame without raising it fails with `InsufficientBufferError`.

## Camera settings

Named properties cover the common SFNC features:

```python
camera.width, camera.height, camera.offset_x, camera.offset_y
camera.exposure_us, camera.gain_db, camera.frame_rate
camera.pixel_format, camera.pixel_formats
camera.payload_size  # read-only
camera.exposure_range_us.min  # FloatRange(value, min, max)
camera.nodes.int_range("Width").inc  # IntRange(value, min, max, inc)
```

Anything else goes through the node map directly:

```python
camera.nodes.set_enum("TriggerMode", "On")
camera.nodes.set_enum("TriggerSource", "Software")
camera.nodes.execute("TriggerSoftware")
camera.nodes.get_int("GevTimestampTickFrequency")
```

Two behaviours worth knowing, both measured rather than assumed:

- A missing node and a wrong-typed access return the **same** status, so both raise
  `GenICamError`. When one appears, check the spelling *and* the type.
- Float features are quantised to a hardware step the node map does not expose. Writing
  `gain_db = 1.0` reads back as `1.0052`. Write, then read, and trust the second value.

## GigE transport

The knobs that decide whether streaming works at all:

```python
camera.tune_packet_size()  # probe the path, apply what it carries
camera.enable_resend(True)  # retransmit packets the host missed

camera.start_grabbing(node_count=4)
for frame in camera.frames(timeout_ms=5000):
    ...
    stats = camera.statistics  # only valid while acquisition runs
camera.stop_grabbing()

stats.lost_packets, stats.lost_frames, stats.resent_packets
```

Incomplete frames almost always mean a packet size the network path cannot carry — start
with `tune_packet_size()`. Counters live only between `start_grabbing()` and
`stop_grabbing()` and reset on every start.

## Errors

Everything derives from `HikrobotError`, so you can catch one thing. Below it, one class
per vendor error range, plus named classes for the codes callers actually branch on:

```text
HikrobotError
├── SDKNotFoundError, SDKLoadError, UnsupportedPlatformError
├── CameraStateError, BufferReleasedError, UnsupportedPixelFormatError
└── StatusError                      .status  .name  .operation
    ├── GeneralError                 InvalidHandleError, CallOrderError, NoDataError,
    │                                IncompleteImageError, InsufficientBufferError, …
    ├── GenICamError                 CameraTimeoutError, ValueOutOfRangeError, …
    ├── GigEError                    AccessDeniedError, DeviceBusyError, NetworkError, …
    ├── USBError
    └── UpgradeError
```

An untranslated code still arrives as its range's class carrying `.status` and `.name`, so
nothing is lost:

```python
try:
    camera.open()
except hikrobot.AccessDeniedError:
    ...  # held by another process
except hikrobot.HikrobotError as exc:
    print(exc)  # MV_CC_OpenDevice failed: MV_E_NETER (0x80000206) - network error
```

## Development

```bash
pip install -e ".[dev]"

pytest tests/unit              # no SDK, no camera, no CUDA — runs anywhere
pytest tests --hardware        # opt-in; needs one reachable camera
ruff check . && ruff format --check . && mypy
```

Unit tests run against a fake that sits at the `CDLL` boundary, so struct packing,
`argtypes` and the status-to-exception mapping stay under test without hardware.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Hikrobot and MVS are trademarks of Hangzhou Hikrobot Co., Ltd. This project is not
affiliated with or endorsed by them, and distributes none of their software.
