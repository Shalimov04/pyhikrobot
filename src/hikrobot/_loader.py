# SPDX-License-Identifier: Apache-2.0
"""Locating and opening the vendor MVS shared library.

Nothing here runs at import time. :func:`load` is called on first real use and caches the
opened library for the lifetime of the process.

Resolution is split into a pure part (:func:`candidate_paths`, which only builds a search
list from an environment mapping) and an impure part (:func:`load`, which touches the
filesystem and calls ``dlopen``). Tests exercise the pure part against a temporary
directory posing as an SDK tree.

Platform notes that are easy to get wrong:

* Windows needs ``WinDLL`` (``__stdcall``). ``CDLL`` links and appears to work, then
  corrupts the stack on the first call that takes more than a couple of arguments.
* Linux needs ``RTLD_GLOBAL``. Without it the SDK's GenICam plugins cannot resolve each
  other's symbols and the process segfaults inside ``MV_CC_CreateHandle``.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._errors import SDKLoadError, SDKNotFoundError, UnsupportedPlatformError

__all__ = ["candidate_paths", "find_library", "load", "reset"]

WINDOWS_LIBRARY_NAME = "MvCameraControl.dll"
LINUX_LIBRARY_NAME = "libMvCameraControl.so"

#: ``platform.machine()`` -> directory name under ``$MVCAM_SDK_PATH/lib`` on Linux.
#: An unlisted architecture is an error rather than a guess.
LINUX_ARCH_DIRS = {
    "x86_64": "64",
    "amd64": "64",
    "i386": "32",
    "i686": "32",
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "armv7l": "arm",
    "armv7": "arm",
    "armhf": "arm",
}

_DEFAULT_LINUX_ROOT = "/opt/MVS"
_DEFAULT_WINDOWS_COMMON = r"C:\Program Files (x86)\Common Files"

_library: ctypes.CDLL | None = None
_dll_directory_cookie: Any = None


def _linux_arch_dir(machine: str) -> str:
    try:
        return LINUX_ARCH_DIRS[machine.lower()]
    except KeyError:
        raise UnsupportedPlatformError(
            f"no known MVS library directory for CPU architecture {machine!r}; "
            f"known architectures: {', '.join(sorted(set(LINUX_ARCH_DIRS)))}"
        ) from None


def _windows_arch_dir(pointer_bits: int) -> str:
    # The interpreter's pointer size decides this, not the OS: a 32-bit Python on 64-bit
    # Windows must load the 32-bit library.
    return "Win64_x64" if pointer_bits == 64 else "Win32_i86"


def _unique(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path).lower() if os.name == "nt" else str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _windows_candidates(env: Mapping[str, str], machine: str, pointer_bits: int) -> list[Path]:
    if machine.lower() in {"arm64", "aarch64"}:
        raise UnsupportedPlatformError(
            "the MVS SDK does not ship a Windows build for ARM64; x86 or x86-64 Windows is required"
        )
    arch_dir = _windows_arch_dir(pointer_bits)
    lib_dir = "win64" if pointer_bits == 64 else "win32"

    roots: list[Path] = []
    for var in ("MVCAM_SDK_PATH", "MVCAM_COMMON_RUNENV"):
        value = env.get(var)
        if value:
            roots.append(Path(value))
    for var in ("CommonProgramFiles(x86)", "CommonProgramW6432", "CommonProgramFiles"):
        value = env.get(var)
        if value:
            roots.append(Path(value) / "MVS" / "Runtime")
    roots.append(Path(_DEFAULT_WINDOWS_COMMON) / "MVS" / "Runtime")

    candidates: list[Path] = []
    for root in roots:
        # The installer keeps the importable DLLs under Common Files\MVS\Runtime\<arch>,
        # while MVCAM_SDK_PATH / MVCAM_COMMON_RUNENV point at the Development tree; try
        # both shapes rather than assuming which variable is set.
        candidates.extend(
            [
                root / arch_dir / WINDOWS_LIBRARY_NAME,
                root / "Runtime" / arch_dir / WINDOWS_LIBRARY_NAME,
                root / "Libraries" / lib_dir / WINDOWS_LIBRARY_NAME,
                root / "bin" / WINDOWS_LIBRARY_NAME,
                root / WINDOWS_LIBRARY_NAME,
            ]
        )
    return _unique(candidates)


def _linux_candidates(env: Mapping[str, str], machine: str) -> list[Path]:
    arch_dir = _linux_arch_dir(machine)

    roots: list[Path] = []
    value = env.get("MVCAM_SDK_PATH")
    if value:
        roots.append(Path(value))
    value = env.get("MVCAM_COMMON_RUNENV")
    if value:
        roots.append(Path(value))
    roots.append(Path(_DEFAULT_LINUX_ROOT))

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "lib" / arch_dir / LINUX_LIBRARY_NAME,
                root / arch_dir / LINUX_LIBRARY_NAME,
                root / "lib" / LINUX_LIBRARY_NAME,
                root / LINUX_LIBRARY_NAME,
            ]
        )
    return _unique(candidates)


def candidate_paths(
    env: Mapping[str, str] | None = None,
    system: str | None = None,
    machine: str | None = None,
    pointer_bits: int | None = None,
) -> list[Path]:
    """Build the ordered list of paths that would be probed for the MVS library.

    Pure: touches neither the filesystem nor the process environment. The arguments exist
    so that tests can describe a platform they are not running on.

    Args:
        env: Environment mapping to read; defaults to ``os.environ``.
        system: ``platform.system()`` value; defaults to the running system.
        machine: ``platform.machine()`` value; defaults to the running CPU.
        pointer_bits: Interpreter pointer size, 32 or 64; defaults to this interpreter.

    Returns:
        Candidate paths, most specific first. May be empty on no platform; unsupported
        platforms raise instead.

    Raises:
        UnsupportedPlatformError: The OS/CPU combination has no known SDK layout.
    """
    env = os.environ if env is None else env
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    pointer_bits = (64 if sys.maxsize > 2**32 else 32) if pointer_bits is None else pointer_bits

    if system == "Windows":
        return _windows_candidates(env, machine, pointer_bits)
    if system == "Linux":
        return _linux_candidates(env, machine)
    raise UnsupportedPlatformError(
        f"the MVS SDK is not available for {system!r}; Linux and Windows are supported"
    )


def find_library(paths: list[Path] | None = None) -> Path | None:
    """Return the first candidate path that exists, or ``None``.

    Args:
        paths: Paths to probe; defaults to :func:`candidate_paths` for this platform.

    Returns:
        The path that would be opened, or ``None`` if none of them exist. ``None`` does
        not mean the SDK is absent - the library may still be reachable by bare name
        through ``PATH`` or the dynamic linker cache.
    """
    if paths is None:
        paths = candidate_paths()
    return next((path for path in paths if path.is_file()), None)


def _prepare_linux_env(env: Any, library_path: Path | None) -> None:
    """Point ``MVCAM_COMMON_RUNENV`` at the library directory when the user has not.

    The SDK reads this variable to find its GenICam XML and plugin tree; leaving it unset
    produces a failure deep inside the vendor library rather than at load time.
    """
    if env.get("MVCAM_COMMON_RUNENV"):
        return
    if library_path is None:
        env["MVCAM_COMMON_RUNENV"] = str(Path(_DEFAULT_LINUX_ROOT) / "lib")
    else:
        env["MVCAM_COMMON_RUNENV"] = str(library_path.parent)


def _load() -> ctypes.CDLL:
    """Locate and open the MVS shared library.

    Separate from :func:`load` so that tests can substitute a fake library at the CDLL
    boundary, keeping struct packing and ``argtypes`` under test.

    Raises:
        SDKNotFoundError: No candidate path exists and the loader's fallback by bare
            library name also failed.
        SDKLoadError: A library was found but could not be opened.
        UnsupportedPlatformError: The OS/CPU combination has no known SDK layout.
    """
    candidates = candidate_paths()
    found = find_library(candidates)

    system = platform.system()
    bare_name = WINDOWS_LIBRARY_NAME if system == "Windows" else LINUX_LIBRARY_NAME

    if system == "Linux":
        _prepare_linux_env(os.environ, found)

    if found is not None:
        try:
            return _open(found)
        except OSError as exc:
            raise SDKLoadError(
                f"found the MVS library at {found} but could not open it: {exc}"
            ) from exc

    # Last resort: let the platform loader search PATH / ld.so cache. The MVS installer
    # puts its runtime directory on PATH, so this succeeds on stock Windows installs even
    # when none of the candidate directories match.
    try:
        return _open(Path(bare_name))
    except OSError:
        searched = "\n  ".join(str(path) for path in candidates)
        raise SDKNotFoundError(
            "the Hikrobot MVS SDK was not found. Install it from the vendor and, if it "
            "lives outside the default location, set MVCAM_SDK_PATH to its root.\n"
            f"Searched:\n  {searched}\n  {bare_name} (via the system library path)"
        ) from None


def _open(path: Path) -> ctypes.CDLL:
    """Open one library path with the calling convention this platform requires."""
    global _dll_directory_cookie
    if sys.platform == "win32":
        if path.is_absolute() and _dll_directory_cookie is None:
            # Python 3.8+ no longer searches PATH for a loaded DLL's own dependencies;
            # MvCameraControl.dll needs its GenICam siblings from the same directory.
            # The cookie is kept for the process lifetime on purpose - closing it would
            # break the SDK's later, lazy plugin loads.
            _dll_directory_cookie = os.add_dll_directory(str(path.parent))
        # WinDLL, not CDLL: the SDK is __stdcall and CDLL corrupts the stack.
        return ctypes.WinDLL(str(path))
    else:
        # RTLD_GLOBAL, or the GenICam plugins fail to resolve each other and segfault.
        return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def load() -> ctypes.CDLL:
    """Return the opened MVS shared library, loading it on first call.

    Raises:
        SDKNotFoundError: The SDK is not installed or is installed somewhere unexpected.
        SDKLoadError: The library was found but could not be opened.
        UnsupportedPlatformError: The OS/CPU combination has no known SDK layout.
    """
    global _library
    if _library is None:
        _library = _load()
    return _library


def reset() -> None:
    """Drop the cached library handle.

    For tests only. The underlying library stays loaded in the process; this just forces
    the next :func:`load` to go through :func:`_load` again.
    """
    global _library
    _library = None
