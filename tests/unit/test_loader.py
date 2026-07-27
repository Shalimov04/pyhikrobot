# SPDX-License-Identifier: Apache-2.0
"""Loader tests: path resolution, architecture mapping, failure modes.

No SDK, no camera. Platforms the test machine is not running on are described through
the explicit ``system`` / ``machine`` / ``pointer_bits`` arguments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hikrobot import SDKLoadError, SDKNotFoundError, UnsupportedPlatformError, _loader
from hikrobot._loader import (
    LINUX_LIBRARY_NAME,
    WINDOWS_LIBRARY_NAME,
    candidate_paths,
    find_library,
    load,
    reset,
)


def linux_paths(env: dict[str, str], machine: str = "x86_64") -> list[Path]:
    return candidate_paths(env=env, system="Linux", machine=machine)


def windows_paths(env: dict[str, str], pointer_bits: int = 64) -> list[Path]:
    return candidate_paths(env=env, system="Windows", machine="AMD64", pointer_bits=pointer_bits)


class TestLinuxCandidates:
    def test_defaults_to_opt_mvs(self) -> None:
        paths = linux_paths({})
        assert Path("/opt/MVS/lib/64") / LINUX_LIBRARY_NAME in paths

    def test_sdk_path_wins_over_default(self) -> None:
        paths = linux_paths({"MVCAM_SDK_PATH": "/srv/mvs"})
        assert paths[0] == Path("/srv/mvs/lib/64") / LINUX_LIBRARY_NAME
        assert Path("/opt/MVS/lib/64") / LINUX_LIBRARY_NAME in paths

    def test_runenv_is_searched_too(self) -> None:
        paths = linux_paths({"MVCAM_COMMON_RUNENV": "/srv/mvs/lib"})
        assert Path("/srv/mvs/lib") / LINUX_LIBRARY_NAME in paths

    @pytest.mark.parametrize(
        ("machine", "arch_dir"),
        [
            ("x86_64", "64"),
            ("aarch64", "aarch64"),
            ("armv7l", "arm"),
            ("i686", "32"),
        ],
    )
    def test_arch_directory_mapping(self, machine: str, arch_dir: str) -> None:
        paths = linux_paths({"MVCAM_SDK_PATH": "/opt/MVS"}, machine=machine)
        assert paths[0] == Path("/opt/MVS/lib") / arch_dir / LINUX_LIBRARY_NAME

    def test_unknown_arch_is_an_error_not_a_guess(self) -> None:
        with pytest.raises(UnsupportedPlatformError, match="riscv64"):
            linux_paths({}, machine="riscv64")

    def test_no_duplicates(self) -> None:
        paths = linux_paths({"MVCAM_SDK_PATH": "/opt/MVS", "MVCAM_COMMON_RUNENV": "/opt/MVS"})
        assert len(paths) == len(set(paths))


class TestWindowsCandidates:
    def test_common_files_runtime_is_searched(self) -> None:
        paths = windows_paths({"CommonProgramFiles(x86)": r"C:\Program Files (x86)\Common Files"})
        assert (
            Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64")
            / WINDOWS_LIBRARY_NAME
            in paths
        )

    def test_32_bit_interpreter_gets_the_32_bit_library(self) -> None:
        paths = windows_paths({"CommonProgramFiles(x86)": r"C:\CF"}, pointer_bits=32)
        assert Path(r"C:\CF\MVS\Runtime\Win32_i86") / WINDOWS_LIBRARY_NAME in paths
        assert all("Win64_x64" not in str(path) for path in paths)

    def test_sdk_path_root_expands_to_runtime_subdir(self) -> None:
        # The installer points MVCAM_SDK_PATH / MVCAM_COMMON_RUNENV at the Development
        # tree, which holds no DLL; both the plain and the Runtime\<arch> shape are tried.
        paths = windows_paths({"MVCAM_SDK_PATH": r"C:\MVS\Development"})
        assert Path(r"C:\MVS\Development\Runtime\Win64_x64") / WINDOWS_LIBRARY_NAME in paths
        assert Path(r"C:\MVS\Development\Win64_x64") / WINDOWS_LIBRARY_NAME in paths

    def test_windows_on_arm_is_unsupported(self) -> None:
        with pytest.raises(UnsupportedPlatformError, match="ARM64"):
            candidate_paths(env={}, system="Windows", machine="ARM64", pointer_bits=64)


def test_unsupported_operating_system() -> None:
    with pytest.raises(UnsupportedPlatformError, match="Darwin"):
        candidate_paths(env={}, system="Darwin", machine="arm64")


class TestFindLibrary:
    def test_picks_the_first_existing_candidate(self, tmp_path: Path) -> None:
        root = tmp_path / "MVS"
        lib = root / "lib" / "64" / LINUX_LIBRARY_NAME
        lib.parent.mkdir(parents=True)
        lib.write_bytes(b"not really an ELF")

        paths = linux_paths({"MVCAM_SDK_PATH": str(root)})
        assert find_library(paths) == lib

    def test_returns_none_when_nothing_exists(self, tmp_path: Path) -> None:
        assert find_library(linux_paths({"MVCAM_SDK_PATH": str(tmp_path / "absent")})) is None

    def test_defaults_to_this_platforms_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probed: list[Path] = []
        monkeypatch.setattr(_loader, "candidate_paths", lambda: probed)
        assert find_library() is None
        assert probed == []

    def test_ignores_a_directory_of_the_right_name(self, tmp_path: Path) -> None:
        root = tmp_path / "MVS"
        (root / "lib" / "64" / LINUX_LIBRARY_NAME).mkdir(parents=True)
        assert find_library(linux_paths({"MVCAM_SDK_PATH": str(root)})) is None


class TestPrepareLinuxEnv:
    def test_defaults_runenv_to_the_library_directory(self) -> None:
        env: dict[str, str] = {}
        _loader._prepare_linux_env(env, Path("/srv/mvs/lib/aarch64/") / LINUX_LIBRARY_NAME)
        assert env["MVCAM_COMMON_RUNENV"] == str(Path("/srv/mvs/lib/aarch64"))

    def test_falls_back_to_the_default_root_when_nothing_was_found(self) -> None:
        # No library on disk, but the SDK may still be reachable by bare name; it still
        # needs the variable pointed somewhere sane before the load is attempted.
        env: dict[str, str] = {}
        _loader._prepare_linux_env(env, None)
        assert env["MVCAM_COMMON_RUNENV"] == str(Path("/opt/MVS/lib"))

    def test_does_not_override_the_user(self) -> None:
        env = {"MVCAM_COMMON_RUNENV": "/custom"}
        _loader._prepare_linux_env(env, Path("/opt/MVS/lib/64") / LINUX_LIBRARY_NAME)
        assert env["MVCAM_COMMON_RUNENV"] == "/custom"


class TestLoad:
    def test_missing_sdk_raises_with_actionable_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def refuse(path: Path) -> object:
            raise OSError("no such file")

        # Set so that _prepare_linux_env leaves the real environment alone.
        monkeypatch.setenv("MVCAM_COMMON_RUNENV", str(tmp_path))
        monkeypatch.setattr(_loader, "candidate_paths", lambda: [tmp_path / "nowhere"])
        monkeypatch.setattr(_loader, "_open", refuse)

        with pytest.raises(SDKNotFoundError) as excinfo:
            load()
        message = str(excinfo.value)
        assert "MVCAM_SDK_PATH" in message
        assert str(tmp_path / "nowhere") in message

    def test_found_but_unopenable_library_is_a_load_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        lib = tmp_path / LINUX_LIBRARY_NAME
        lib.write_bytes(b"truncated")

        def refuse(path: Path) -> object:
            raise OSError("wrong ELF class")

        monkeypatch.setenv("MVCAM_COMMON_RUNENV", str(tmp_path))
        monkeypatch.setattr(_loader, "candidate_paths", lambda: [lib])
        monkeypatch.setattr(_loader, "_open", refuse)

        with pytest.raises(SDKLoadError, match="wrong ELF class"):
            load()

    def test_library_is_loaded_once_and_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def fake_load() -> object:
            calls.append(1)
            return object()

        monkeypatch.setattr(_loader, "_load", fake_load)
        first = load()
        assert load() is first
        assert len(calls) == 1

        reset()
        assert load() is not first
