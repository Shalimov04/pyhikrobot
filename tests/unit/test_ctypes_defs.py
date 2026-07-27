# SPDX-License-Identifier: Apache-2.0
"""Layout of the transcribed MVS structures and the entry-point declarations.

No SDK, no camera. The expected sizes and offsets are computed by hand from the C headers,
so a transposed field or a wrong integer width fails here instead of corrupting a device
list at run time.

The size assertions assume an LP64/LLP64 target - the platforms this package targets. On a
32-bit build pointer members shrink and, on 32-bit Linux, ``int64_t`` needs only 4-byte
alignment, so the numbers below would not hold.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from typing import Any

import pytest

from hikrobot import _ctypes_defs, _loader
from hikrobot._ctypes_defs import (
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_CXP_DEVICE_INFO,
    MV_FRAME_OUT,
    MV_FRAME_OUT_INFO_EX,
    MV_GIGE_DEVICE_INFO,
    MV_MAX_DEVICE_NUM,
    MV_USB3_DEVICE_INFO,
    PROTOTYPES,
    MV_CamL_DEV_INFO,
    apply_prototypes,
    sdk,
)

pytestmark = pytest.mark.skipif(
    ctypes.sizeof(ctypes.c_void_p) != 8,
    reason="the hand-computed layout assumes a 64-bit build",
)


class FakeFunction:
    def __init__(self) -> None:
        self.argtypes: Any = None
        self.restype: Any = None


class FakeLibrary:
    """Stands in at the CDLL boundary: any symbol resolves to a settable function."""

    def __init__(self, missing: str | None = None) -> None:
        self._missing = missing
        self.functions: dict[str, FakeFunction] = {}

    def __getattr__(self, name: str) -> FakeFunction:
        if name == self._missing:
            raise AttributeError(name)
        return self.functions.setdefault(name, FakeFunction())


class TestStructureSizes:
    @pytest.mark.parametrize(
        ("structure", "size"),
        [
            # 5 uints + 32 + 32 + 32 + 48 + 16 + 16 bytes + uint + 4 uints.
            (MV_GIGE_DEVICE_INFO, 216),
            # 4 bytes + 2 ushorts + uint + 8 * 64 bytes + 2 uints + 2 uints.
            (MV_USB3_DEVICE_INFO, 540),
            # 6 * 64 bytes + 38 uints.
            (MV_CamL_DEV_INFO, 536),
            # 8 * 64 bytes + 7 uints; the CML and XoF structures share this layout.
            (MV_CXP_DEVICE_INFO, 540),
        ],
    )
    def test_transport_specific_info(self, structure: type[ctypes.Structure], size: int) -> None:
        assert ctypes.sizeof(structure) == size

    def test_device_info_union_is_as_large_as_its_biggest_member(self) -> None:
        # 32 bytes of header plus the union, whose size is the 540-byte USB3/CXP member.
        # Getting this wrong makes MV_CC_EnumDevices write past the end of the list.
        assert ctypes.sizeof(MV_CC_DEVICE_INFO) == 32 + 540

    def test_device_info_list(self) -> None:
        # uint, 4 bytes of padding before the pointer array, then 256 pointers.
        assert ctypes.sizeof(MV_CC_DEVICE_INFO_LIST) == 8 + MV_MAX_DEVICE_NUM * 8

    def test_frame_out_info(self) -> None:
        assert ctypes.sizeof(MV_FRAME_OUT_INFO_EX) == 256

    def test_frame_out(self) -> None:
        # Pointer, the 256-byte info block, 16 reserved uints.
        assert ctypes.sizeof(MV_FRAME_OUT) == 8 + 256 + 64


class TestCriticalOffsets:
    @pytest.mark.parametrize(
        ("field", "offset"),
        [
            ("nWidth", 0),
            ("nHeight", 2),
            ("enPixelType", 4),
            ("nFrameNum", 8),
            # nReserved0 exists purely to 8-byte-align this member.
            ("nHostTimeStamp", 24),
            ("nFrameLen", 32),
            ("nOffsetX", 88),
            ("nUnparsedChunkNum", 100),
            ("UnparsedChunkList", 104),
            ("nExtendWidth", 112),
            ("nFrameLenEx", 120),
        ],
    )
    def test_frame_info_field_offsets(self, field: str, offset: int) -> None:
        assert getattr(MV_FRAME_OUT_INFO_EX, field).offset == offset

    def test_frame_info_starts_after_the_buffer_pointer(self) -> None:
        assert MV_FRAME_OUT.stFrameInfo.offset == 8

    def test_special_info_follows_the_fixed_header(self) -> None:
        assert MV_CC_DEVICE_INFO.SpecialInfo.offset == 32

    def test_device_pointers_follow_the_count(self) -> None:
        assert MV_CC_DEVICE_INFO_LIST.pDeviceInfo.offset == 8


class TestFieldWidths:
    def test_pixel_type_is_unsigned(self) -> None:
        # PixelType_Gvsp_Undefined is 0xFFFFFFFF and custom formats set bit 31. As a
        # signed field they would all read back negative.
        info = MV_FRAME_OUT_INFO_EX()
        info.enPixelType = 0xFFFFFFFF
        assert info.enPixelType == 0xFFFFFFFF

    def test_host_timestamp_is_signed_64_bit(self) -> None:
        info = MV_FRAME_OUT_INFO_EX()
        info.nHostTimeStamp = -1
        assert info.nHostTimeStamp == -1

    def test_frame_length_ex_is_unsigned_64_bit(self) -> None:
        info = MV_FRAME_OUT_INFO_EX()
        info.nFrameLenEx = 2**40
        assert info.nFrameLenEx == 2**40

    def test_char_arrays_hold_raw_bytes(self) -> None:
        # The vendor declares these as unsigned char, and the contents are not guaranteed
        # to be valid UTF-8; decoding is the public layer's problem.
        info = MV_GIGE_DEVICE_INFO()
        info.chSerialNumber[0] = 0xFF
        assert bytes(info.chSerialNumber)[0] == 0xFF


class TestPrototypes:
    def test_every_entry_point_gets_argtypes_and_restype(self) -> None:
        lib = FakeLibrary()
        apply_prototypes(lib)

        assert set(lib.functions) == {symbol for symbol, _, _ in PROTOTYPES}
        for symbol, argtypes, restype in PROTOTYPES:
            assert lib.functions[symbol].argtypes == argtypes
            assert lib.functions[symbol].restype is restype

    def test_no_entry_point_is_left_with_a_default_restype(self) -> None:
        for symbol, _, restype in PROTOTYPES:
            assert restype is not None, symbol

    def test_prototype_symbols_are_unique(self) -> None:
        symbols = [symbol for symbol, _, _ in PROTOTYPES]
        assert len(symbols) == len(set(symbols))

    def test_missing_symbol_is_reported(self) -> None:
        lib = FakeLibrary(missing="MV_CC_GetImageBuffer")
        with pytest.raises(AttributeError, match="MV_CC_GetImageBuffer"):
            apply_prototypes(lib)

    def test_grab_entry_points_take_a_frame_pointer(self) -> None:
        # The one argument that must not degrade to a plain integer.
        by_symbol = {symbol: argtypes for symbol, argtypes, _ in PROTOTYPES}
        assert by_symbol["MV_CC_GetImageBuffer"][1] is ctypes.POINTER(MV_FRAME_OUT)
        assert by_symbol["MV_CC_FreeImageBuffer"][1] is ctypes.POINTER(MV_FRAME_OUT)


class TestSdk:
    def test_library_is_configured_once_and_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lib = FakeLibrary()
        loads: list[int] = []

        def fake_load() -> FakeLibrary:
            loads.append(1)
            return lib

        monkeypatch.setattr(_loader, "load", fake_load)
        assert sdk() is lib
        assert sdk() is lib
        assert len(loads) == 1
        assert lib.functions["MV_CC_EnumDevices"].restype is ctypes.c_int

    def test_reset_forces_a_reload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_loader, "load", FakeLibrary)
        first = sdk()
        _ctypes_defs.reset()
        assert sdk() is not first


def test_importing_the_module_does_not_load_the_sdk() -> None:
    # Structures and prototypes are built at import time; the library is not. A fresh
    # interpreter, because an earlier test in this process could have loaded it.
    code = (
        "import hikrobot._ctypes_defs as defs, hikrobot._loader as loader;"
        "assert defs._sdk is None, 'the SDK was configured at import time';"
        "assert loader._library is None, 'the SDK was loaded at import time';"
        "print(len(defs.PROTOTYPES))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(len(PROTOTYPES))
