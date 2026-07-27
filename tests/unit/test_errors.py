# SPDX-License-Identifier: Apache-2.0
"""Status-code mapping and the checked-call helper.

No SDK, no camera. The transcription itself is under test here: a code lost while copying
from the header is a silent hole in the error mapping, so the table is checked for
self-consistency as well as for behaviour.
"""

from __future__ import annotations

import pickle
from typing import Any

import pytest

import hikrobot
from hikrobot import (
    AccessDeniedError,
    CallOrderError,
    CameraTimeoutError,
    GeneralError,
    GenICamError,
    GigEError,
    HikrobotError,
    IncompleteImageError,
    InsufficientBufferError,
    InvalidHandleError,
    NoDataError,
    StatusError,
    UpgradeError,
    USBError,
)
from hikrobot._errors import _CODES, _GROUPS, _NAMED, MV_OK, _Entry, call, check


class FakeEntryPoint:
    """Stands in for a ctypes function pointer, which exposes ``__name__``."""

    def __init__(self, status: int, name: str = "MV_CC_Fake") -> None:
        self.__name__ = name
        self._status = status
        self.args: tuple[Any, ...] = ()

    def __call__(self, *args: Any) -> int:
        self.args = args
        return self._status


class TestCheck:
    def test_ok_does_not_raise(self) -> None:
        assert check(MV_OK, "MV_CC_OpenDevice") == MV_OK

    @pytest.mark.parametrize("entry", _CODES, ids=lambda entry: entry.name)
    def test_every_transcribed_code_maps_to_its_group(self, entry: _Entry) -> None:
        with pytest.raises(StatusError) as excinfo:
            check(entry.code, "MV_CC_Whatever")

        error = excinfo.value
        assert error.status == entry.code
        assert error.name == entry.name
        assert error.description == entry.description

        expected_group = next(group for low, high, group in _GROUPS if low <= entry.code <= high)
        assert isinstance(error, expected_group)

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (0x80000000, InvalidHandleError),
            (0x80000003, CallOrderError),
            (0x80000007, NoDataError),
            (0x8000000B, IncompleteImageError),
            (0x8000000D, InsufficientBufferError),
            (0x80000107, CameraTimeoutError),
            (0x80000203, AccessDeniedError),
        ],
    )
    def test_named_codes_get_their_own_class(self, code: int, expected: type[StatusError]) -> None:
        with pytest.raises(expected):
            check(code, "MV_CC_Whatever")

    @pytest.mark.parametrize(
        ("code", "group"),
        [
            (0x800000FE, GeneralError),
            (0x800001FE, GenICamError),
            (0x800002FE, GigEError),
            (0x800003FE, USBError),
            (0x800004FE, UpgradeError),
        ],
    )
    def test_unnamed_code_in_a_known_range_raises_exactly_the_group(
        self, code: int, group: type[StatusError]
    ) -> None:
        with pytest.raises(group) as excinfo:
            check(code, "MV_CC_Whatever")
        # Exactly the group, not some subclass that happens to inherit from it.
        assert type(excinfo.value) is group
        assert excinfo.value.name == ""

    def test_code_outside_every_range_is_a_plain_status_error(self) -> None:
        with pytest.raises(StatusError) as excinfo:
            check(0x81234567, "MV_CC_Whatever")
        assert type(excinfo.value) is StatusError
        assert "unknown status 0x81234567" in str(excinfo.value)

    def test_isp_algorithm_codes_are_not_mistaken_for_camera_errors(self) -> None:
        # MV_ALG_E_MEM_NULL from MvISPErrorDefine.h; deliberately not transcribed.
        with pytest.raises(StatusError) as excinfo:
            check(0x10000002, "MV_CC_Whatever")
        assert type(excinfo.value) is StatusError
        assert excinfo.value.status == 0x10000002

    def test_signed_status_is_normalised(self) -> None:
        # ctypes defaults restype to a signed c_long, so MV_E_HANDLE arrives negative.
        with pytest.raises(InvalidHandleError) as excinfo:
            check(-2147483648, "MV_CC_StartGrabbing")
        assert excinfo.value.status == 0x80000000

    def test_survives_a_pickle_round_trip(self) -> None:
        # Multi-camera pipelines move errors between processes; the custom __init__
        # signature breaks the default exception pickling without __reduce__.
        with pytest.raises(AccessDeniedError) as excinfo:
            check(0x80000203, "MV_CC_OpenDevice")
        restored = pickle.loads(pickle.dumps(excinfo.value))
        assert type(restored) is AccessDeniedError
        assert restored.status == 0x80000203
        assert restored.name == "MV_E_ACCESS_DENIED"
        assert restored.operation == "MV_CC_OpenDevice"
        assert str(restored) == str(excinfo.value)

    def test_message_carries_operation_symbol_and_hex(self) -> None:
        with pytest.raises(AccessDeniedError) as excinfo:
            check(0x80000203, "MV_CC_OpenDevice")
        message = str(excinfo.value)
        assert "MV_CC_OpenDevice" in message
        assert "MV_E_ACCESS_DENIED" in message
        assert "0x80000203" in message


class TestTableConsistency:
    def test_all_55_codes_are_transcribed(self) -> None:
        # MvErrorDefine.h (SDK 4.4.1) defines 55 MV_E_* symbols. Guards against a line
        # lost while transcribing.
        assert len(_CODES) == 55

    def test_per_range_counts_match_the_header(self) -> None:
        # general 23, GenICam 10, GigE 10, USB 7, upgrade 5. A code transcribed into the
        # wrong section shows up here even though the total still adds up.
        counts = [
            sum(1 for entry in _CODES if low <= entry.code <= high) for low, high, _ in _GROUPS
        ]
        assert counts == [23, 10, 10, 7, 5]

    def test_codes_are_unique(self) -> None:
        codes = [entry.code for entry in _CODES]
        assert len(codes) == len(set(codes))

    def test_all_symbols_use_the_vendor_prefix(self) -> None:
        for entry in _CODES:
            assert entry.name.startswith("MV_E_"), entry.name

    def test_symbolic_names_are_unique(self) -> None:
        names = [entry.name for entry in _CODES]
        assert len(names) == len(set(names))

    def test_every_code_falls_in_exactly_one_group(self) -> None:
        for entry in _CODES:
            matches = [group for low, high, group in _GROUPS if low <= entry.code <= high]
            assert len(matches) == 1, entry.name

    def test_groups_do_not_overlap(self) -> None:
        for index, (low, high, _) in enumerate(_GROUPS):
            assert low <= high
            for other_low, other_high, _ in _GROUPS[index + 1 :]:
                assert high < other_low or other_high < low

    def test_named_classes_only_cover_transcribed_codes(self) -> None:
        transcribed = {entry.code for entry in _CODES}
        assert set(_NAMED) <= transcribed

    def test_named_class_matches_the_group_of_its_code(self) -> None:
        for code, exception_class in _NAMED.items():
            group = next(group for low, high, group in _GROUPS if low <= code <= high)
            assert issubclass(exception_class, group)

    def test_every_public_exception_derives_from_the_base(self) -> None:
        for name in hikrobot.__all__:
            attribute = getattr(hikrobot, name)
            if isinstance(attribute, type) and issubclass(attribute, BaseException):
                assert issubclass(attribute, HikrobotError), name

    def test_descriptions_are_present_and_ascii(self) -> None:
        # The vendor header stores its Chinese comments in GBK, which reads as mojibake in
        # UTF-8. A non-ASCII description means one of them was pasted in by mistake.
        for entry in _CODES:
            assert entry.description.strip(), entry.name
            assert entry.description.isascii(), entry.name


class TestCall:
    def test_success_returns_ok_and_forwards_arguments(self) -> None:
        func = FakeEntryPoint(MV_OK)
        assert call(func, 1, "two", None) == MV_OK
        assert func.args == (1, "two", None)

    def test_failure_raises_with_the_entry_point_name(self) -> None:
        func = FakeEntryPoint(0x80000203, name="MV_CC_OpenDevice")
        with pytest.raises(AccessDeniedError) as excinfo:
            call(func)
        assert excinfo.value.operation == "MV_CC_OpenDevice"

    def test_allowed_status_is_returned_instead_of_raised(self) -> None:
        func = FakeEntryPoint(0x80000007, name="MV_CC_GetImageBuffer")
        assert call(func, allow=(0x80000007,)) == 0x80000007

    def test_status_outside_the_allow_list_still_raises(self) -> None:
        func = FakeEntryPoint(0x80000003, name="MV_CC_GetImageBuffer")
        with pytest.raises(CallOrderError):
            call(func, allow=(0x80000007,))

    def test_signed_status_from_the_callable_is_normalised(self) -> None:
        func = FakeEntryPoint(-2147483648, name="MV_CC_StartGrabbing")
        with pytest.raises(InvalidHandleError) as excinfo:
            call(func)
        assert excinfo.value.status == 0x80000000

    def test_callable_without_a_name_still_produces_a_message(self) -> None:
        def anonymous(*args: Any) -> int:
            return 0x80000203

        nameless: Any = type("Nameless", (), {"__call__": staticmethod(anonymous)})()
        assert not hasattr(nameless, "__name__")
        with pytest.raises(AccessDeniedError) as excinfo:
            call(nameless)
        assert excinfo.value.operation
