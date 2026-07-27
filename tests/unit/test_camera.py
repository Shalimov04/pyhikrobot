# SPDX-License-Identifier: Apache-2.0
"""Handle lifecycle: create, open, close, destroy - and the failure paths between them."""

from __future__ import annotations

import pytest

from conftest import FakeMvsLibrary, gige_device
from hikrobot import (
    AccessDeniedError,
    Camera,
    CameraStateError,
    DeviceBusyError,
    enumerate_devices,
)
from hikrobot._ctypes_defs import (
    MV_ACCESS_Exclusive,
    MV_ACCESS_Monitor,
)


@pytest.fixture
def camera(fake_sdk: FakeMvsLibrary) -> Camera:
    fake_sdk.devices = [gige_device(model="MV-CA050-10GM", serial="S1")]
    (device,) = enumerate_devices()
    return Camera(device)


class TestLifecycle:
    def test_construction_touches_no_sdk_state(self, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.devices = [gige_device()]
        (device,) = enumerate_devices()

        cam = Camera(device)
        assert cam.is_open is False
        assert fake_sdk.open_handles == set()

    def test_open_creates_a_handle_and_opens_the_device(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.open()

        assert camera.is_open is True
        assert len(fake_sdk.open_handles) == 1
        assert fake_sdk.opened_handles == fake_sdk.open_handles

    def test_close_closes_the_device_and_destroys_the_handle(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.open()
        camera.close()

        assert camera.is_open is False
        assert fake_sdk.open_handles == set()
        assert fake_sdk.opened_handles == set()

    def test_close_is_idempotent(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.open()
        camera.close()
        camera.close()

        assert fake_sdk.entry_point("MV_CC_CloseDevice").calls != []
        assert len(fake_sdk.entry_point("MV_CC_CloseDevice").calls) == 1

    def test_closing_a_camera_that_never_opened_does_nothing(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.close()
        assert fake_sdk.entry_point("MV_CC_CloseDevice").calls == []

    def test_reopening_after_close_works(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.open()
        camera.close()
        camera.open()

        assert camera.is_open is True
        assert len(fake_sdk.open_handles) == 1
        assert len(fake_sdk.entry_point("MV_CC_CreateHandle").calls) == 2

    def test_opening_twice_is_rejected_before_the_sdk_is_called(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.open()
        with pytest.raises(CameraStateError, match="already open"):
            camera.open()
        assert len(fake_sdk.entry_point("MV_CC_CreateHandle").calls) == 1


class TestAccessModes:
    def test_default_is_exclusive(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.open()
        assert fake_sdk.open_arguments == [(MV_ACCESS_Exclusive, 0)]

    def test_mode_and_key_are_forwarded(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.open(access="monitor", switchover_key=0x1234)
        assert fake_sdk.open_arguments == [(MV_ACCESS_Monitor, 0x1234)]

    def test_unknown_mode_is_rejected_before_a_handle_is_created(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        with pytest.raises(ValueError, match="readonly"):
            camera.open(access="readonly")
        assert fake_sdk.entry_point("MV_CC_CreateHandle").calls == []
        assert camera.is_open is False


class TestFailurePaths:
    def test_failed_open_does_not_leak_the_handle(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        # The classic leak: MV_CC_CreateHandle succeeded, MV_CC_OpenDevice did not, and
        # nobody destroyed the handle. It stays held for the life of the process.
        fake_sdk.statuses["MV_CC_OpenDevice"] = 0x80000203

        with pytest.raises(AccessDeniedError):
            camera.open()

        assert camera.is_open is False
        assert fake_sdk.open_handles == set()
        assert len(fake_sdk.entry_point("MV_CC_DestroyHandle").calls) == 1

    def test_a_failing_destroy_does_not_hide_why_the_open_failed(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        fake_sdk.statuses["MV_CC_OpenDevice"] = 0x80000203
        fake_sdk.statuses["MV_CC_DestroyHandle"] = 0x80000000

        with pytest.raises(AccessDeniedError):
            camera.open()

    def test_failed_create_leaves_the_camera_closed(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        fake_sdk.statuses["MV_CC_CreateHandle"] = 0x80000006

        with pytest.raises(Exception, match="MV_E_RESOURCE"):
            camera.open()

        assert camera.is_open is False
        assert fake_sdk.entry_point("MV_CC_OpenDevice").calls == []

    def test_the_handle_is_destroyed_even_when_close_fails(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.open()
        fake_sdk.statuses["MV_CC_CloseDevice"] = 0x80000204

        with pytest.raises(DeviceBusyError):
            camera.close()

        assert camera.is_open is False
        assert len(fake_sdk.entry_point("MV_CC_DestroyHandle").calls) == 1

    def test_a_failed_close_still_forgets_the_handle(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        # Otherwise a retry would try to close an already-destroyed handle.
        camera.open()
        fake_sdk.statuses["MV_CC_CloseDevice"] = 0x80000204
        with pytest.raises(DeviceBusyError):
            camera.close()

        camera.close()
        assert len(fake_sdk.entry_point("MV_CC_CloseDevice").calls) == 1


class TestContextManager:
    def test_opens_and_closes(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        with camera as entered:
            assert entered is camera
            assert len(fake_sdk.open_handles) == 1
        assert camera.is_open is False
        assert fake_sdk.open_handles == set()

    def test_closes_on_an_exception_in_the_body(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        with pytest.raises(RuntimeError):  # noqa: SIM117 - the nesting is what is tested
            with camera:
                raise RuntimeError("boom")
        assert camera.is_open is False
        assert fake_sdk.open_handles == set()

    def test_an_already_open_camera_is_not_opened_twice(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.open()
        with camera:
            pass
        assert len(fake_sdk.entry_point("MV_CC_OpenDevice").calls) == 1


class TestIntrospection:
    def test_is_connected_is_false_while_closed(self, camera: Camera) -> None:
        assert camera.is_connected is False

    def test_is_connected_asks_the_sdk_while_open(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.open()
        assert camera.is_connected is True

        fake_sdk.connected = False
        assert camera.is_connected is False

    def test_info_is_the_device_it_was_built_from(self, camera: Camera) -> None:
        assert camera.info.serial_number == "S1"

    def test_repr_shows_the_state(self, camera: Camera) -> None:
        assert "closed" in repr(camera)
        camera.open()
        assert "open" in repr(camera)
        assert "S1" in repr(camera)
