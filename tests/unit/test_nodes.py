# SPDX-License-Identifier: Apache-2.0
"""Node map access and the typed properties built on it."""

from __future__ import annotations

import pytest

from conftest import FakeMvsLibrary, gige_device
from hikrobot import (
    Camera,
    CameraStateError,
    FloatRange,
    GenICamError,
    IntRange,
    ValueOutOfRangeError,
    enumerate_devices,
)


@pytest.fixture
def camera(fake_sdk: FakeMvsLibrary) -> Camera:
    fake_sdk.devices = [gige_device(serial="S1")]
    (device,) = enumerate_devices()
    cam = Camera(device)
    cam.open()
    return cam


class TestIntegerNodes:
    def test_read(self, camera: Camera) -> None:
        assert camera.nodes.get_int("Width") == 1920

    def test_range(self, camera: Camera) -> None:
        assert camera.nodes.int_range("Width") == IntRange(value=1920, min=32, max=1920, inc=2)

    def test_write_round_trips(self, camera: Camera) -> None:
        camera.nodes.set_int("Width", 640)
        assert camera.nodes.get_int("Width") == 640

    def test_out_of_range_write(self, camera: Camera) -> None:
        with pytest.raises(ValueOutOfRangeError):
            camera.nodes.set_int("Width", 999999)

    def test_off_increment_write(self, camera: Camera) -> None:
        with pytest.raises(ValueOutOfRangeError):
            camera.nodes.set_int("Width", 641)

    def test_64_bit_values_survive(self, fake_sdk: FakeMvsLibrary, camera: Camera) -> None:
        # The non-Ex structure is 32-bit; a timestamp-sized node would be truncated by it.
        big = 2**40
        fake_sdk.int_nodes["Timestamp"] = (big, 0, 2**62, 1)
        assert camera.nodes.get_int("Timestamp") == big


class TestFloatNodes:
    def test_read(self, camera: Camera) -> None:
        assert camera.nodes.get_float("ExposureTime") == pytest.approx(5000.0)

    def test_range(self, camera: Camera) -> None:
        found = camera.nodes.float_range("Gain")
        assert isinstance(found, FloatRange)
        assert found.min == pytest.approx(0.0)
        assert found.max == pytest.approx(23.981, rel=1e-6)

    def test_write_round_trips(self, camera: Camera) -> None:
        camera.nodes.set_float("ExposureTime", 1234.5)
        assert camera.nodes.get_float("ExposureTime") == pytest.approx(1234.5)

    def test_out_of_range_write(self, camera: Camera) -> None:
        with pytest.raises(ValueOutOfRangeError):
            camera.nodes.set_float("ExposureTime", 1.0)


class TestBoolNodes:
    def test_read_and_write(self, camera: Camera) -> None:
        assert camera.nodes.get_bool("ReverseX") is False
        camera.nodes.set_bool("ReverseX", True)
        assert camera.nodes.get_bool("ReverseX") is True


class TestEnumNodes:
    def test_read_returns_the_symbolic_name(self, camera: Camera) -> None:
        assert camera.nodes.get_enum("PixelFormat") == "Mono8"

    def test_entries(self, camera: Camera) -> None:
        assert camera.nodes.enum_entries("PixelFormat") == [
            "Mono8",
            "Mono10",
            "Mono10Packed",
            "Mono12",
            "Mono12Packed",
        ]

    def test_write_by_name_round_trips(self, camera: Camera) -> None:
        camera.nodes.set_enum("PixelFormat", "Mono12")
        assert camera.nodes.get_enum("PixelFormat") == "Mono12"

    def test_unknown_entry_is_refused(self, camera: Camera) -> None:
        with pytest.raises(GenICamError):
            camera.nodes.set_enum("PixelFormat", "BayerRG8")

    def test_more_than_64_entries_are_all_reported(
        self, fake_sdk: FakeMvsLibrary, camera: Camera
    ) -> None:
        # The reason the Ex structure is used: the plain one caps the list at 64.
        entries = [f"Format{index}" for index in range(120)]
        fake_sdk.enum_nodes["Wide"] = ("Format0", entries)
        assert camera.nodes.enum_entries("Wide") == entries


class TestStringsAndCommands:
    def test_string_round_trips(self, camera: Camera) -> None:
        assert camera.nodes.get_string("DeviceUserID") == "7"
        camera.nodes.set_string("DeviceUserID", "left")
        assert camera.nodes.get_string("DeviceUserID") == "left"

    def test_command_executes(self, fake_sdk: FakeMvsLibrary, camera: Camera) -> None:
        camera.nodes.execute("TriggerSoftware")
        assert fake_sdk.executed == ["TriggerSoftware"]

    def test_unknown_command(self, camera: Camera) -> None:
        with pytest.raises(GenICamError):
            camera.nodes.execute("NoSuchCommand")


class TestFailureModes:
    @pytest.mark.parametrize(
        "access",
        [
            lambda nodes: nodes.get_int("NoSuchNode"),
            lambda nodes: nodes.get_float("NoSuchNode"),
            lambda nodes: nodes.get_bool("NoSuchNode"),
            lambda nodes: nodes.get_enum("NoSuchNode"),
            lambda nodes: nodes.get_string("NoSuchNode"),
        ],
    )
    def test_missing_node_raises_the_genicam_group(self, camera: Camera, access: object) -> None:
        # Deliberately the group class: a missing node and a wrong-typed access come back
        # with the same status, so a more specific exception would be a guess.
        with pytest.raises(GenICamError):
            access(camera.nodes)  # type: ignore[operator]

    def test_wrong_accessor_for_an_existing_node_looks_the_same(self, camera: Camera) -> None:
        with pytest.raises(GenICamError):
            camera.nodes.get_int("ExposureTime")

    def test_non_ascii_node_name_is_rejected_locally(self, camera: Camera) -> None:
        with pytest.raises(ValueError, match="ASCII"):
            camera.nodes.get_int("Ширина")

    def test_a_closed_camera_refuses_node_access(self, camera: Camera) -> None:
        camera.close()
        with pytest.raises(CameraStateError, match="not open"):
            camera.nodes.get_int("Width")

    def test_handle_is_unavailable_while_closed(self, camera: Camera) -> None:
        camera.close()
        with pytest.raises(CameraStateError):
            _ = camera.handle


class TestTypedProperties:
    def test_geometry(self, camera: Camera) -> None:
        assert camera.width == 1920
        assert camera.height == 1200
        assert camera.offset_x == 0
        assert camera.offset_y == 0
        assert camera.payload_size == 2304000

    def test_geometry_setters(self, camera: Camera) -> None:
        camera.width = 640
        camera.height = 480
        assert (camera.width, camera.height) == (640, 480)

    def test_roi_origin_setters(self, fake_sdk: FakeMvsLibrary, camera: Camera) -> None:
        fake_sdk.int_nodes["OffsetX"] = (0, 0, 1280, 2)
        fake_sdk.int_nodes["OffsetY"] = (0, 0, 720, 2)
        camera.offset_x = 64
        camera.offset_y = 32
        assert (camera.offset_x, camera.offset_y) == (64, 32)

    def test_exposure_and_gain(self, camera: Camera) -> None:
        assert camera.exposure_us == pytest.approx(5000.0)
        assert camera.gain_db == pytest.approx(0.0)

        camera.exposure_us = 2500.0
        camera.gain_db = 6.0
        assert camera.exposure_us == pytest.approx(2500.0)
        assert camera.gain_db == pytest.approx(6.0)

    def test_ranges_are_exposed_without_node_names(self, camera: Camera) -> None:
        assert camera.exposure_range_us.min == pytest.approx(15.0)
        assert camera.gain_range_db.max == pytest.approx(23.981, rel=1e-6)

    def test_frame_rate(self, camera: Camera) -> None:
        assert camera.frame_rate == pytest.approx(41.0)
        camera.frame_rate = 10.0
        assert camera.frame_rate == pytest.approx(10.0)

    def test_pixel_format(self, camera: Camera) -> None:
        assert camera.pixel_format == "Mono8"
        assert "Mono12" in camera.pixel_formats

        camera.pixel_format = "Mono12"
        assert camera.pixel_format == "Mono12"

    def test_properties_need_an_open_camera(self, camera: Camera) -> None:
        camera.close()
        with pytest.raises(CameraStateError):
            _ = camera.width

    def test_out_of_range_property_write(self, camera: Camera) -> None:
        with pytest.raises(ValueOutOfRangeError):
            camera.exposure_us = 0.0
