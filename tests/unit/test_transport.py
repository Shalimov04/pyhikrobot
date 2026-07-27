# SPDX-License-Identifier: Apache-2.0
"""GigE transport tuning: packet size, resend, loss counters."""

from __future__ import annotations

import pytest

from conftest import FakeMvsLibrary, gige_device
from hikrobot import (
    CallOrderError,
    Camera,
    CameraStateError,
    InvalidHandleError,
    TransportStats,
    enumerate_devices,
)


@pytest.fixture
def camera(fake_sdk: FakeMvsLibrary) -> Camera:
    fake_sdk.devices = [gige_device(serial="S1")]
    (device,) = enumerate_devices()
    cam = Camera(device)
    cam.open()
    return cam


class TestPacketSize:
    def test_read_and_write(self, camera: Camera) -> None:
        assert camera.packet_size == 1500
        camera.packet_size = 8164
        assert camera.packet_size == 8164

    def test_packet_delay(self, camera: Camera) -> None:
        assert camera.packet_delay == 400
        camera.packet_delay = 0
        assert camera.packet_delay == 0

    def test_optimal_size_is_a_size_not_a_status(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        # The entry point returns the size itself; only a value with the high bit set is
        # an error. Reading it as a status would turn 1500 into a bogus success.
        fake_sdk.optimal_packet_size = 8164
        assert camera.optimal_packet_size == 8164

    def test_optimal_size_reports_a_returned_error_code(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        # A real SDK answers MV_E_HANDLE to a null handle - the failure arrives through
        # the return value, not through a status argument.
        fake_sdk.statuses["MV_CC_GetOptimalPacketSize"] = 0x80000000
        with pytest.raises(InvalidHandleError) as excinfo:
            _ = camera.optimal_packet_size
        assert excinfo.value.operation == "MV_CC_GetOptimalPacketSize"

    def test_tune_probes_then_applies(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.optimal_packet_size = 8164
        assert camera.tune_packet_size() == 8164
        assert camera.packet_size == 8164

    def test_tune_is_refused_while_grabbing(self, camera: Camera) -> None:
        # The SDK requires the stream channel idle, and the size is fixed once acquisition
        # starts; catching it here says why instead of letting the SDK say "call order".
        camera.start_grabbing()
        with pytest.raises(CameraStateError, match="grabbing"):
            camera.tune_packet_size()

    def test_needs_an_open_camera(self, camera: Camera) -> None:
        camera.close()
        with pytest.raises(CameraStateError):
            _ = camera.optimal_packet_size


class TestResend:
    def test_defaults_are_forwarded(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.enable_resend()
        assert fake_sdk.resend_settings == (True, 10, 50)

    def test_arguments_are_forwarded(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.enable_resend(True, max_resend_percent=25, timeout_ms=200)
        assert fake_sdk.resend_settings == (True, 25, 200)

    def test_can_be_disabled(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.enable_resend(False)
        assert fake_sdk.resend_settings == (False, 10, 50)


class TestStatistics:
    def test_unavailable_before_acquisition_starts(self, camera: Camera) -> None:
        # Measured on a real camera: the counters live only between start and stop.
        with pytest.raises(CallOrderError):
            _ = camera.statistics

    def test_reports_the_counters(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.stats_received_bytes = 2304000 * 4
        fake_sdk.stats_lost_packets = 7
        fake_sdk.stats_lost_frames = 1
        camera.start_grabbing()

        stats = camera.statistics
        assert isinstance(stats, TransportStats)
        assert stats.received_bytes == 2304000 * 4
        assert stats.lost_packets == 7
        assert stats.lost_frames == 1

    def test_counts_frames_actually_delivered(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        # Read from inside the acquisition: the counters are gone once it stops.
        camera.start_grabbing()
        try:
            for index, _frame in enumerate(camera.frames(timeout_ms=100)):
                if index == 2:
                    break
            assert camera.statistics.received_frames == 3
        finally:
            camera.stop_grabbing()

    def test_unavailable_once_acquisition_stops(self, camera: Camera) -> None:
        camera.start_grabbing()
        camera.stop_grabbing()
        with pytest.raises(CallOrderError):
            _ = camera.statistics

    def test_counters_reset_on_each_start(self, camera: Camera) -> None:
        camera.start_grabbing()
        for index, _frame in enumerate(camera.frames(timeout_ms=100)):
            if index == 1:
                break
        assert camera.statistics.received_frames == 2
        camera.stop_grabbing()

        camera.start_grabbing()
        try:
            assert camera.statistics.received_frames == 0
        finally:
            camera.stop_grabbing()

    def test_needs_an_open_camera(self, camera: Camera) -> None:
        camera.close()
        with pytest.raises(CameraStateError):
            _ = camera.statistics
