# SPDX-License-Identifier: Apache-2.0
"""Hardware smoke tests. Opt-in: ``pytest tests/hardware --hardware``.

Everything here needs an installed MVS SDK. The camera tests additionally assume exactly
one reachable device and must leave no handle behind, including on failure.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest

from hikrobot import (
    BufferReleasedError,
    CallOrderError,
    Camera,
    CameraStateError,
    DeviceInfo,
    GenICamError,
    InsufficientBufferError,
    NoDataError,
    ValueOutOfRangeError,
    enumerate_devices,
)
from hikrobot._loader import load


@pytest.fixture(scope="module")
def device() -> DeviceInfo:
    devices = enumerate_devices()
    if not devices:
        pytest.skip("no reachable camera")
    return devices[0]


@pytest.mark.hardware
def test_sdk_loads() -> None:
    assert load() is not None


@pytest.mark.hardware
def test_enumeration_reports_usable_records() -> None:
    for found in enumerate_devices():
        assert found.transport in {"gige", "usb", "unknown"}
        if found.transport == "gige":
            assert found.ip_address is not None
            assert found.mac_address is not None


@pytest.mark.hardware
def test_gige_address_fields_are_self_consistent() -> None:
    """The subnet mask is what pins down the byte order of the address fields.

    Only one reading of these words produces a mask with contiguous leading ones, so this
    catches a byte-order regression without needing to know the camera's configuration.
    """
    from hikrobot.device import _ipv4

    for found in enumerate_devices(["gige"]):
        gige = found.raw.SpecialInfo.stGigEInfo
        mask = _ipv4(gige.nCurrentSubNetMask)
        octets = [int(part) for part in mask.split(".")]
        bits = "".join(f"{octet:08b}" for octet in octets)
        assert bits == "1" * bits.count("1") + "0" * bits.count("0"), mask

        # Only the low 16 bits of the high word carry MAC octets.
        assert (found.raw.nMacAddrHigh >> 16) == 0


@pytest.mark.hardware
def test_open_close_round_trip(device: DeviceInfo) -> None:
    camera = Camera(device)
    try:
        camera.open()
        assert camera.is_open
        assert camera.is_connected
    finally:
        camera.close()
    assert not camera.is_open


def _assert_settles_near(
    target: float,
    write: Callable[[float], None],
    read: Callable[[], float],
) -> None:
    """Check a float node accepts a value and then holds it.

    Exact equality is the wrong assertion: cameras quantise float features to a step the
    node map does not expose, so an MV-CS023-10GM answers ``1.0052`` to a written ``1.0``.
    What must hold is that the value moved close to the target and that writing the
    reported value back is a fixed point - that is what lets a caller read after writing
    and trust the result.
    """
    write(target)
    settled = read()
    assert settled == pytest.approx(target, rel=0.01), "the write did not take effect"

    write(settled)
    assert read() == pytest.approx(settled, rel=1e-6), "the reported value is not stable"


@pytest.fixture
def open_camera(device: DeviceInfo) -> Iterator[Camera]:
    camera = Camera(device)
    camera.open()
    try:
        yield camera
    finally:
        camera.close()


@pytest.mark.hardware
class TestNodeMap:
    def test_integer_node_reports_a_usable_range(self, open_camera: Camera) -> None:
        width = open_camera.nodes.int_range("Width")
        assert width.min <= width.value <= width.max
        assert width.inc >= 1
        assert (width.value - width.min) % width.inc == 0

    def test_float_node_reports_a_usable_range(self, open_camera: Camera) -> None:
        exposure = open_camera.nodes.float_range("ExposureTime")
        assert exposure.min <= exposure.value <= exposure.max
        assert exposure.min > 0

    def test_enum_current_value_is_one_of_its_entries(self, open_camera: Camera) -> None:
        entries = open_camera.nodes.enum_entries("PixelFormat")
        assert entries
        assert open_camera.nodes.get_enum("PixelFormat") in entries

    def test_string_node(self, open_camera: Camera) -> None:
        assert open_camera.nodes.get_string("DeviceModelName") == open_camera.info.model_name

    def test_missing_node_and_wrong_type_are_indistinguishable(self, open_camera: Camera) -> None:
        """Both come back as MV_E_GC_GENERIC, which is why neither gets a named class."""
        with pytest.raises(GenICamError) as missing:
            open_camera.nodes.get_int("NoSuchNodeAtAll")
        with pytest.raises(GenICamError) as wrong_type:
            open_camera.nodes.get_int("ExposureTime")  # a float node
        assert missing.value.name == wrong_type.value.name == "MV_E_GC_GENERIC"

    def test_out_of_range_write_is_distinguishable(self, open_camera: Camera) -> None:
        with pytest.raises(ValueOutOfRangeError) as excinfo:
            open_camera.nodes.set_int("Width", 10**9)
        assert excinfo.value.name == "MV_E_GC_RANGE"


@pytest.mark.hardware
class TestTypedProperties:
    def test_geometry_matches_the_node_map(self, open_camera: Camera) -> None:
        assert open_camera.width == open_camera.nodes.get_int("Width")
        assert open_camera.height == open_camera.nodes.get_int("Height")
        assert open_camera.payload_size > 0

    def test_exposure_round_trips(self, open_camera: Camera) -> None:
        original = open_camera.exposure_us
        bounds = open_camera.exposure_range_us
        target = max(bounds.min, min(bounds.max, 2500.0))
        try:
            _assert_settles_near(
                target,
                lambda value: setattr(open_camera, "exposure_us", value),
                lambda: open_camera.exposure_us,
            )
        finally:
            open_camera.exposure_us = original

    def test_gain_round_trips(self, open_camera: Camera) -> None:
        original = open_camera.gain_db
        bounds = open_camera.gain_range_db
        target = min(bounds.max, bounds.min + 1.0)
        try:
            _assert_settles_near(
                target,
                lambda value: setattr(open_camera, "gain_db", value),
                lambda: open_camera.gain_db,
            )
        finally:
            open_camera.gain_db = original

    def test_pixel_format_round_trips(self, open_camera: Camera) -> None:
        original = open_camera.pixel_format
        options = open_camera.pixel_formats
        assert original in options
        other = next((name for name in options if name != original), None)
        if other is None:
            pytest.skip("camera offers a single pixel format")
        try:
            open_camera.pixel_format = other
            assert open_camera.pixel_format == other
        finally:
            open_camera.pixel_format = original

    def test_roi_change_updates_the_payload(self, open_camera: Camera) -> None:
        original_width = open_camera.width
        bounds = open_camera.nodes.int_range("Width")
        narrow = max(bounds.min, bounds.max // 2 // bounds.inc * bounds.inc)
        if narrow == original_width:
            pytest.skip("width is already at the test value")
        try:
            before = open_camera.payload_size
            open_camera.width = narrow
            assert open_camera.width == narrow
            assert open_camera.payload_size < before
        finally:
            open_camera.width = original_width

    def test_properties_need_an_open_camera(self, device: DeviceInfo) -> None:
        closed = Camera(device)
        with pytest.raises(CameraStateError):
            _ = closed.width


@pytest.mark.hardware
class TestStreaming:
    def test_grabs_a_frame_matching_the_node_map(self, open_camera: Camera) -> None:
        for frame in open_camera.frames(timeout_ms=5000):
            assert frame.width == open_camera.width
            assert frame.height == open_camera.height
            assert frame.pixel_format == open_camera.pixel_format
            assert frame.size_bytes == open_camera.payload_size
            assert frame.data.shape[:2] == (frame.height, frame.width)
            break

    def test_frames_are_released_after_each_body(self, open_camera: Camera) -> None:
        seen = []
        for index, frame in enumerate(open_camera.frames(timeout_ms=5000)):
            seen.append(frame)
            if index == 2:
                break
        assert [frame.is_released for frame in seen] == [True, True, True]

    def test_data_after_release_raises_on_real_memory(self, open_camera: Camera) -> None:
        escaped = []
        for frame in open_camera.frames(timeout_ms=5000):
            escaped.append(frame)
            break
        with pytest.raises(BufferReleasedError):
            _ = escaped[0].data

    def test_copy_outlives_the_frame_and_matches_it(self, open_camera: Camera) -> None:
        kept = None
        checksum = None
        for frame in open_camera.frames(timeout_ms=5000):
            kept = frame.copy()
            checksum = int(frame.data.sum())
            break
        assert kept is not None
        assert int(kept.sum()) == checksum
        assert kept.flags.writeable

    def test_the_node_pool_is_fixed_and_recycled(self, open_camera: Camera) -> None:
        """The reason a view must not outlive its frame, shown on the real driver."""
        addresses = []
        for index, frame in enumerate(open_camera.frames(timeout_ms=5000, node_count=3)):
            addresses.append(frame.data.ctypes.data)
            if index == 7:
                break
        assert len(addresses) == 8
        assert len(set(addresses)) < len(addresses), "no node was reused in eight frames"

    def test_frames_raw_hands_over_the_release(self, open_camera: Camera) -> None:
        # Acquisition is started here, not by the iterator: an iterator that owns it stops
        # it on the way out, and stopping reclaims every outstanding node. The pool must
        # also be bigger than the number of frames held at once - it holds exactly
        # node_count, and the SDK's default is one.
        open_camera.start_grabbing(node_count=4)
        held = []
        try:
            for index, frame in enumerate(open_camera.frames_raw(timeout_ms=5000)):
                held.append(frame)
                if index == 1:
                    break
            assert [frame.is_released for frame in held] == [False, False]
            assert held[0].data.ctypes.data != held[1].data.ctypes.data
        finally:
            for frame in held:
                frame.release()
            open_camera.stop_grabbing()

    def test_the_pool_starves_when_frames_are_not_released(self, open_camera: Camera) -> None:
        """Holding more nodes than the pool has is a stall, not a silent drop."""
        open_camera.start_grabbing(node_count=2)
        held = []
        try:
            with pytest.raises(InsufficientBufferError):
                for frame in open_camera.frames_raw(timeout_ms=5000):
                    held.append(frame)
        finally:
            for frame in held:
                frame.release()
            open_camera.stop_grabbing()
        assert len(held) == 2, "the pool served exactly node_count frames"

    def test_stop_grabbing_reclaims_an_outstanding_frame(self, open_camera: Camera) -> None:
        open_camera.start_grabbing(node_count=2)
        frame = next(iter(open_camera.frames_raw(timeout_ms=5000)))
        open_camera.stop_grabbing()

        assert frame.is_released is True
        with pytest.raises(BufferReleasedError):
            _ = frame.data

    def test_frame_numbers_and_timestamps_advance(self, open_camera: Camera) -> None:
        numbers, host_ms, ticks = [], [], []
        for index, frame in enumerate(open_camera.frames(timeout_ms=5000)):
            numbers.append(frame.frame_number)
            host_ms.append(frame.host_timestamp_ms)
            ticks.append(frame.device_timestamp_ticks)
            if index == 2:
                break
        assert numbers == sorted(numbers)
        assert host_ms == sorted(host_ms)
        assert ticks == sorted(ticks)
        # Milliseconds since the Unix epoch, not nanoseconds and not since boot.
        assert host_ms[0] > 1_700_000_000_000
        assert host_ms[0] < 4_000_000_000_000

    def test_device_ticks_agree_with_the_tick_frequency_node(self, open_camera: Camera) -> None:
        frequency = open_camera.nodes.get_int("GevTimestampTickFrequency")
        stamps = []
        for index, frame in enumerate(open_camera.frames(timeout_ms=5000)):
            stamps.append((frame.host_timestamp_ms, frame.device_timestamp_ticks))
            if index == 3:
                break
        host_span_s = (stamps[-1][0] - stamps[0][0]) / 1000
        tick_span_s = (stamps[-1][1] - stamps[0][1]) / frequency
        assert tick_span_s == pytest.approx(host_span_s, rel=0.05, abs=0.01)

    def test_timeout_raises_when_the_camera_waits_for_a_trigger(self, open_camera: Camera) -> None:
        original = open_camera.nodes.get_enum("TriggerMode")
        open_camera.nodes.set_enum("TriggerMode", "On")
        try:
            with pytest.raises(NoDataError):
                next(iter(open_camera.frames(timeout_ms=300)))
        finally:
            open_camera.nodes.set_enum("TriggerMode", original)

    def test_acquisition_is_left_as_it_was_found(self, open_camera: Camera) -> None:
        for _ in open_camera.frames(timeout_ms=5000):
            break
        assert open_camera.is_grabbing is False

        open_camera.start_grabbing()
        try:
            for _ in open_camera.frames(timeout_ms=5000):
                break
            assert open_camera.is_grabbing is True
        finally:
            open_camera.stop_grabbing()


@pytest.mark.hardware
class TestTransport:
    def test_optimal_packet_size_is_plausible(self, open_camera: Camera) -> None:
        size = open_camera.optimal_packet_size
        # A GigE stream packet is at least a minimum Ethernet frame and at most a jumbo
        # one. Anything outside that means the return value was read as a status.
        assert 576 <= size <= 16000, size

    def test_tune_applies_what_it_probed(self, open_camera: Camera) -> None:
        original = open_camera.packet_size
        try:
            applied = open_camera.tune_packet_size()
            assert open_camera.packet_size == applied
        finally:
            open_camera.packet_size = original

    def test_tune_is_refused_while_grabbing(self, open_camera: Camera) -> None:
        open_camera.start_grabbing()
        try:
            with pytest.raises(CameraStateError):
                open_camera.tune_packet_size()
        finally:
            open_camera.stop_grabbing()

    def test_resend_is_accepted(self, open_camera: Camera) -> None:
        open_camera.enable_resend(True, max_resend_percent=20, timeout_ms=100)
        open_camera.enable_resend(False)

    def test_statistics_exist_only_while_acquisition_runs(self, open_camera: Camera) -> None:
        """Measured rule: the counters live between start and stop, and reset on start."""
        with pytest.raises(CallOrderError):
            _ = open_camera.statistics

        open_camera.start_grabbing(node_count=3)
        try:
            assert open_camera.statistics.received_frames == 0
            for frame in open_camera.frames_raw(timeout_ms=5000):
                frame.release()
                break
            assert open_camera.statistics.received_frames >= 1
        finally:
            open_camera.stop_grabbing()

        with pytest.raises(CallOrderError):
            _ = open_camera.statistics

    def test_statistics_reset_on_each_start(self, open_camera: Camera) -> None:
        open_camera.start_grabbing(node_count=3)
        try:
            for frame in open_camera.frames_raw(timeout_ms=5000):
                frame.release()
                break
            assert open_camera.statistics.received_frames >= 1
        finally:
            open_camera.stop_grabbing()

        open_camera.start_grabbing(node_count=3)
        try:
            assert open_camera.statistics.received_frames == 0
        finally:
            open_camera.stop_grabbing()

    def test_a_sustained_run_loses_nothing(self, open_camera: Camera) -> None:
        """The point of step 8: does the configured link actually carry the stream?

        A packet size the path cannot carry, or a link too slow for the frame rate, shows
        up here as lost packets or dropped frames - not as a wrapper bug.
        """
        open_camera.tune_packet_size()
        open_camera.enable_resend(True)

        wanted = 20
        received = 0
        incomplete = 0
        payload = 0
        # Acquisition is owned here, not by the iterator: the counters vanish the moment
        # it stops, so the run has to be measured from inside it.
        open_camera.start_grabbing(node_count=4)
        started = time.monotonic()
        try:
            for frame in open_camera.frames(timeout_ms=5000):
                received += 1
                payload += frame.size_bytes
                if frame.lost_packets:
                    incomplete += 1
                if received == wanted:
                    break
            elapsed = time.monotonic() - started
            stats = open_camera.statistics
        finally:
            open_camera.stop_grabbing()

        rate = received / elapsed
        throughput = payload / elapsed / 1e6
        print(
            f"\n    {received} frames in {elapsed:.2f}s = {rate:.2f} fps, "
            f"{throughput:.1f} MB/s, packet={open_camera.packet_size}\n"
            f"    lost packets={stats.lost_packets} lost frames={stats.lost_frames} "
            f"resent={stats.resent_packets} incomplete={incomplete}"
        )

        assert received == wanted
        assert incomplete == 0, "frames arrived with missing packets"
        assert stats.lost_frames == 0, "the driver dropped frames"

    def test_the_link_can_carry_the_configured_frame_rate(self, open_camera: Camera) -> None:
        """Separates a slow link from a slow consumer, which look identical otherwise."""
        link_mbps = open_camera.nodes.get_int("DeviceLinkSpeed")
        payload_bits = open_camera.payload_size * 8
        ceiling = link_mbps * 1e6 / payload_bits

        achievable = open_camera.nodes.get_float("ResultingFrameRate")
        assert achievable <= ceiling * 1.05, (
            f"the camera claims {achievable:.2f} fps but a {link_mbps} Mbps link caps "
            f"{open_camera.payload_size}-byte frames at {ceiling:.2f} fps"
        )
        if link_mbps < 1000:
            pytest.skip(
                f"link negotiated at {link_mbps} Mbps, capping this camera at "
                f"{ceiling:.2f} fps - check the cable and the switch port"
            )


@pytest.mark.hardware
def test_exclusive_access_is_exclusive(device: DeviceInfo) -> None:
    from hikrobot import AccessDeniedError

    first = Camera(device)
    second = Camera(device)
    try:
        first.open()
        with pytest.raises(AccessDeniedError):
            second.open()
        assert not second.is_open
    finally:
        second.close()
        first.close()
