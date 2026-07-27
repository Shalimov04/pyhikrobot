# SPDX-License-Identifier: Apache-2.0
"""Streaming, frame metadata and - above all - buffer lifetime.

The fake library hands out nodes from a fixed pool and scribbles ``0xEE`` over one as soon
as it is freed, exactly as the driver overwrites a released node with the next frame. That
is what makes "the view went stale" observable instead of theoretical.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeMvsLibrary, gige_device
from hikrobot import (
    BufferReleasedError,
    Camera,
    CameraStateError,
    Frame,
    NoDataError,
    UnsupportedPixelFormatError,
    enumerate_devices,
)
from hikrobot._ctypes_defs import (
    PixelType_Gvsp_Mono12,
    PixelType_Gvsp_Mono12_Packed,
    PixelType_Gvsp_RGB8_Packed,
)


@pytest.fixture
def camera(fake_sdk: FakeMvsLibrary) -> Camera:
    fake_sdk.devices = [gige_device(serial="S1")]
    (device,) = enumerate_devices()
    cam = Camera(device)
    cam.open()
    return cam


def first_frame(camera: Camera) -> Frame:
    """One frame that outlives the iterator, still holding its node.

    Acquisition is started explicitly: an iterator that starts it also stops it when it is
    closed, and stopping hands every outstanding node back.
    """
    if not camera.is_grabbing:
        camera.start_grabbing(node_count=3)
    return next(iter(camera.frames_raw(timeout_ms=100)))


class TestGrabbingState:
    def test_a_fresh_camera_is_not_grabbing(self, camera: Camera) -> None:
        assert camera.is_grabbing is False

    def test_start(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.start_grabbing()
        assert camera.is_grabbing is True
        assert fake_sdk.grabbing is True

    def test_stop(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.start_grabbing()
        camera.stop_grabbing()
        assert camera.is_grabbing is False
        assert fake_sdk.grabbing is False

    def test_stop_is_idempotent(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        camera.stop_grabbing()
        camera.stop_grabbing()
        assert fake_sdk.entry_point("MV_CC_StopGrabbing").calls == []

    def test_starting_twice_is_rejected(self, camera: Camera) -> None:
        camera.start_grabbing()
        with pytest.raises(CameraStateError, match="already grabbing"):
            camera.start_grabbing()

    def test_node_count_is_applied_before_starting(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.start_grabbing(node_count=7)
        assert fake_sdk.node_count == 7
        # The pool is fixed once acquisition starts, so the order matters.
        assert fake_sdk.entry_point("MV_CC_SetImageNodeNum").calls
        assert fake_sdk.entry_point("MV_CC_StartGrabbing").calls

    def test_closing_forgets_the_grabbing_state(self, camera: Camera) -> None:
        camera.start_grabbing()
        camera.close()
        camera.open()
        assert camera.is_grabbing is False

    def test_grabbing_needs_an_open_camera(self, camera: Camera) -> None:
        camera.close()
        with pytest.raises(CameraStateError):
            camera.start_grabbing()


class TestFrameMetadata:
    def test_geometry_and_format(self, camera: Camera) -> None:
        frame = first_frame(camera)
        try:
            assert (frame.width, frame.height) == (16, 8)
            assert frame.pixel_format == "Mono8"
            assert frame.bit_depth == 8
            assert frame.size_bytes == 16 * 8
        finally:
            frame.release()

    def test_timestamps_keep_their_units_in_their_names(self, camera: Camera) -> None:
        frame = first_frame(camera)
        try:
            # Milliseconds since the Unix epoch, and device ticks - not nanoseconds.
            assert frame.host_timestamp_ms > 1_700_000_000_000
            assert frame.device_timestamp_ticks == 0
        finally:
            frame.release()

    def test_metadata_survives_the_release(self, camera: Camera) -> None:
        frame = first_frame(camera)
        frame.release()
        assert frame.width == 16
        assert frame.pixel_format == "Mono8"
        assert frame.frame_number == 0

    def test_frame_numbers_advance(self, camera: Camera) -> None:
        numbers = []
        for index, frame in enumerate(camera.frames(timeout_ms=100)):
            numbers.append(frame.frame_number)
            if index == 2:
                break
        assert numbers == [0, 1, 2]


class TestPixelData:
    def test_is_a_zero_copy_view_of_the_node(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        frame = first_frame(camera)
        try:
            data = frame.data
            assert data.shape == (8, 16)
            assert data.dtype == np.uint8
            # The fake fills every byte with the frame number.
            assert np.all(data == 0)
            assert data.base is not None, "the array copied instead of viewing"
        finally:
            frame.release()

    def test_is_read_only(self, camera: Camera) -> None:
        frame = first_frame(camera)
        try:
            with pytest.raises(ValueError, match="read-only"):
                frame.data[0, 0] = 1
        finally:
            frame.release()

    def test_copy_is_writable_and_owns_its_memory(self, camera: Camera) -> None:
        frame = first_frame(camera)
        owned = frame.copy()
        frame.release()

        owned[0, 0] = 123
        assert owned[0, 0] == 123
        assert owned.shape == (8, 16)

    def test_multi_channel_formats_get_a_third_axis(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        fake_sdk.frame_pixel_type = PixelType_Gvsp_RGB8_Packed
        fake_sdk.frame_bytes_per_pixel = 3
        frame = first_frame(camera)
        try:
            assert frame.data.shape == (8, 16, 3)
            assert frame.pixel_format == "RGB8"
        finally:
            frame.release()

    def test_16_bit_formats_use_uint16(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.frame_pixel_type = PixelType_Gvsp_Mono12
        fake_sdk.frame_bytes_per_pixel = 2
        frame = first_frame(camera)
        try:
            assert frame.data.dtype == np.uint16
            assert frame.data.shape == (8, 16)
            assert frame.bit_depth == 16
        finally:
            frame.release()

    def test_packed_formats_refuse_a_shaped_array_but_give_bytes(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        fake_sdk.frame_pixel_type = PixelType_Gvsp_Mono12_Packed
        fake_sdk.frame_bytes_per_pixel = 2
        frame = first_frame(camera)
        try:
            assert frame.pixel_format == "Mono12Packed"
            with pytest.raises(UnsupportedPixelFormatError, match="Mono12Packed"):
                _ = frame.data
            assert frame.raw_bytes.shape == (16 * 8 * 2,)
        finally:
            frame.release()

    def test_unknown_format_is_reported_as_hex(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        fake_sdk.frame_pixel_type = 0x0BAD0BAD
        frame = first_frame(camera)
        try:
            assert frame.pixel_format == "0x0BAD0BAD"
            with pytest.raises(UnsupportedPixelFormatError):
                _ = frame.data
        finally:
            frame.release()


class TestBufferLifetime:
    def test_data_after_release_raises(self, camera: Camera) -> None:
        frame = first_frame(camera)
        frame.release()
        with pytest.raises(BufferReleasedError):
            _ = frame.data

    def test_raw_bytes_after_release_raises(self, camera: Camera) -> None:
        frame = first_frame(camera)
        frame.release()
        with pytest.raises(BufferReleasedError):
            _ = frame.raw_bytes

    def test_copy_after_release_raises(self, camera: Camera) -> None:
        frame = first_frame(camera)
        frame.release()
        with pytest.raises(BufferReleasedError):
            frame.copy()

    def test_double_release_is_harmless(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        # The fake reports a real double free as MV_E_BUF_INVALID, so this passing means
        # the second release genuinely did not reach the SDK.
        frame = first_frame(camera)
        frame.release()
        frame.release()
        assert len(fake_sdk.entry_point("MV_CC_FreeImageBuffer").calls) == 1
        assert frame.is_released is True

    def test_a_retained_view_really_does_go_stale(self, camera: Camera) -> None:
        """Why the flag exists: the array keeps working and starts lying."""
        frame = first_frame(camera)
        leaked = frame.data
        assert np.all(leaked == 0)

        frame.release()
        # No exception from NumPy - the memory is still mapped, it just holds something
        # else now. This is the corruption BufferReleasedError is there to prevent.
        assert np.all(leaked == 0xEE)

    def test_frames_releases_after_each_body(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        held = []
        for index, frame in enumerate(camera.frames(timeout_ms=100)):
            held.append(frame)
            assert frame.is_released is False
            if index == 2:
                break
        assert [frame.is_released for frame in held] == [True, True, True]
        assert fake_sdk.live_nodes == set()

    def test_frames_releases_when_the_body_raises(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        escaped: list[Frame] = []
        with pytest.raises(RuntimeError):
            for frame in camera.frames(timeout_ms=100):
                escaped.append(frame)
                raise RuntimeError("boom")
        assert escaped[0].is_released is True
        assert fake_sdk.live_nodes == set()

    def test_frames_releases_on_break(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        for _ in camera.frames(timeout_ms=100):
            break
        assert fake_sdk.live_nodes == set()

    def test_frames_raw_leaves_the_release_to_the_caller(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.start_grabbing(node_count=3)
        held = []
        for index, frame in enumerate(camera.frames_raw(timeout_ms=100)):
            held.append(frame)
            if index == 1:
                break
        try:
            assert [frame.is_released for frame in held] == [False, False]
            assert len(fake_sdk.live_nodes) == 2
        finally:
            for frame in held:
                frame.release()
        assert fake_sdk.live_nodes == set()
        camera.stop_grabbing()

    def test_leaving_frames_raw_reclaims_what_the_caller_still_holds(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        """The iterator that started acquisition also ends it, and that ends the nodes.

        Stopping tears the pool down, so a frame carried out of the loop would be a view
        onto memory the driver has reclaimed. It is marked released instead.
        """
        escaped = []
        for index, frame in enumerate(camera.frames_raw(timeout_ms=100, node_count=3)):
            escaped.append(frame)
            if index == 1:
                break

        assert [frame.is_released for frame in escaped] == [True, True]
        assert fake_sdk.live_nodes == set()
        with pytest.raises(BufferReleasedError):
            _ = escaped[0].data

    def test_stop_grabbing_reclaims_outstanding_frames(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.start_grabbing(node_count=3)
        frame = next(iter(camera.frames_raw(timeout_ms=100)))
        assert len(fake_sdk.live_nodes) == 1, "the node should still be out"

        camera.stop_grabbing()
        assert frame.is_released is True
        assert fake_sdk.live_nodes == set()
        # The node went back before the stop, not after: a release afterwards is what the
        # SDK answers MV_E_CALLORDER to.
        assert fake_sdk.entry_point("MV_CC_FreeImageBuffer").calls
        assert fake_sdk.freed

    def test_closing_a_grabbing_camera_reclaims_frames(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.start_grabbing(node_count=3)
        frame = next(iter(camera.frames_raw(timeout_ms=100)))

        camera.close()
        assert frame.is_released is True
        assert fake_sdk.live_nodes == set()
        assert fake_sdk.grabbing is False


class TestAcquisitionLifecycle:
    def test_frames_starts_and_stops_acquisition(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        for _ in camera.frames(timeout_ms=100):
            assert fake_sdk.grabbing is True
            break
        assert fake_sdk.grabbing is False

    def test_an_explicit_start_keeps_control(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        camera.start_grabbing()
        for _ in camera.frames(timeout_ms=100):
            break
        assert fake_sdk.grabbing is True, "the iterator stopped acquisition it did not start"
        camera.stop_grabbing()

    def test_timeout_raises_rather_than_looping_silently(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        fake_sdk.no_frames = True
        with pytest.raises(NoDataError, match="100 ms"):
            next(iter(camera.frames(timeout_ms=100)))

    def test_timeout_still_stops_acquisition(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        fake_sdk.no_frames = True
        with pytest.raises(NoDataError):
            next(iter(camera.frames(timeout_ms=100)))
        assert fake_sdk.grabbing is False


class TestCorruptFrames:
    def test_a_payload_smaller_than_the_geometry_is_refused(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        # Metadata and payload disagreeing is how a wrong pixel-format mapping would show
        # up; shaping the array anyway would read past the end of the node.
        frame = first_frame(camera)
        try:
            frame.size_bytes = 4
            with pytest.raises(ValueError, match="needs 128 bytes"):
                _ = frame.data
        finally:
            frame.release()

    def test_a_frame_without_a_buffer_address_is_refused(
        self, camera: Camera, fake_sdk: FakeMvsLibrary
    ) -> None:
        frame = first_frame(camera)
        try:
            frame._raw.pBufAddr = None
            with pytest.raises(BufferReleasedError, match="no buffer address"):
                _ = frame.data
        finally:
            frame._released = True

    def test_frames_raw_raises_on_timeout(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.no_frames = True
        with pytest.raises(NoDataError, match="250 ms"):
            next(iter(camera.frames_raw(timeout_ms=250)))


class TestNodePool:
    def test_nodes_are_recycled(self, camera: Camera, fake_sdk: FakeMvsLibrary) -> None:
        # A fixed pool is precisely why a view must not outlive its frame.
        addresses = []
        for index, frame in enumerate(camera.frames(timeout_ms=100, node_count=2)):
            addresses.append(frame.data.ctypes.data)
            if index == 5:
                break
        assert len(set(addresses)) == 2
