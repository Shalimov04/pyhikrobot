# SPDX-License-Identifier: Apache-2.0
"""Enumeration, against the fake library at the CDLL boundary."""

from __future__ import annotations

import pytest

from conftest import FakeMvsLibrary, gige_device, usb_device
from hikrobot import DeviceInfo, enumerate_devices
from hikrobot._ctypes_defs import (
    MV_CC_DEVICE_INFO,
    MV_GIGE_DEVICE,
    MV_USB_DEVICE,
    MV_VIR_GIGE_DEVICE,
)
from hikrobot._errors import AccessDeniedError, ParameterError


class TestEnumerate:
    def test_no_devices_is_an_empty_list_not_an_error(self, fake_sdk: FakeMvsLibrary) -> None:
        assert enumerate_devices() == []

    def test_multiple_devices_keep_the_sdk_order(self, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.devices = [
            gige_device(serial="AAA"),
            usb_device(serial="BBB"),
            gige_device(serial="CCC"),
        ]
        assert [device.serial_number for device in enumerate_devices()] == ["AAA", "BBB", "CCC"]

    def test_gige_fields_are_decoded(self, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.devices = [gige_device(model="MV-CA050-10GM", serial="GIGE42", ip=(10, 0, 3, 7))]
        (device,) = enumerate_devices()

        assert device.transport == "gige"
        assert device.is_virtual is False
        assert device.model_name == "MV-CA050-10GM"
        assert device.serial_number == "GIGE42"
        assert device.manufacturer_name == "Hikrobot"
        assert device.device_version == "V1.2.3"
        assert device.ip_address == "10.0.3.7"
        assert device.mac_address == "00:ab:cd:ef:01:23"

    def test_usb_fields_are_decoded_and_carry_no_network_details(
        self, fake_sdk: FakeMvsLibrary
    ) -> None:
        fake_sdk.devices = [usb_device(model="MV-CU060-10UM", serial="USB7")]
        (device,) = enumerate_devices()

        assert device.transport == "usb"
        assert device.model_name == "MV-CU060-10UM"
        assert device.serial_number == "USB7"
        assert device.ip_address is None
        assert device.mac_address is None

    def test_virtual_devices_are_flagged(self, fake_sdk: FakeMvsLibrary) -> None:
        raw = gige_device(serial="VIRT")
        raw.nTLayerType = MV_VIR_GIGE_DEVICE
        fake_sdk.devices = [raw]

        (device,) = enumerate_devices()
        assert device.transport == "gige"
        assert device.is_virtual is True

    def test_undecoded_transport_is_reported_not_dropped(self, fake_sdk: FakeMvsLibrary) -> None:
        raw = MV_CC_DEVICE_INFO()
        raw.nTLayerType = 0x00000100  # GenTL CoaXPress
        fake_sdk.devices = [raw]

        (device,) = enumerate_devices()
        assert device.transport == "unknown"
        assert device.model_name == ""

    def test_a_null_entry_in_the_list_is_skipped(self, fake_sdk: FakeMvsLibrary) -> None:
        # nDeviceNum can outrun the pointers the SDK actually filled in; dereferencing a
        # null entry would be a segfault rather than an exception.
        fake_sdk.devices = [gige_device(serial="REAL")]
        fake_sdk.null_entries = 1

        assert [device.serial_number for device in enumerate_devices()] == ["REAL"]

    def test_name_falls_back_from_user_defined_to_model(self, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.devices = [
            gige_device(model="MV-A", user_defined_name=""),
            gige_device(model="MV-B", user_defined_name="left-camera"),
        ]
        assert [device.name for device in enumerate_devices()] == ["MV-A", "left-camera"]

    def test_undecodable_name_bytes_do_not_break_enumeration(
        self, fake_sdk: FakeMvsLibrary
    ) -> None:
        raw = gige_device()
        raw.SpecialInfo.stGigEInfo.chUserDefinedName[0] = 0xFF
        raw.SpecialInfo.stGigEInfo.chUserDefinedName[1] = 0x00
        fake_sdk.devices = [raw]

        (device,) = enumerate_devices()
        assert device.user_defined_name  # replaced, not raised


class TestTransportSelection:
    def test_default_scans_gige_and_usb_including_virtual(self, fake_sdk: FakeMvsLibrary) -> None:
        enumerate_devices()
        (mask,) = fake_sdk.enumerated_masks
        for bit in (MV_GIGE_DEVICE, MV_USB_DEVICE, MV_VIR_GIGE_DEVICE):
            assert mask & bit

    def test_a_single_transport_narrows_the_mask(self, fake_sdk: FakeMvsLibrary) -> None:
        enumerate_devices(["usb"])
        (mask,) = fake_sdk.enumerated_masks
        assert mask & MV_USB_DEVICE
        assert not mask & MV_GIGE_DEVICE

    def test_unknown_transport_is_rejected_before_the_sdk_is_called(
        self, fake_sdk: FakeMvsLibrary
    ) -> None:
        with pytest.raises(ValueError, match="cameralink"):
            enumerate_devices(["cameralink"])
        assert fake_sdk.enumerated_masks == []

    def test_empty_transport_list_is_rejected(self, fake_sdk: FakeMvsLibrary) -> None:
        with pytest.raises(ValueError, match="at least one"):
            enumerate_devices([])


class TestFailures:
    def test_sdk_refusal_is_mapped_to_a_typed_exception(self, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.statuses["MV_CC_EnumDevices"] = 0x80000004
        with pytest.raises(ParameterError):
            enumerate_devices()

    def test_status_carries_the_failing_entry_point(self, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.statuses["MV_CC_EnumDevices"] = 0x80000203
        with pytest.raises(AccessDeniedError) as excinfo:
            enumerate_devices()
        assert excinfo.value.operation == "MV_CC_EnumDevices"


class TestDeviceInfoSnapshot:
    def test_the_record_survives_a_second_enumeration(self, fake_sdk: FakeMvsLibrary) -> None:
        # The SDK reuses its own structures; a DeviceInfo that merely pointed at them
        # would start describing a different camera after the next scan.
        fake_sdk.devices = [gige_device(model="first", serial="S1")]
        (first,) = enumerate_devices()

        reused = fake_sdk.devices[0]
        reused.SpecialInfo.stGigEInfo.chModelName[0] = ord("X")
        fake_sdk.devices = [reused]
        enumerate_devices()

        assert first.model_name == "first"
        assert bytes(first.raw.SpecialInfo.stGigEInfo.chModelName).startswith(b"first")

    def test_two_devices_compare_by_their_public_fields(self, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.devices = [gige_device(serial="SAME"), gige_device(serial="SAME")]
        first, second = enumerate_devices()
        # The raw record is excluded from comparison; two identical cameras compare equal.
        assert first == second

    def test_repr_hides_the_raw_record(self, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.devices = [gige_device(serial="R1")]
        (device,) = enumerate_devices()
        assert "R1" in repr(device)
        assert "MV_CC_DEVICE_INFO" not in repr(device)

    def test_is_a_frozen_dataclass(self, fake_sdk: FakeMvsLibrary) -> None:
        fake_sdk.devices = [gige_device()]
        (device,) = enumerate_devices()
        with pytest.raises(AttributeError):
            device.serial_number = "nope"  # type: ignore[misc]
        assert isinstance(device, DeviceInfo)
