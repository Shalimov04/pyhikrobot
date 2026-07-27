# SPDX-License-Identifier: Apache-2.0
"""The package must import with no SDK and no camera present.

This is what lets CI, docs builds and hardware-less containers work, so it is guarded by
a test rather than by convention.
"""

from __future__ import annotations

import subprocess
import sys

import hikrobot


def test_public_names_are_importable() -> None:
    for name in hikrobot.__all__:
        assert hasattr(hikrobot, name), name


def test_version_is_set() -> None:
    assert hikrobot.__version__


def test_import_does_not_touch_the_sdk() -> None:
    # A fresh interpreter: an SDK loaded by an earlier test in this process would hide a
    # regression here. MVCAM_SDK_PATH is pointed at nothing on purpose.
    code = (
        "import os; os.environ['MVCAM_SDK_PATH'] = '/nonexistent-sdk-root';"
        "import hikrobot; import hikrobot._loader as loader;"
        "assert loader._library is None, 'the SDK was loaded at import time';"
        "print(hikrobot.__version__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == hikrobot.__version__
