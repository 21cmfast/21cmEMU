"""Pytest configuration for py21cmemu tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from py21cmemu import Emulator
    from py21cmemu.outputs import MHEmulatorOutput
    from py21cmemu.properties import MHEmulatorProperties


# ══════════════════════════════════════════════════════════════════════════════
# Test Data Paths
# ══════════════════════════════════════════════════════════════════════════════

# All test data files live in docs/tutorials/
_TUTORIALS_DIR = Path(__file__).parent.parent / "docs" / "tutorials"

PS_2D_TEST_H5 = _TUTORIALS_DIR / "ps_2d_test_subsample.h5"
PS_1D_LOGLIN_H5 = _TUTORIALS_DIR / "ps_1d_loglin_db_test.h5"
TEST_DATABASE_H5 = _TUTORIALS_DIR / "test_database.h5"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (e.g., score model evaluation)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "main_only: marks tests to only run on merge to main",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection based on markers and options."""
    # Check if we're running in CI
    is_ci = os.environ.get("CI", "false").lower() == "true"

    run_slow = config.getoption("--run-slow")

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_main_only = pytest.mark.skip(
        reason="only runs in CI or locally with --run-slow"
    )

    for item in items:
        # Handle slow tests
        if "slow" in item.keywords:
            if not run_slow:
                item.add_marker(skip_slow)

        # Handle main-only tests (run in any CI environment or with --run-slow)
        if "main_only" in item.keywords:
            if not (run_slow or is_ci):
                item.add_marker(skip_main_only)


# ══════════════════════════════════════════════════════════════════════════════
# Shared Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def test_set_h5_path() -> Path:
    """Return path to test_database.h5."""
    return TEST_DATABASE_H5


@pytest.fixture(scope="session")
def ps_2d_test_h5_path() -> Path:
    """Return path to ps_2d_test_subsample.h5."""
    return PS_2D_TEST_H5


@pytest.fixture(scope="session")
def test_params() -> np.ndarray:
    """Load test parameters from test_database.h5 (already in log-space)."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(TEST_DATABASE_H5, "r") as f:
        return np.asarray(f["input_params"][:5], dtype=float)


@pytest.fixture(scope="session")
def single_test_param() -> np.ndarray:
    """Load a single test parameter from test_database.h5 (already in log-space)."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(TEST_DATABASE_H5, "r") as f:
        return np.asarray(f["input_params"][:1], dtype=float)


@pytest.fixture(scope="module")
def mh_emulator() -> Emulator:
    """Create MH emulator without 2D PS (fast)."""
    from py21cmemu import Emulator

    return Emulator(emulator="mcg", emulate_2d_ps=False)


@pytest.fixture(scope="module")
def mh_emulator_with_2d_ps() -> Emulator:
    """Create MH emulator with 2D PS enabled (slow)."""
    from py21cmemu import Emulator

    return Emulator(emulator="mcg", emulate_2d_ps=True)


@pytest.fixture(scope="module")
def mh_properties() -> MHEmulatorProperties:
    """Get MH emulator properties."""
    from py21cmemu.properties import get_emulator_properties

    return get_emulator_properties(emulator="mcg")


@pytest.fixture(scope="module")
def mh_output_no_2d_ps(mh_emulator, single_test_param) -> MHEmulatorOutput:
    """Get MH emulator output without 2D PS."""
    _, output, _ = mh_emulator.predict(single_test_param)
    return output


@pytest.fixture(scope="module")
def test_db_params() -> np.ndarray:
    """Load one parameter set from test_database.h5 (pre-converted, no log needed)."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(TEST_DATABASE_H5, "r") as f:
        return np.asarray(f["input_params"][:1], dtype=float)
