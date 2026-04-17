"""Pytest configuration for py21cmemu tests."""

from __future__ import annotations

import os

import pytest


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
    # Check if we're in CI and if this is a merge to main
    is_ci = os.environ.get("CI", "false").lower() == "true"
    is_main_merge = os.environ.get("GITHUB_EVENT_NAME") == "push" and (
        os.environ.get("GITHUB_REF") == "refs/heads/main"
        or os.environ.get("GITHUB_BASE_REF") == "main"
    )
    
    run_slow = config.getoption("--run-slow")
    
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_main_only = pytest.mark.skip(
        reason="only runs on merge to main (set GITHUB_REF=refs/heads/main or use --run-slow)"
    )
    
    for item in items:
        # Handle slow tests
        if "slow" in item.keywords:
            if not run_slow:
                item.add_marker(skip_slow)
        
        # Handle main-only tests
        if "main_only" in item.keywords:
            if not (run_slow or (is_ci and is_main_merge)):
                item.add_marker(skip_main_only)
