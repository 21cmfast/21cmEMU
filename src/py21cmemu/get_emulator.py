"""Download and install the emulator data."""
from __future__ import annotations

import logging

import git

from .config import CONFIG


log = logging.getLogger(__name__)


def get_emu_data(version: str = "latest"):
    """Download a version of the 21cmEMU emulator.

    Parameters
    ----------
    version : str, optional
        When multiple versions will be available, one will be able to specify
        the version number instead of the link. Default is 'latest'.
    """
    if (CONFIG.data_path / "21cmEMU").exists():
        repo = git.Repo(CONFIG.data_path / "21cmEMU")
    else:
        URL = "https://huggingface.co/DanielaBreitman/21cmEMU"
        repo = git.Repo.clone_from(URL, CONFIG.data_path / "21cmEMU")

    # Pull latest changes
    repo.remotes.origin.pull()

    versions = sorted(
        [tag.name.lower() for tag in repo.tags if tag.name.lower().startswith("v")]
    )

    if version == "latest":
        version = repo.git.checkout("main")
    elif version.lower() in versions:
        # Checkout the version
        repo.git.checkout(version)
    else:
        raise ValueError(
            f"Version {version} not available. Must be one of {versions}. "
        )
