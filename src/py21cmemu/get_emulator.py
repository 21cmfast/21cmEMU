"""Download and install the emulator data."""
from __future__ import annotations

import logging
from warnings import warn

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
    elif not CONFIG["disable-network"]:
        URL = "https://huggingface.co/DanielaBreitman/21cmEMU"
        repo = git.Repo.clone_from(URL, CONFIG.data_path / "21cmEMU")

    # Check download
    p = CONFIG.data_path / "21cmEMU" / "21cmEMU" / "saved_model.pb"
    if not p.exists() or p.stat().st_size < 1e6:
        raise RuntimeError(
            "The emulator huggingface repo was not cloned properly.\n"
            "Check that git-lfs is installed properly on your system.\n"
            "If git-lfs cannot be installed or internet "
            "connection is not available, "
            "manually clone the repo on another machine with git-lfs"
            "and internet using\n"
            "git clone -v -- https://huggingface.co/DanielaBreitman/21cmEMU\n"
            "Then, ensure that it downloaded fully by running: "
            "du -sh 21cmEMU \n"
            "The folder should be about 500M. "
            "Now copy this folder and its contents "
            "over to your other machine and put it in "
            " ~/.local/share/py21cmEMU/21cmEMU "
        )

    # Pull latest changes
    try:
        if CONFIG["disable-network"]:
            raise RuntimeError("Network is disabled via config")

        repo.remotes.origin.pull()
    except Exception as e:
        warn(f"Skipping the pulling step. Error received: {e}", stacklevel=2)

    versions = sorted(
        [tag.name.lower() for tag in repo.tags if tag.name.lower().startswith("v")]
    )

    if version == "latest":
        version = repo.git.checkout("main")
    elif version.lower() in versions:
        repo.git.checkout(version)
    else:
        raise ValueError(
            f"Version {version} not available. Must be one of {versions}. "
        )
