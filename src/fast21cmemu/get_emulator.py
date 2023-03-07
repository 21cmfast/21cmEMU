"""Download and install the emulator data."""
from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import CONFIG
from .config import LATEST


log = logging.getLogger(__name__)


def download_emu_data(version: str = "latest"):
    """Download a version of the 21cmEMU emulator.

    Parameters
    ----------
    destination_dir : str, optional
        path where to download 21cmEMU. Default is in the current directory.
    version : str, optional
        When multiple versions will be available, one will be able to specify
        the version number instead of the link. Default is 'latest'.
    """
    if version == "latest":
        version = LATEST

    if version in CONFIG["emu-versions"]:
        log.warning(f"Emulator version {version} already exists. Not downloading!")
        return

    urls = {
        "v1": "https://www.dropbox.com/s/rv55tetjy22lple/21cmEMU.zip?dl=1",
    }
    if version not in urls:
        raise ValueError(
            f"Version {version} not available to download. Must be one of "
            f"{list(urls.keys())}. You currently have the following versions already "
            f"installed: {CONFIG['emu-versions']}."
        )
    url = urls[version]

    dest = CONFIG.data_path

    # download and extract the emulator
    zipf = dest / "zipped_emulator"
    outd = dest / "21cmEMU"

    try:
        urllib.request.urlretrieve(url, zipf)
        if zipf.exists() and zipf.is_file():
            log.info("Downloaded the emulator successfully")
    except Exception as e:
        raise ValueError("Download failed!") from e

    try:
        with zipfile.ZipFile(zipf, "r") as zip_ref:
            zip_ref.extractall(dest)
        if outd.exists() and outd.is_dir():
            log.info("Extracted the emulator successfully")
    except Exception as e:
        raise OSError("The downloaded file is not in the correct format!") from e

    # Now remove the zipfile
    zipf.unlink()

    # And move the emulator to a location tagged with the version
    outd.rename(dest / version)
    CONFIG.add_emulator(version)


def move_emulator_data(dest: str | Path):
    """Move the emulator data to a new location."""
    src = Path(CONFIG["data-path"])

    dest = Path(dest)
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
        CONFIG["data-path"] = str(dest)

    if src == dest:
        log.info("Emulator data already in the desired location.")
        return

    # Move the emulator data
    for f in src.glob("*"):
        f.rename(dest / f.name)

    # Remove the old directory
    src.rmdir()


def add_emulator_data(emu_data: str | Path, label: str):
    """Add arbitrary emulator data to the known emulator data cache, with a label."""
    emu_data = Path(emu_data)
    if not emu_data.exists():
        raise ValueError("The emulator data does not exist.")

    if label in CONFIG["emu-versions"]:
        raise ValueError("The label already exists.")

    CONFIG.add_emulator(label)
    dest = CONFIG.get_emulator(label)

    shutil.copytree(emu_data, dest)
