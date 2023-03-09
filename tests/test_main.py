"""Test cases for the __main__ module."""
import numpy as np

from py21cmemu import Emulator


def test_basic_prediction():
    """Simply test that we can make a prediction without erroring."""
    emu = Emulator(version="latest")
    theta = np.random.rand(9 * 5).reshape((5, 9))
    emu.predict(theta)
