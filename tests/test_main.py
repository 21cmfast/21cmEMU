"""Test cases for the __main__ module."""
import numpy as np

from py21cmemu import py21cmEMU


def test_basic_prediction():
    """Simply test that we can make a prediction without erroring."""
    emu = EMU21cmFAST(version="latest")
    theta = np.random.rand(9 * 5).reshape((5, 9))
    emu.predict(theta)
