"""Test cases for the __main__ module."""
import pytest
from emu21cmfast import EMU21cmFAST
import numpy as np

def test_basic_prediction():
    emu = EMU21cmFAST(version='latest')
    theta = np.random.rand(9*5).reshape((5,9))
    emu.predict(theta)