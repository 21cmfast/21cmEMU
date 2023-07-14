"""Conftest.py."""
import pytest

from ._socket_toggle import SocketToggle


@pytest.fixture(scope="function")
def disable_socket(request):
    """Disable socket.socket for duration of this test function."""
    SocketToggle().disable_socket()
    # request.addfinalizer(_socket_toggle.enable_socket)
