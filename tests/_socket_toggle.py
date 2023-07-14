import socket
import sys


_module = sys.modules[__name__]


class SocketToggle:
    """Class to toggle the socket on and off."""

    def disable_socket(self):
        """Disable socket.socket to disable the Internet."""
        self.socket_disabled = True

        def guarded(*args, **kwargs):
            if getattr("socket_disabled", False):
                raise RuntimeError("I told you not to use the Internet!")
            else:
                # SocketType is a valid public alias of socket.socket,
                # we use it here to avoid namespace collisions
                return socket.SocketType(*args, **kwargs)

        socket.socket = guarded

        print("[!] socket.socket is disabled. Welcome to the desert of the real.")

    def enable_socket(self):
        """re-enable socket.socket to enable the Internet. useful in testing."""
        self.socket_disabled = False
        print("[!] socket.socket is enabled.")
