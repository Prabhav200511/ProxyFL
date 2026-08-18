# network.py — TCP message passing for the ProxyFL VANET simulation
import socket
import pickle
import struct
import threading


def send_msg(addr, msg):
    """Send a pickled message to (host, port) with a 4-byte length prefix.

    Returns True on success, False on failure (logged to stderr).
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10)
            s.connect(addr)
            data = pickle.dumps(msg)
            s.sendall(struct.pack('>I', len(data)) + data)
            return True
    except ConnectionRefusedError:
        print(f"[NET] Connection refused to {addr[1]} "
              f"(target may be offline)")
        return False
    except Exception as e:
        print(f"[NET] Send failed to {addr[1]}: {e}")
        return False


class Receiver:
    """TCP server that deserializes incoming messages and dispatches them
    to a callback function.
    """

    def __init__(self, port, callback):
        self.port = port
        self.callback = callback
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.listen()

    def start(self):
        threading.Thread(target=self._listen, daemon=True).start()

    def shutdown(self):
        """Close the listening socket so the port is freed."""
        try:
            self.sock.close()
        except Exception:
            pass

    def _listen(self):
        while True:
            try:
                conn, _ = self.sock.accept()
                threading.Thread(
                    target=self._handle, args=(conn,), daemon=True).start()
            except OSError:
                break

    def _handle(self, conn):
        try:
            raw_msglen = self._recvall(conn, 4)
            if not raw_msglen:
                return
            msglen = struct.unpack('>I', raw_msglen)[0]
            data = self._recvall(conn, msglen)
            if data:
                self.callback(pickle.loads(data))
        except Exception as e:
            print(f"[NET] Receive error on port {self.port}: {e}")
        finally:
            conn.close()

    def _recvall(self, conn, n):
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data