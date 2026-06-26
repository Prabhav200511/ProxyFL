# network.py
import socket
import pickle
import struct
import threading

def send_msg(addr, msg):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(addr)
            data = pickle.dumps(msg)
            s.sendall(struct.pack('>I', len(data)) + data)
    except Exception as e:
        pass

class Receiver:
    def __init__(self, port, callback):
        self.port = port
        self.callback = callback
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.listen()

    def start(self):
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while True:
            conn, _ = self.sock.accept()
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            raw_msglen = self._recvall(conn, 4)
            if not raw_msglen:
                return
            msglen = struct.unpack('>I', raw_msglen)[0]
            data = self._recvall(conn, msglen)
            if data:
                self.callback(pickle.loads(data))
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