#!/usr/bin/env python3
"""Compile standalone wire code and exercise partial framed reads twice/session.

This does NOT compile or validate the OpenFOAM plugin itself.
"""
import json
import argparse
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading

HERE = Path(__file__).resolve().parent

def exact(sock, n):
    out = b""
    while len(out) < n:
        data = sock.recv(n-len(out))
        if not data:
            raise EOFError("unexpected socket close")
        out += data
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc", action="store_true", help="Also create a Unix socket and test persistent framing")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="h2-wire-") as td:
        root = Path(td)
        exe = root / "wire-test"
        subprocess.run(["c++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                        str(HERE / "wire_test.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
        if not args.ipc:
            return
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        path = root / "solver.sock"
        server.bind(str(path))
        server.listen(1)
        errors = []
        def serve():
            try:
                with server.accept()[0] as client:
                    for _ in range(2):
                        n = struct.unpack("!Q", exact(client, 8))[0]
                        assert json.loads(exact(client, n)) == {"n": 2}
                        data = json.dumps(dict(success=True, x_native=[1., -.25], cycles=3,
                              threshold=1e-9, initial_residual=1., final_residual=1e-10)).encode()
                        frame = struct.pack("!Q", len(data)) + data
                        for start in range(0, len(frame), 7):
                            client.sendall(frame[start:start+7])
            except BaseException as exc:
                errors.append(exc)
        t = threading.Thread(target=serve, daemon=True)
        t.start()
        subprocess.run([str(exe), str(path)], check=True, timeout=15)
        t.join(timeout=5)
        server.close()
        if errors:
            raise errors[0]
        assert not t.is_alive()

if __name__ == "__main__":
    main()
