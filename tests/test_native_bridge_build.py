"""The optional Windows bridge must build from the retained source subset."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


VCVARS = Path(
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)


@unittest.skipUnless(os.name == "nt" and VCVARS.is_file(), "requires Windows MSVC")
class NativeBridgeBuildTests(unittest.TestCase):
    def test_bridge_builds_without_generated_headers_or_obsolete_mirrors(self):
        # Catches compiling the upstream @WL@ template without configuring it,
        # or accidentally depending on untracked local native build artifacts.
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="proxyfl native build ") as temp:
            isolated = Path(temp)
            vendor = isolated / "core-master" / "c"
            bridge = isolated / "crypto_protocol"
            vendor.mkdir(parents=True)
            bridge.mkdir()
            for name in ("hash.c", "aes.c", "gcm.c", "core.h", "arch.h"):
                shutil.copy2(repository / "core-master" / "c" / name, vendor / name)
            for pattern in ("*.bat", "*.py", "miracl_core_bridge.c"):
                for source in (repository / "crypto_protocol").glob(pattern):
                    shutil.copy2(source, bridge / source.name)

            result = subprocess.run(
                ["cmd.exe", "/d", "/c", str(bridge / "build_miracl_bridge.bat")],
                cwd=isolated, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            dll = bridge / "miracl_core.dll"
            self.assertTrue(dll.is_file(), "build succeeded without producing the DLL")
            # Load in a child so Windows releases the DLL before temp cleanup.
            probe = subprocess.run(
                [sys.executable, "-B", "-c", """
import ctypes
import sys
library = ctypes.CDLL(sys.argv[1])
function = library.proxyfl_miracl_sha256
pointer = ctypes.POINTER(ctypes.c_ubyte)
function.argtypes = [pointer, ctypes.c_int, pointer]
function.restype = None
source = (ctypes.c_ubyte * 3)(*b'abc')
digest = (ctypes.c_ubyte * 32)()
function(source, 3, digest)
assert bytes(digest).hex() == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
assert library.proxyfl_miracl_gcm_encrypt
assert library.proxyfl_miracl_gcm_decrypt
""", str(dll)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)


if __name__ == "__main__":
    unittest.main()
