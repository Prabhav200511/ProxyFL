# MIRACL Core subset retained by ProxyFL

ProxyFL keeps the upstream `c/` directory for the optional native SHA-256 and
AES-GCM bridge. `crypto_protocol/build_miracl_bridge.bat` compiles `hash.c`,
`aes.c`, and `gcm.c`; these use `arch.h` and `core.h` from this directory.
The build helper configures the upstream `arch.h` template for x64 in an
ignored build directory, leaving the vendor originals unchanged. Building
requires Python 3 and the Visual Studio 2022 Community C++ toolchain.
The upstream license, notices, contributor information, and reference documents
are retained without changes.

The unused Arduino, C++, Go, Java, JavaScript, Python, Rust, Swift, tools, and
WebAssembly directories were removed from this vendored copy. The upstream
readme still describes the complete upstream distribution.

Runtime NIST256 elliptic-curve arithmetic lives in `../miracl_python/`, not in
the removed upstream Python examples. No FL algorithm or cryptographic
operation was changed by this cleanup. The active native DLL is kept locally
and ignored by Git; users can rebuild it with the existing build script.
