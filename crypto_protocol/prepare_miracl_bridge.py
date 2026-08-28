"""Configure the minimal x64 MIRACL bridge build without editing vendor sources."""

from pathlib import Path
import shutil


def prepare_build() -> Path:
    bridge_root = Path(__file__).resolve().parent
    source_root = bridge_root.parent / "core-master" / "c"
    build_root = bridge_root / "build"
    build_root.mkdir(exist_ok=True)

    # Quoted C includes search the source file's directory first. Compile
    # copies beside the configured arch.h rather than the upstream template.
    for name in ("hash.c", "aes.c", "gcm.c", "core.h"):
        shutil.copyfile(source_root / name, build_root / name)
    architecture = (source_root / "arch.h").read_text(encoding="utf-8")
    architecture = architecture.replace("@WL@", "64").replace("@CORE_CHUNK@", "64")
    (build_root / "arch.h").write_text(architecture, encoding="utf-8")
    return build_root


if __name__ == "__main__":
    prepare_build()
