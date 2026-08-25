"""Compile-and-link the fused RMSNorm CUDA extension. No GPU required.

The two ways `load_inline` wiring goes wrong — a duplicate `PYBIND11_MODULE` in
the CUDA source, and a `cpp_sources` that never declares the exported function —
are a link error and a compile error respectively. Neither needs a GPU to
detect, only a CUDA toolchain, which is why this runs on an ordinary CI runner.

It reproduces exactly what `load_inline` emits (see `_generated_main_cpp` /
`_generated_cuda_cu`, mirroring torch/utils/cpp_extension.py) and drives it
through setuptools + `BuildExtension`, stopping at the built `.so`. It never
imports the result: importing would pull in `libcuda.so.1`, which ships with the
driver and is absent on a GPU-less machine. Building does not need it.

    python scripts/check_kernel_builds.py              # build the real sources
    python scripts/check_kernel_builds.py --self-test  # + prove it catches both bugs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llminf import rmsnorm as rms  # noqa: E402

FUNCTIONS = ["rmsnorm_forward"]


def _generated_main_cpp(cpp_sources: str, functions: list[str]) -> str:
    """The C++ translation unit `load_inline` writes to main.cpp."""
    parts = [cpp_sources, "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {"]
    parts += [f'm.def("{fn}", torch::wrap_pybind_function({fn}), "{fn}");' for fn in functions]
    parts.append("}")
    return "\n".join(parts)


def _generated_cuda_cu(cuda_sources: str) -> str:
    """The CUDA translation unit, with the prologue `load_inline` prepends."""
    return "\n".join([
        "#include <torch/types.h>",
        "#include <cuda.h>",
        "#include <cuda_runtime.h>",
        cuda_sources,
    ])


_SETUP_PY = '''
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

link_args = []
# libtorch_cuda.so records libcuda.so.1 as a dependency; the toolkit ships a
# link-time stub so a driverless machine can still link against it.
for sub in ("lib64/stubs", "lib/stubs"):
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    stubs = os.path.join(cuda_home, sub)
    if os.path.isdir(stubs):
        link_args += ["-L" + stubs]

setup(
    name="llminf_rmsnorm",
    ext_modules=[CUDAExtension("llminf_rmsnorm", ["main.cpp", "cuda.cu"],
                               extra_link_args=link_args)],
    cmdclass={"build_ext": BuildExtension},
)
'''


@dataclass
class BuildResult:
    ok: bool
    log: str
    artifact: Path | None


def build(cpp_sources: str, cuda_sources: str, workdir: Path) -> BuildResult:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "main.cpp").write_text(_generated_main_cpp(cpp_sources, FUNCTIONS))
    (workdir / "cuda.cu").write_text(_generated_cuda_cu(cuda_sources))
    (workdir / "setup.py").write_text(_SETUP_PY)

    proc = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=workdir, capture_output=True, text=True,
    )
    log = proc.stdout + proc.stderr
    built = sorted(workdir.glob("llminf_rmsnorm*.so"))
    ok = proc.returncode == 0 and bool(built)
    return BuildResult(ok, log, built[0] if built else None)


def _require_toolchain() -> None:
    if shutil.which("nvcc") is None:
        sys.exit("nvcc not found on PATH — install the CUDA toolkit (no GPU needed).")
    import torch
    if not hasattr(torch.version, "cuda") or torch.version.cuda is None:
        sys.exit(f"this torch build has no CUDA support (torch {torch.__version__}); "
                 "install a CUDA wheel, e.g. --index-url .../whl/cu124")
    print(f"torch {torch.__version__} (cuda {torch.version.cuda}), "
          f"nvcc {shutil.which('nvcc')}")


def _tail(log: str, n: int = 40) -> str:
    lines = [ln for ln in log.splitlines() if ln.strip()]
    return "\n".join(f"    {ln}" for ln in lines[-n:])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="also assert the check FAILS on each known-bad wiring")
    args = ap.parse_args(argv)

    _require_toolchain()
    tmp = Path(tempfile.mkdtemp(prefix="llminf-kernel-build-"))
    failures: list[str] = []

    try:
        print("\n[1] building the shipped sources ...")
        r = build(rms._CPP_SRC, rms._CUDA_SRC, tmp / "real")
        if r.ok:
            print(f"    OK -> {r.artifact.name}")
        else:
            failures.append("the shipped kernel sources do not build")
            print("    FAILED\n" + _tail(r.log))

        if args.self_test:
            # A guard that cannot fail proves nothing. Re-introduce each defect
            # and require the build to reject it.
            print("\n[2] self-test: duplicate PYBIND11_MODULE in the CUDA source ...")
            dup = rms._CUDA_SRC + """
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "Fused RMSNorm forward (CUDA)");
}
"""
            r2 = build(rms._CPP_SRC, dup, tmp / "dup")
            if r2.ok:
                failures.append("a duplicate PYBIND11_MODULE built successfully — "
                                "this check would not catch the original defect")
                print("    UNEXPECTEDLY OK")
            else:
                print("    rejected, as required")

            print("\n[3] self-test: cpp_sources missing the declaration ...")
            r3 = build("#include <torch/extension.h>\n", rms._CUDA_SRC, tmp / "nodecl")
            if r3.ok:
                failures.append("an undeclared rmsnorm_forward built successfully — "
                                "this check would not catch the original defect")
                print("    UNEXPECTEDLY OK")
            else:
                print("    rejected, as required")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("kernel build check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
