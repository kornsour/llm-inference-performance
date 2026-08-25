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
import os
import re
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

# torch derives -gencode flags from the *local* GPU; with no device it raises
# IndexError deep inside _get_cuda_arch_flags. Naming the targets explicitly is
# what makes a GPU-less build possible at all. These are the cards this repo is
# actually run on -- 8.6 Ampere (RTX 3060 Ti), 8.9 Ada (RTX 4070) -- plus PTX so
# the result stays forward-compatible.
DEFAULT_ARCH_LIST = "8.6;8.9+PTX"


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


def _require_toolchain() -> str:
    if shutil.which("nvcc") is None:
        sys.exit("nvcc not found on PATH — install the CUDA toolkit (no GPU needed).")
    import torch
    if not hasattr(torch.version, "cuda") or torch.version.cuda is None:
        sys.exit(f"this torch build has no CUDA support (torch {torch.__version__}); "
                 "install a CUDA wheel, e.g. --index-url .../whl/cu124")
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST") or DEFAULT_ARCH_LIST
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    print(f"torch {torch.__version__} (cuda {torch.version.cuda}), "
          f"nvcc {shutil.which('nvcc')}, arch {arch}")
    return arch


_NOISE = re.compile(r"Error compiling objects|raise RuntimeError|subprocess\.")


def _diagnostics(log: str, n: int = 60) -> str:
    """The compiler's own complaint, not the Python traceback that reports it.

    setuptools re-raises ninja failures as `RuntimeError: Error compiling
    objects for extension`, so the last lines of the log are always the same
    uninformative traceback while the nvcc/gcc diagnostic sits above it.
    """
    lines = [ln for ln in log.splitlines() if ln.strip()]
    hits = [i for i, ln in enumerate(lines)
            if re.search(r"\berror\b|^FAILED:", ln, re.I) and not _NOISE.search(ln)]
    if hits:
        lines = lines[max(0, hits[0] - 3):min(len(lines), hits[-1] + 4)]
    return "\n".join(f"    {ln}" for ln in lines[:n])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="also assert the check FAILS on each known-bad wiring")
    args = ap.parse_args(argv)

    _require_toolchain()
    tmp = Path(tempfile.mkdtemp(prefix="llminf-kernel-build-"))
    failures: list[str] = []

    # A guard that cannot fail proves nothing -- but neither does one that always
    # fails. Each case names the diagnostic that must appear, so a build broken
    # for an unrelated reason is reported as inconclusive rather than counted as
    # a pass. (This is not hypothetical: a missing TORCH_CUDA_ARCH_LIST once made
    # every variant fail, and an earlier version of this script called that a
    # clean sweep.)
    cases = [
        ("duplicate PYBIND11_MODULE in the CUDA source",
         rms._CPP_SRC,
         rms._CUDA_SRC + """
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "Fused RMSNorm forward (CUDA)");
}
""",
         ("multiple definition", "PyInit_llminf_rmsnorm", "redefinition")),
        ("cpp_sources missing the declaration",
         "#include <torch/extension.h>\n",
         rms._CUDA_SRC,
         ("was not declared", "undeclared identifier", "has not been declared")),
    ]

    try:
        print("\n[1] building the shipped sources ...")
        r = build(rms._CPP_SRC, rms._CUDA_SRC, tmp / "real")
        if r.ok:
            print(f"    OK -> {r.artifact.name}")
        else:
            failures.append("the shipped kernel sources do not build")
            print("    FAILED\n" + _diagnostics(r.log))

        if args.self_test:
            for i, (label, cpp, cuda, expected) in enumerate(cases, start=2):
                print(f"\n[{i}] self-test: {label} ...")
                res = build(cpp, cuda, tmp / f"bad{i}")
                if res.ok:
                    failures.append(f"{label!r} built successfully — this check "
                                    "would not catch the original defect")
                    print("    UNEXPECTEDLY OK")
                    continue
                hit = next((tok for tok in expected if tok.lower() in res.log.lower()), None)
                if hit:
                    print(f"    rejected for the right reason ({hit!r})")
                else:
                    failures.append(
                        f"{label!r} was rejected, but for none of {expected} — the "
                        "self-test cannot confirm this guard catches the real defect")
                    print("    REJECTED FOR THE WRONG REASON\n" + _diagnostics(res.log, 25))
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
