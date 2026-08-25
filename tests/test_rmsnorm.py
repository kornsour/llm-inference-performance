from pathlib import Path

import pytest
import torch

from llminf import rmsnorm as rms
from llminf.rmsnorm import backend, rmsnorm, rmsnorm_reference

KERNEL_CU = Path(__file__).resolve().parents[1] / "kernels" / "rmsnorm_kernel.cu"


def test_reference_matches_manual():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    w = torch.randn(32)
    eps = 1e-6
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    torch.testing.assert_close(rmsnorm_reference(x, w, eps), expected)


def test_dispatch_uses_reference_on_cpu():
    x = torch.randn(2, 16)
    w = torch.ones(16)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_reference(x, w))
    assert backend(x) == "pytorch-reference"


def test_browsable_kernel_matches_compiled_source():
    """kernels/*.cu is documented as the source that actually gets compiled."""
    on_disk = KERNEL_CU.read_text()
    body = rms._CUDA_SRC.lstrip("\n")
    assert on_disk.endswith(body), (
        "kernels/rmsnorm_kernel.cu has drifted from rmsnorm._CUDA_SRC"
    )


def test_load_inline_wiring_is_consistent():
    """Guard the two mistakes that stop the extension from building at all.

    `load_inline(functions=[...])` generates the PYBIND11_MODULE block into its
    own C++ translation unit, so a second one in the CUDA source leaves two
    PyInit_ symbols to link, and the generated bindings only compile if
    cpp_sources declares every exported function.
    """
    assert "PYBIND11_MODULE" not in rms._CUDA_SRC
    assert "PYBIND11_MODULE" not in rms._CPP_SRC
    for fn in ("rmsnorm_forward",):
        assert f"{fn}(" in rms._CPP_SRC, f"{fn} not declared in cpp_sources"
        assert f"torch::Tensor {fn}(" in rms._CUDA_SRC, f"{fn} not defined in cuda_sources"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_kernel_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(8, 128, device="cuda")
    w = torch.randn(128, device="cuda")
    assert rms.load_error() is None, f"kernel failed to build: {rms.load_error()}"
    assert backend(x) == "cuda-fused"
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_reference(x, w), rtol=1e-5, atol=1e-5)
