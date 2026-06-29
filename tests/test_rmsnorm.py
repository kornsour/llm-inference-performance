import torch

from llminf.rmsnorm import backend, rmsnorm, rmsnorm_reference


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
