import pytest
import torch

from llminf.model import GPT, GPTConfig
from llminf.quantize import logit_mse, model_size_bytes, quantize_int8

_HAS_ENGINE = len(torch.backends.quantized.supported_engines) > 0 and \
    torch.backends.quantized.supported_engines != ["none"]


@pytest.mark.skipif(not _HAS_ENGINE, reason="no quantized engine on this platform")
def test_int8_is_smaller_and_close():
    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layer=3, n_head=3, n_embd=96, block_size=128)).eval()
    qmodel = quantize_int8(model)
    assert model_size_bytes(qmodel) < model_size_bytes(model)

    idx = torch.randint(0, model.cfg.vocab_size, (1, 16))
    mse = logit_mse(model, qmodel, idx)
    assert mse < 0.05  # int8 drift stays small
