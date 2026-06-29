"""int8 dynamic quantization of the decoder's linear layers.

Dynamic quantization stores weights as int8 and quantizes activations on the fly
at matmul time — a CPU-friendly win that shrinks the model and can speed up the
linear-heavy decode path. We measure model size before/after and the numerical
drift (logit MSE) so the quality cost is explicit, not hand-waved.
"""

from __future__ import annotations

import copy
import io

import torch
import torch.nn as nn

from .model import GPT


def _select_engine() -> str | None:
    supported = list(torch.backends.quantized.supported_engines)
    for eng in ("fbgemm", "x86", "qnnpack"):  # prefer server engines, fall back to ARM
        if eng in supported:
            torch.backends.quantized.engine = eng
            return eng
    return None


def quantize_int8(model: GPT) -> nn.Module:
    """Return an int8 dynamically-quantized copy (Linear layers only)."""
    if _select_engine() is None:
        raise RuntimeError("no quantized engine available on this platform")
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    return torch.ao.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8
    )


def model_size_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


@torch.no_grad()
def logit_mse(a: nn.Module, b: nn.Module, idx: torch.Tensor) -> float:
    la, _ = a(idx)
    lb, _ = b(idx)
    return float(torch.mean((la - lb) ** 2).item())
