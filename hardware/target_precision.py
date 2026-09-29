"""Optional FP32 output projection without converting the decoder or KV cache.

This changes only how the language-model output head is evaluated. It is not
a guarantee of identical token decisions across batch shapes or backends.
For full FP32 matrix multiplication, callers must disable TF32 themselves;
this module deliberately does not mutate process-wide backend settings.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FP32OutputHead(nn.Module):
    """Keep the native head intact and project its inputs with FP32 copies."""

    def __init__(self, native_head: nn.Module) -> None:
        super().__init__()
        weight = getattr(native_head, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise TypeError("FP32 output projection requires a two-dimensional head weight")
        bias = getattr(native_head, "bias", None)
        if bias is not None and (not torch.is_tensor(bias) or bias.shape != weight.shape[:1]):
            raise TypeError("output-head bias shape does not match vocabulary size")
        self.native_head = native_head
        self.register_buffer("projection_weight", weight.detach().to(dtype=torch.float32, copy=True))
        self.register_buffer("projection_bias", None if bias is None else bias.detach().to(dtype=torch.float32, copy=True))
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

    @property
    def weight(self):
        return self.projection_weight

    @property
    def bias(self):
        return self.projection_bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.projection_weight.dtype != torch.float32:
            raise RuntimeError("configure the FP32 head after model dtype/device conversion")
        if hidden_states.device != self.projection_weight.device:
            raise RuntimeError("output projection and hidden states must be on the same device")
        # Outer inference autocast must not silently reduce the projection.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return F.linear(hidden_states.float(), self.projection_weight, self.projection_bias)


def configure_output_head(model: nn.Module, mode: str = "native") -> dict:
    """Configure a loaded model's head; return provenance, not a quality claim.

    Call after model placement/dtype conversion and before inference. Native
    mode restores the original module if a prior call installed this wrapper.
    Repeating float32 mode reuses its weight copy. Frozen inference weights
    are required; this helper is not intended for training or fine-tuning.
    """

    if mode not in ("native", "float32"):
        raise ValueError("output-head mode must be native or float32")
    head = model.get_output_embeddings()
    if head is None:
        raise ValueError("model does not expose an output embedding/head")
    if mode == "native":
        if isinstance(head, FP32OutputHead):
            model.set_output_embeddings(head.native_head)
        configured = model.get_output_embeddings()
        native = configured
        extra_bytes = 0
    else:
        if not isinstance(head, FP32OutputHead):
            model.set_output_embeddings(FP32OutputHead(head).eval())
        configured = model.get_output_embeddings()
        native = configured.native_head
        extra_bytes = configured.projection_weight.numel() * configured.projection_weight.element_size()
        if configured.projection_bias is not None:
            extra_bytes += configured.projection_bias.numel() * configured.projection_bias.element_size()
    return {
        "mode": mode,
        "native_weight_dtype": str(native.weight.dtype),
        "projection_weight_dtype": str(configured.weight.dtype),
        "projection_device": str(configured.weight.device),
        "vocabulary_size": int(configured.weight.shape[0]),
        "hidden_size": int(configured.weight.shape[1]),
        "additional_projection_bytes": extra_bytes,
        "decoder_conversion": False,
        "float32_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
