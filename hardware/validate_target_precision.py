"""CPU-only checks for isolated output-head precision changes."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hardware.target_precision import FP32OutputHead, configure_output_head


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(3, 3).half()
        self.lm_head = nn.Linear(3, 5).half()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, head):
        self.lm_head = head


def test_fp32_head_preserves_decoder_and_original_weights():
    model = TinyModel()
    original_head = model.lm_head
    body_weight = model.decoder.weight.detach().clone()
    original_weight = original_head.weight.detach().clone()
    meta = configure_output_head(model, "float32")
    assert isinstance(model.lm_head, FP32OutputHead)
    assert model.lm_head.native_head is original_head
    assert torch.equal(model.decoder.weight, body_weight)
    assert model.decoder.weight.dtype == torch.float16
    assert torch.equal(original_head.weight, original_weight)
    assert original_head.weight.dtype == torch.float16
    assert meta["additional_projection_bytes"] == (3 * 5 + 5) * 4
    assert meta["projection_weight_dtype"] == "torch.float32"
    assert meta["decoder_conversion"] is False


def test_fp32_projection_matches_full_precision_linear_even_under_autocast():
    model = TinyModel()
    configure_output_head(model, "float32")
    hidden = torch.tensor([[[0.5, 0.25, -1.0], [1.0, -0.5, 0.125]]], dtype=torch.float16)
    expected = F.linear(hidden.float(), model.lm_head.weight, model.lm_head.bias)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = model.lm_head(hidden)
    assert actual.dtype == torch.float32
    assert actual.shape == (1, 2, 5)
    assert torch.equal(actual, expected)


def test_repeat_configuration_reuses_copy_and_native_restores_same_module():
    model = TinyModel()
    original = model.lm_head
    configure_output_head(model, "float32")
    copy_pointer = model.lm_head.weight.data_ptr()
    configure_output_head(model, "float32")
    assert model.lm_head.weight.data_ptr() == copy_pointer
    meta = configure_output_head(model, "native")
    assert model.lm_head is original and meta["additional_projection_bytes"] == 0
    assert meta["projection_weight_dtype"] == "torch.float16"


def test_later_half_conversion_is_detected_instead_of_silently_reducing_precision():
    model = TinyModel()
    configure_output_head(model, "float32")
    model.half()
    with pytest.raises(RuntimeError, match="after model dtype"):
        model.lm_head(torch.ones((1, 3), dtype=torch.float16))


def test_invalid_mode_leaves_model_unchanged():
    model = TinyModel()
    original = model.lm_head
    with pytest.raises(ValueError, match="mode"):
        configure_output_head(model, "float64")
    assert model.lm_head is original
