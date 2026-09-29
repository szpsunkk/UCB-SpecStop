"""Compatibility helpers for Hugging Face dynamic and legacy KV caches."""

from __future__ import annotations

from typing import Any

import torch


def cache_sequence_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())
    try:
        return int(past_key_values[0][0].shape[-2])
    except (IndexError, TypeError, AttributeError) as exc:
        raise TypeError("unsupported past_key_values format") from exc


def crop_past_key_values(past_key_values: Any, max_length: int) -> Any:
    if max_length < 0:
        raise ValueError("max_length must be nonnegative")
    crop = getattr(past_key_values, "crop", None)
    if callable(crop):
        crop(max_length)
        return past_key_values

    cropped_layers = []
    for layer in past_key_values:
        cropped = []
        for value in layer:
            if torch.is_tensor(value) and value.ndim >= 3:
                value = value[..., :max_length, :].contiguous()
            cropped.append(value)
        cropped_layers.append(tuple(cropped))
    return tuple(cropped_layers)
