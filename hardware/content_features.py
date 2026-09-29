"""Pre-selection draft-logit features shared by calibration and runtime."""

from __future__ import annotations

from numbers import Integral

import torch


def synchronize_device(device) -> None:
    """Finish queued device work so the caller's measured interval includes it."""

    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def extract_content_features(next_logits: torch.Tensor, context_tokens: int) -> dict:
    """Return four observable features; caller must time this entire function.

    Entropy uses natural logarithms. Probability and entropy calculations use
    float32; the margin is the largest minus second-largest raw logit. CUDA
    synchronization and scalar transfer are included in the function call.
    No selected-arm result, target outcome, or future token is consulted.
    """

    if isinstance(context_tokens, bool) or not isinstance(context_tokens, Integral) or context_tokens < 1:
        raise ValueError("context_tokens must be a positive integer")
    if not isinstance(next_logits, torch.Tensor) or next_logits.ndim != 1 or next_logits.numel() < 2:
        raise ValueError("next_logits must be a one-dimensional vocabulary tensor")
    synchronize_device(next_logits.device)
    with torch.inference_mode():
        logits = next_logits.detach().float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("next_logits must contain finite raw model logits")
        log_probabilities = torch.log_softmax(logits, dim=0)
        probabilities = log_probabilities.exp()
        top_values = torch.topk(logits, 2).values
        values = torch.stack((
            -(probabilities * log_probabilities).sum(),
            probabilities.max(),
            top_values[0] - top_values[1],
        )).cpu().tolist()
    synchronize_device(next_logits.device)
    return {
        "feature_entropy": float(values[0]),
        "feature_top1_probability": float(values[1]),
        "feature_top2_logit_margin": float(values[2]),
        "feature_context_tokens": int(context_tokens),
    }


def extract_draft_features(draft_session) -> dict:
    """Read the next-token logits before drafting or selecting an arm."""

    if getattr(draft_session, "_pending", None) is not None:
        raise ValueError("extract features before drafting, with no pending draft")
    return extract_content_features(draft_session.next_logits, draft_session.kv_tokens)
