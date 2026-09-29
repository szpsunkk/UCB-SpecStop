"""Calibration-only content-conditioned draft selection.

This is an empirical controller, without a regret or confidence guarantee.
It estimates conditional committed yield from decision-time draft features,
uses measured mean costs, and solves a shared Dinkelbach fixed point on the
calibration feature distribution. No evaluation outcome is used in fitting.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


FEATURE_NAMES = (
    "feature_entropy", "feature_top1_probability",
    "feature_top2_logit_margin", "feature_context_tokens",
)


class ContentRatioModel:
    def __init__(self, state: dict):
        self.state = dict(state)
        self.arms = np.asarray(state["arms"], dtype=int)
        self.features = tuple(state["features"])
        self.center = np.asarray(state["feature_center"], dtype=float)
        self.scale = np.asarray(state["feature_scale"], dtype=float)
        self.coefficients = np.asarray(state["yield_coefficients"], dtype=float)
        self.mean_costs = np.asarray(state["mean_costs"], dtype=float)
        self.ratio_parameter = float(state["shared_ratio_parameter"])

    @staticmethod
    def feature_matrix(rows: Sequence[Dict[str, float]], features) -> np.ndarray:
        values = np.asarray([[float(row[name]) for name in features] for row in rows])
        if not np.isfinite(values).all():
            raise ValueError("decision features must be finite")
        if "feature_context_tokens" in features:
            column = list(features).index("feature_context_tokens")
            if np.any(values[:, column] <= 0):
                raise ValueError("context length must be positive")
            values[:, column] = np.log(values[:, column])
        return values

    @classmethod
    def fit(cls, costs, yields, feature_rows, arms, ridge=1.0, simple=False):
        costs, yields = np.asarray(costs, float), np.asarray(yields, float)
        arms = np.asarray(arms, dtype=int)
        if costs.ndim != 2 or costs.shape != yields.shape or costs.shape[1] != len(arms):
            raise ValueError("cost/yield matrices must have shape (cells, arms)")
        if len(costs) < 2 or len(feature_rows) != len(costs):
            raise ValueError("at least two matched calibration cells are required")
        if np.any(arms <= 0) or len(np.unique(arms)) != len(arms):
            raise ValueError("arms must be unique positive lengths")
        if not np.isfinite(costs).all() or np.any(costs <= 0):
            raise ValueError("costs must be finite and positive")
        if not np.isfinite(yields).all() or np.any(yields < 1) or np.any(yields > arms + 1):
            raise ValueError("yields must be finite and lie in [1, k+1]")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive")
        features = ("feature_top1_probability",) if simple else FEATURE_NAMES
        raw = cls.feature_matrix(feature_rows, features)
        center = raw.mean(axis=0)
        scale = np.maximum(raw.std(axis=0), 1e-8)
        design = np.column_stack([np.ones(len(raw)), (raw - center) / scale])
        penalty = np.eye(design.shape[1]) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ yields)
        estimated_yields = np.clip(design @ coefficients, 1.0, arms + 1)
        mean_costs = costs.mean(axis=0)
        ratio_parameter = float(np.min(mean_costs / yields.mean(axis=0)))
        for _ in range(100):
            actions = np.argmin(mean_costs[None, :] - ratio_parameter * estimated_yields, axis=1)
            next_parameter = float(mean_costs[actions].sum() / estimated_yields[np.arange(len(raw)), actions].sum())
            if abs(next_parameter - ratio_parameter) <= 1e-10 * max(1.0, ratio_parameter):
                ratio_parameter = next_parameter
                break
            ratio_parameter = next_parameter
        return cls({
            "schema_version": 1,
            "arms": arms.tolist(), "features": list(features),
            "feature_center": center.tolist(), "feature_scale": scale.tolist(),
            "yield_coefficients": coefficients.tolist(),
            "mean_costs": mean_costs.tolist(),
            "shared_ratio_parameter": ratio_parameter,
            "ridge": float(ridge), "calibration_cells": len(costs),
            "objective": "shared aggregate ratio via predicted conditional yield",
            "confidence_claim": "none; calibration-only empirical predictor",
        })

    def predict(self, feature_rows):
        raw = self.feature_matrix(feature_rows, self.features)
        design = np.column_stack([np.ones(len(raw)), (raw - self.center) / self.scale])
        predicted_yields = np.clip(design @ self.coefficients, 1.0, self.arms + 1)
        return np.broadcast_to(self.mean_costs, predicted_yields.shape), predicted_yields

    def choose(self, features: Dict[str, float]) -> int:
        costs, yields = self.predict([features])
        score = costs[0] - self.ratio_parameter * yields[0]
        return int(self.arms[int(np.argmin(score))])
