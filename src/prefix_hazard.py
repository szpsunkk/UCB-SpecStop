"""Measured-cost, within-round draft continuation from prefix hazards.

This is an empirical myopic policy. Before draft token j, it predicts the
probability of acceptance conditional on earlier acceptance, multiplies the
predicted hazards, and compares a measured mean cost increment with lambda*q_j.
The shared lambda minimizes training ratio-of-sums within this first-crossing
policy family; it is not claimed to solve an unrestricted Dinkelbach oracle.
Only reached, pre-token features are consulted during a live round.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from src.content_ratio import ContentRatioModel, FEATURE_NAMES


FEATURE_MODES = {
    "full": FEATURE_NAMES + ("token_position",),
    "top1": ("feature_top1_probability", "token_position"),
    "entropy": ("feature_entropy", "token_position"),
}


def _point(features, position):
    values = dict(features)
    if "token_position" in values and values["token_position"] != position:
        raise ValueError("feature token_position differs from callback position")
    values["token_position"] = int(position)
    return values


def _sigmoid(values):
    values = np.clip(np.asarray(values, float), -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def _logistic_fit(design, labels, ridge):
    """Damped Newton logistic ridge; all-one/all-zero data use Beta(1,1)."""
    labels = np.asarray(labels, float)
    probability = (labels.sum() + 1) / (len(labels) + 2)
    coefficients = np.zeros(design.shape[1])
    coefficients[0] = np.log(probability / (1-probability))
    if np.all(labels == labels[0]):
        return coefficients, 0
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0

    def objective(beta):
        logits = design @ beta
        return float(np.logaddexp(0, logits).sum() - labels @ logits + .5 * beta @ penalty @ beta)

    for iteration in range(100):
        prediction = _sigmoid(design @ coefficients)
        gradient = design.T @ (prediction-labels) + penalty @ coefficients
        hessian = design.T @ ((prediction*(1-prediction))[:, None] * design) + penalty
        step = np.linalg.solve(hessian + 1e-10*np.eye(len(coefficients)), gradient)
        rate, current = 1.0, objective(coefficients)
        while rate > 1e-8 and objective(coefficients-rate*step) > current:
            rate *= .5
        coefficients -= rate*step
        if np.max(np.abs(rate*step)) < 1e-8:
            return coefficients, iteration+1
    raise RuntimeError("hazard logistic fit did not converge")


class PrefixHazardModel:
    def __init__(self, state):
        self.state = dict(state)
        self.k_max = int(state["k_max"])
        self.feature_names = tuple(state["feature_names"])
        self.center = np.asarray(state["feature_center"], float)
        self.scale = np.asarray(state["feature_scale"], float)
        self.coefficients = np.asarray(state["hazard_coefficients"], float)
        self.mean_costs = np.asarray(state["mean_gate_costs"], float)
        self.increments = np.r_[np.nan, np.diff(self.mean_costs)]
        self.ratio_parameter = float(state["shared_lambda"])

    @classmethod
    def fit(cls, gate_costs, gate_yields, feature_trajectories, arms,
            ridge=1.0, feature_mode="full", prefix_compatible=False):
        if prefix_compatible is not True:
            raise ValueError("hazard feedback requires an explicit compatible-prefix protocol modeling assumption")
        costs, yields = np.asarray(gate_costs, float), np.asarray(gate_yields, float)
        arms = list(arms)
        if not arms or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) for k in arms) or arms != list(range(1, len(arms)+1)):
            raise ValueError("arms must be the complete ordered grid 1..K")
        k_max = len(arms)
        if costs.ndim != 2 or costs.shape != yields.shape or costs.shape[1] != k_max or len(costs) < 2:
            raise ValueError("costs and yields must be matched cells-by-K arrays with at least two cells")
        if len(feature_trajectories) != len(costs):
            raise ValueError("one full feature trajectory is required per calibration cell")
        if not np.isfinite(costs).all() or np.any(costs <= 0):
            raise ValueError("measured gate costs must be finite and positive")
        if not np.isfinite(yields).all() or np.any(yields != np.floor(yields)) or np.any(yields < 1) or np.any(yields > np.asarray(arms)+1):
            raise ValueError("use complete nonterminal integer yields in [1,k+1]")
        expected = np.minimum(yields[:, -1:], np.asarray(arms)[None, :]+1)
        if not np.array_equal(yields, expected):
            raise ValueError("gate yields violate shared-prefix nesting; no rows are repaired or dropped")
        if feature_mode not in FEATURE_MODES or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("invalid feature_mode or ridge")
        points, labels, accepted = [], [], yields[:, -1].astype(int)-1
        for trajectory, prefix in zip(feature_trajectories, accepted):
            if len(trajectory) < k_max:
                raise ValueError("full maximum-arm feature trajectories are required; do not fill censored positions")
            for position in range(1, k_max+1):
                if position > prefix+1:
                    break  # After the first rejection, acceptance is not observed.
                points.append(_point(trajectory[position-1], position))
                labels.append(int(position <= prefix))
        names = FEATURE_MODES[feature_mode]
        raw = ContentRatioModel.feature_matrix(points, names)
        center, scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-8)
        design = np.column_stack([np.ones(len(raw)), (raw-center)/scale])
        coefficients, iterations = _logistic_fit(design, labels, float(ridge))
        state = {
            "schema_version": 1, "k_max": k_max, "feature_mode": feature_mode,
            "feature_names": list(names), "feature_center": center.tolist(), "feature_scale": scale.tolist(),
            "hazard_coefficients": coefficients.tolist(), "mean_gate_costs": costs.mean(axis=0).tolist(),
            "shared_lambda": 0.0, "ridge": float(ridge), "hazard_fit_iterations": iterations,
            "calibration_cells": len(costs), "hazard_training_positions": len(labels),
            "accepted_training_positions": int(sum(labels)), "first_rejection_training_positions": len(labels)-int(sum(labels)),
            "right_censored_positions": len(costs)*k_max-len(labels),
            "minimum_draft_length": 1,
            "cost_scope": "actual forced-stop gated cycles; include reached rejecting-position features and eager materialization; never substitute lazy costs",
            "feedback_scope": "maximum-arm accepted prefix; positions beyond first rejection are censored",
            "compatibility_scope": "caller-declared modeling assumption and complete calibration nesting check; not protocol-wide certification",
            "objective": "training aggregate ratio minimized over one shared-lambda first-crossing policy family",
            "guarantee": "none; myopic continuation, not unrestricted or finite-request optimality",
            "tail_scope": "request cap only limits available draft lengths; cap-specific costs are not reconstructed",
        }
        model = cls(state)
        thresholds = []
        cell_thresholds = []
        for trajectory in feature_trajectories:
            survival = 1.0
            values = []
            for position in range(1, k_max+1):
                survival *= model.predict_hazard(position, trajectory[position-1])
                if position > 1:
                    threshold = model.increments[position-1]/survival
                    values.append(float(threshold))
                    if np.isfinite(threshold) and threshold >= 0:
                        thresholds.append(float(threshold))
            cell_thresholds.append(values)
        cell_thresholds = np.asarray(cell_thresholds, float)
        # The selected depth is piecewise constant in lambda. Evaluate its
        # complete nonnegative family using just-right-of-breakpoint values.
        candidates = [0.0] + [float(np.nextafter(value, np.inf)) for value in sorted(set(thresholds))]
        candidate_costs = []
        candidate_depths = []
        for parameter in candidates:
            # Features and predicted hazards are independent of lambda. This
            # prefix product exactly reproduces stopping at the first failure,
            # without repeatedly running prediction during calibration fitting.
            continuation = parameter > cell_thresholds
            columns = np.cumprod(continuation, axis=1).sum(axis=1).astype(int)
            depths = (columns+1).tolist()
            candidate_costs.append(float(costs[np.arange(len(costs)), columns].sum()
                                         / yields[np.arange(len(costs)), columns].sum()))
            candidate_depths.append(depths)
        winner = int(np.argmin(candidate_costs))
        state.update(shared_lambda=candidates[winner], lambda_candidate_count=len(candidates),
                     calibration_selected_ratio=candidate_costs[winner],
                     calibration_selected_depths=candidate_depths[winner],
                     lambda_candidates=candidates, lambda_candidate_ratios=candidate_costs)
        return cls(state)

    def predict_hazard(self, position, features):
        raw = ContentRatioModel.feature_matrix([_point(features, position)], self.feature_names)[0]
        design = np.r_[1.0, (raw-self.center)/self.scale]
        return float(np.clip(_sigmoid(design @ self.coefficients), 1e-6, 1-1e-6))

    def new_round(self, max_draft_tokens=None):
        maximum = self.k_max if max_draft_tokens is None else max_draft_tokens
        if isinstance(maximum, (bool, np.bool_)) or not isinstance(maximum, Integral) or not 0 <= maximum <= self.k_max:
            raise ValueError("max_draft_tokens must lie in 0..K")
        return PrefixHazardGate(self, int(maximum))

    def depth_for_trajectory(self, trajectory, max_draft_tokens=None):
        gate = self.new_round(max_draft_tokens)
        depth = 0
        for position in range(1, gate.max_draft_tokens+1):
            if not gate(position, trajectory[position-1]):
                break
            depth = position
        return depth


class PrefixHazardGate:
    def __init__(self, model, max_draft_tokens):
        self.model, self.max_draft_tokens = model, max_draft_tokens
        self.survival = 1.0
        self.decisions = []
        self.stopped = False

    def __call__(self, position, features):
        if self.stopped or position != len(self.decisions)+1 or not 1 <= position <= self.max_draft_tokens:
            raise ValueError("gate callbacks must visit each reachable position once, before its token")
        hazard = self.model.predict_hazard(position, features)
        self.survival *= hazard
        incremental_cost = None if position == 1 else float(self.model.increments[position-1])
        threshold = None if position == 1 else incremental_cost/self.survival
        should_continue = position == 1 or self.model.ratio_parameter > threshold
        self.decisions.append({"token_position": int(position), "predicted_hazard": hazard,
                               "predicted_prefix_survival": self.survival,
                               "incremental_gate_cost_ms": incremental_cost,
                               "continuation_threshold_ms_per_token": threshold,
                               "shared_lambda": self.model.ratio_parameter,
                               "continue_draft": bool(should_continue)})
        self.stopped = not should_continue
        return bool(should_continue)
