#!/usr/bin/env python3
"""Build signed mover value targets for teacher distillation."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _finite_unit(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < -1.0 or result > 1.0:
        raise ValueError(f"{name} must be finite and in [-1,1]")
    return result


def validate_gamma(gamma: float) -> float:
    """Return a valid temporal discount factor."""
    result = float(gamma)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("gamma must be finite and in [0,1]")
    return result


def _positive_temperature(temperature: float) -> float:
    result = float(temperature)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return result


def signed_game_outcome(game_outcome: float, side_to_move: int) -> float:
    """Convert a side-0 outcome to the mover frame."""
    outcome = _finite_unit(game_outcome, "game_outcome")
    if side_to_move not in (0, 1):
        raise ValueError("side_to_move must be 0 or 1")
    return outcome if side_to_move == 0 else -outcome


def discount_signed_value(value: float, remaining_plies: int, gamma: float) -> float:
    """Discount a signed value toward zero by the remaining ply count."""
    signed = _finite_unit(value, "value")
    discount = validate_gamma(gamma)
    plies = int(remaining_plies)
    if plies != remaining_plies or plies < 0:
        raise ValueError("remaining_plies must be a non-negative integer")
    return signed * discount ** plies


def discount_game_outcome(
    game_outcome: float,
    side_to_move: int,
    remaining_plies: int,
    gamma: float,
) -> float:
    """Sign an outcome for the mover and then discount that signed value."""
    signed = signed_game_outcome(game_outcome, side_to_move)
    return discount_signed_value(signed, remaining_plies, gamma)


def temper_policy(policy: Sequence[float], temperature: float) -> np.ndarray:
    """Apply a temperature to probability mass and normalize the result."""
    scale = _positive_temperature(temperature)
    values = np.asarray(policy, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("policy must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("policy must contain finite non-negative values")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("policy must contain positive mass")
    values = values / total
    powered = np.power(values, 1.0 / scale)
    powered_total = float(powered.sum())
    if not math.isfinite(powered_total) or powered_total <= 0.0:
        raise ValueError("temperature produced an invalid policy")
    return (powered / powered_total).astype(np.float32)


def temper_value(value: float, temperature: float) -> float:
    """Apply a log-odds temperature to one signed value."""
    signed = _finite_unit(value, "value")
    scale = _positive_temperature(temperature)
    if signed == 0.0 or scale == 1.0:
        return signed
    if abs(signed) == 1.0:
        return signed
    return math.tanh(math.atanh(signed) / scale)


def _validated_weights(weights: Sequence[float], count: int, name: str) -> list[float]:
    if len(weights) != count:
        raise ValueError(f"{name} count does not match the value count")
    result = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in result):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def blend_value_targets(
    teacher_values: Sequence[float],
    teacher_weights: Sequence[float],
    *,
    outcome_value: float | None = None,
    outcome_weight: float = 0.0,
    bootstrap_value: float | None = None,
    bootstrap_weight: float = 0.0,
) -> float:
    """Blend teacher, outcome, and bootstrap values in the signed mover frame."""
    clean_values = [_finite_unit(value, "teacher value") for value in teacher_values]
    clean_weights = _validated_weights(teacher_weights, len(clean_values), "teacher weight")
    components = list(zip(clean_values, clean_weights))

    extra = (
        (outcome_value, float(outcome_weight), "outcome"),
        (bootstrap_value, float(bootstrap_weight), "bootstrap"),
    )
    for value, weight, name in extra:
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{name} weight must be finite and non-negative")
        if weight > 0.0:
            if value is None:
                raise ValueError(f"{name} value is required when its weight is positive")
            components.append((_finite_unit(value, f"{name} value"), weight))

    total = sum(weight for _, weight in components)
    if total <= 0.0:
        raise ValueError("at least one value weight must be positive")
    blended = sum(value * weight for value, weight in components) / total
    return max(-1.0, min(1.0, blended))
