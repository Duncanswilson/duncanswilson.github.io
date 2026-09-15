"""Transparent parameter comparisons; objectives are model proxies, never pleasure."""
from dataclasses import replace
from itertools import product
import numpy as np
from .types import SimulationConfig, SimulationResult


OBJECTIVES = {
    "dopamine": "Mean modeled dopamine concentration (arbitrary units)",
    "reward_proxy": "Mean dopamine_reward_proxy + mean npf_exposure_proxy - 2 * mean saturation_fraction; arbitrary engineering score",
}


def objective(result: SimulationResult, name: str) -> float:
    if name not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {name}")
    def metric(key):
        value = float(result.metrics[key])
        if not np.isfinite(value):
            raise ValueError(f"Invalid metric: {key}")
        return value
    if name == "dopamine":
        return metric("dopamine_concentration_mean")
    return metric("dopamine_reward_proxy_mean") + metric("npf_exposure_proxy_mean") - 2 * metric("mean_saturation_fraction")


def candidates(base: SimulationConfig, trials: int = 12) -> list[SimulationConfig]:
    """Deterministic coverage of a finite grid, including its extreme settings."""
    grid = list(product((0.25, 0.6, 1.0), (0.1, 0.3, 1.0), (0.0, 0.5, 1.0), (True, False)))
    if not 1 <= trials <= len(grid):
        raise ValueError(f"trials must be between 1 and {len(grid)}")
    indices = np.linspace(0, len(grid) - 1, trials, dtype=int)
    return [replace(base, preset=f"candidate_{i+1:02d}", dopamine_drive=d, reuptake_factor=r,
                    npf_drive=n, tolerance=t) for i, (d, r, n, t) in enumerate(grid[j] for j in indices)]


def rank_results(results: list[SimulationResult], target: str) -> dict:
    if not results:
        raise ValueError("At least one result is required")
    ranking = sorted([{"preset": r.config["preset"], "score": objective(r, target), "config": r.config}
                      for r in results], key=lambda x: x["score"], reverse=True)
    return {"objective": target, "definition": OBJECTIVES[target], "trials": len(results),
            "winner": ranking[0], "ranking": ranking,
            "scope": "Best among the tested finite grid at this duration and seed. Not a global maximum or evidence of subjective pleasure."}
