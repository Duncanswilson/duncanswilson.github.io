import numpy as np
import pytest
from flyreward.experiments import candidates, objective, rank_results
from flyreward.types import SimulationConfig, SimulationResult


def result(name, da, npf=0, saturation=0):
    return SimulationResult({"preset": name}, np.arange(3), {
        "dopamine_concentration": np.full(3, da), "dopamine_reward_proxy": np.full(3, da),
        "npf_exposure_proxy": np.full(3, npf), "saturation_fraction": np.full(3, saturation)},
        {"dopamine_concentration_mean": da, "dopamine_reward_proxy_mean": da,
         "npf_exposure_proxy_mean": npf, "mean_saturation_fraction": saturation})


def test_score_penalizes_saturation_and_ranking_is_explicit():
    good = result("good", .5, .1, 0)
    saturated = result("saturated", 1, 0, 1)
    ranking = rank_results([saturated, good], "reward_proxy")
    assert ranking["winner"]["preset"] == "good"
    assert "Not a global maximum" in ranking["scope"]
    assert objective(good, "reward_proxy") == pytest.approx(.6)


def test_grid_is_reproducible_unique_and_preserves_execution_settings():
    base = SimulationConfig(duration=1.25, dt=.005, seed=12)
    a = candidates(base, 54)
    assert a == candidates(base, 54)
    assert len({(x.dopamine_drive, x.reuptake_factor, x.npf_drive, x.tolerance) for x in a}) == 54
    assert all(x.duration == 1.25 and x.seed == 12 for x in a)
    with pytest.raises(ValueError):
        candidates(base, 55)


def test_nonfinite_objective_fails_closed():
    with pytest.raises(ValueError):
        objective(result("bad", float("nan")), "dopamine")
