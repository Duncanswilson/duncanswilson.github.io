"""Independent checks of forces, contact, convergence and checkpoint continuity."""
from copy import deepcopy
import json

import numpy as np
import pytest

from flyreward.body import BodyConfig, ReducedBody


def evolve(body, inputs, duration):
    for _ in range(round(duration / .01)):
        body.advance(inputs, .01)
    return body.diagnostics()


def test_equal_antagonists_without_gravity_do_not_generate_movement():
    body = ReducedBody(BodyConfig(gravity=0))
    initial = body.q.copy()
    evolve(body, np.ones(24), .3)
    np.testing.assert_allclose(body.q, initial, atol=1e-12)
    np.testing.assert_allclose(body.velocity, 0, atol=1e-12)


def test_contact_supports_weight_and_constant_inputs_settle():
    body = ReducedBody()
    state = evolve(body, np.full(24, .2), 1.)
    foot_force = np.array(state["contact_forces_mg_mm_per_second_squared"])[:, 2].sum()
    body_force = state["body_contact_force_mg_mm_per_second_squared"][2]
    mass = body.config.body_mass + 6 * (body.config.femur_mass + body.config.distal_mass)
    assert foot_force + body_force == pytest.approx(mass * body.config.gravity, abs=1e-5)
    assert body.q[2] > .15
    assert np.max(np.abs(body.velocity)) < 1e-6
    assert any(state["proprioception"]["contact"])


def test_torque_changes_angles_velocity_load_and_body_position():
    body, control = ReducedBody(), ReducedBody()
    baseline = np.full(24, .2)
    evolve(body, baseline, .6)
    evolve(control, baseline, .6)
    before = body.q.copy()
    drive = baseline.copy()
    drive[0], drive[1] = 1., 0.
    speeds = []
    for _ in range(8):
        body.advance(drive, .01)
        speeds.append(np.max(np.abs(body.velocity[3:])))
    evolve(control, baseline, .08)
    assert np.max(np.abs(body.q[3:] - control.q[3:])) > .002
    assert max(speeds) > .01
    assert np.max(np.abs(body.last_contact - control.last_contact)) > 1.
    assert np.linalg.norm(body.q[:2] - before[:2]) > 1e-5


def test_half_substep_agrees_in_contact():
    first = ReducedBody()
    refined = ReducedBody(BodyConfig(max_substep=.000125))
    drive = np.full(24, .2)
    drive[0] = .8
    evolve(first, drive, .3)
    evolve(refined, drive, .3)
    np.testing.assert_allclose(first.q, refined.q, atol=.002, rtol=0)


def test_contact_force_is_unilateral_and_within_friction_cone():
    body = ReducedBody()
    force, _ = body._contact(np.array([0., 0., -.01]), np.array([100., -20., -1.]))
    assert force[2] >= 0
    assert np.linalg.norm(force[:2]) <= body.config.friction * force[2] + 1e-12
    assert np.dot(force[:2], [100., -20.]) <= 0
    # Separating rapidly cannot create an adhesive normal force.
    force, _ = body._contact(np.array([0., 0., -.01]), np.array([0., 0., 100.]))
    assert force[2] == 0
    force, _ = body._contact(np.array([0., 0., .01]), np.array([0., 0., -100.]))
    np.testing.assert_array_equal(force, 0)


def test_mass_matrix_is_positive_definite_and_matches_kinetic_energy():
    body = ReducedBody()
    body.q[3:] = np.tile([.1, -.1], 6)
    body.velocity[:] = np.linspace(-.1, .2, 15)
    matrix, _, _ = body._forces()
    assert np.linalg.eigvalsh(matrix).min() > 0
    expected = .5 * body.config.body_mass * np.dot(body.velocity[:3], body.velocity[:3])
    expected += .5 * body.config.additional_joint_inertia * np.dot(body.velocity[3:], body.velocity[3:])
    for leg in range(6):
        for mass, first, second in [(body.config.femur_mass, body.config.femur_length / 2, 0.),
                                    (body.config.distal_mass, body.config.femur_length, body.config.distal_length / 2)]:
            _, jacobian, _ = body._point(leg, first, second)
            speed = jacobian @ body.velocity
            expected += .5 * mass * np.dot(speed, speed)
    assert .5 * body.velocity @ matrix @ body.velocity == pytest.approx(expected, rel=1e-12)


def test_checkpoint_resumes_identical_dynamics_and_does_not_alias_state():
    first, restored = ReducedBody(), ReducedBody()
    drive = np.full(24, .2)
    drive[0] = .9
    evolve(first, drive, .15)
    checkpoint = first.state_dict()
    restored.restore(json.loads(json.dumps(checkpoint, allow_nan=False)))
    checkpoint["q"][0] += 10
    np.testing.assert_array_equal(first.q, restored.q)
    evolve(first, drive, .08)
    evolve(restored, drive, .08)
    assert first.state_dict() == restored.state_dict()
    assert first.pose() == restored.pose()


@pytest.mark.parametrize("mutation", [
    lambda state: state.update(schema_version=999),
    lambda state: state.update(extra="unknown"),
    lambda state: state.update(q=[0.] * 14),
    lambda state: state["q"].__setitem__(2, float("nan")),
    lambda state: state["q"].__setitem__(3, 4.),
    lambda state: state["velocity"].__setitem__(1, float("inf")),
    lambda state: state["activation"][0][0].__setitem__(0, 2.),
    lambda state: state["parameters"].update(friction=.9),
    lambda state: state.update(t=-1),
])
def test_invalid_checkpoint_is_rejected_atomically(mutation):
    body = ReducedBody()
    body.advance(np.full(24, .2), .01)
    before = body.state_dict()
    damaged = deepcopy(before)
    mutation(damaged)
    with pytest.raises(ValueError):
        body.restore(damaged)
    assert body.state_dict() == before


@pytest.mark.parametrize("inputs,dt", [
    (np.zeros(23), .01), (np.full(24, np.nan), .01),
    (np.full(24, 1.1), .01), (np.zeros(24), 0),
])
def test_invalid_motor_input_is_rejected_before_mutation(inputs, dt):
    body = ReducedBody()
    before = body.state_dict()
    with pytest.raises(ValueError):
        body.advance(inputs, dt)
    assert body.state_dict() == before


def test_pose_exposes_physical_geometry_without_unsupported_motor_assignments():
    body = ReducedBody()
    body.advance(np.full(24, .2), .01)
    pose = body.pose()
    assert set(pose["legs"]) == {"LF", "LM", "LH", "RF", "RM", "RH"}
    assert pose["bodyRoll"] == pose["headAngle"] == pose["abdomenAngle"] == 0
    assert pose["wingAngles"] == {"L": 0., "R": 0.}
    for points in pose["legPointsMm"].values():
        hip, knee, foot = [np.array(points[key]) for key in ("hip", "knee", "foot")]
        assert np.linalg.norm(knee - hip) == pytest.approx(body.config.femur_length)
        assert np.linalg.norm(foot - knee) == pytest.approx(body.config.distal_length)
    json.dumps({"pose": pose, "diagnostics": body.diagnostics(), "meta": body.metadata()}, allow_nan=False)
