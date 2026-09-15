"""Regression checks for explicit motor poses, without pixels or a browser."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_motor_renderer_kinematics_and_time_independence():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to exercise the offline JavaScript renderer")
    renderer = Path(__file__).resolve().parents[1] / "visualizer" / "fly.js"
    checked = subprocess.run(
        [node, "-e", _CHECK_RENDERER, str(renderer)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "Motor renderer invariants passed" in checked.stdout


_CHECK_RENDERER = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const sandbox = {window: {}};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const actor = sandbox.window.FlyActor;
const plain = value => JSON.parse(JSON.stringify(value));
const ids = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
const joints = ['base', 'hip', 'knee', 'ankle', 'foot'];
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
function interior(a, pivot, b) {
  const u = [a[0] - pivot[0], a[1] - pivot[1]];
  const v = [b[0] - pivot[0], b[1] - pivot[1]];
  const cosine = (u[0] * v[0] + u[1] * v[1]) / (Math.hypot(...u) * Math.hypot(...v));
  return Math.acos(Math.max(-1, Math.min(1, cosine)));
}
const neutral = plain(actor.motorKinematics({}));
assert.deepEqual(Object.keys(neutral.legs).sort(), [...ids].sort());

// Inspect coordinates, rather than trusting the renderer's reported angles.
// A positive flexor signal must close the actual drawn joints on both sides.
for (const id of ids) {
  const before = neutral.legs[id];
  const tibia = actor.motorKinematics({legs: {[id]: {tibia: 0.2}}}).legs[id];
  const trochanter = actor.motorKinematics({legs: {[id]: {trochanter: 0.2}}}).legs[id];
  const kneeBefore = interior(before.hip, before.knee, before.ankle);
  const hipBefore = interior(before.base, before.hip, before.knee);
  assert.ok(interior(tibia.hip, tibia.knee, tibia.ankle) < kneeBefore - 0.19,
            `${id}: positive tibia flexion must close the knee`);
  assert.ok(interior(trochanter.base, trochanter.hip, trochanter.knee) < hipBefore - 0.19,
            `${id}: positive trochanter flexion must close the proximal joint`);

  const moved = actor.motorKinematics({legs: {[id]: {coxa: 0.3, trochanter: 0.4, tibia: 0.5}}}).legs[id];
  for (let i = 1; i < joints.length; i++) {
    const a = joints[i - 1], b = joints[i];
    assert.ok(Math.abs(distance(before[a], before[b]) - distance(moved[a], moved[b])) < 1e-8,
              `${id}: rotating joints must preserve ${a}-${b} length`);
  }
  for (const joint of joints) {
    assert.equal(moved[joint].length, 2);
    assert.ok(moved[joint].every(Number.isFinite), `${id}: invalid joint coordinate`);
  }
  const extended = actor.motorKinematics({legs: {[id]: {tibia: -100}}}).legs[id];
  assert.ok(Math.abs(interior(extended.hip, extended.knee, extended.ankle) - Math.PI) < 1e-7,
            `${id}: extensor input must stop at straight, not curl backward`);
}

// One recorded motor channel cannot silently animate other joints or legs.
const isolated = plain(actor.motorKinematics({legs: {LF: {tibia: 0.3}}}));
for (const id of ids.filter(id => id !== 'LF')) assert.deepEqual(isolated.legs[id], neutral.legs[id]);
for (const joint of ['base', 'hip', 'knee']) assert.deepEqual(isolated.legs.LF[joint], neutral.legs.LF[joint]);
assert.deepEqual(plain(actor.motorKinematics({legs: {RF: {tibia: NaN}, LM: {trochanter: Infinity}}})), neutral);

// Record the complete Canvas instruction stream. This covers wings, antennae,
// abdomen, head, shadows and body transforms, not just the returned leg points.
// Playback is NOT marked reduced-motion: that would mask an accidental return
// to the procedural animation and make this regression check too weak.
function recorder() {
  const calls = [], state = {globalAlpha: 1};
  const ctx = new Proxy(state, {
    get(target, key) {
      if (key in target) return target[key];
      if (key === 'createLinearGradient' || key === 'createRadialGradient') {
        return (...args) => {
          calls.push([key, ...args]);
          return {addColorStop: (...args) => calls.push(['addColorStop', ...args])};
        };
      }
      return (...args) => calls.push([key, ...args]);
    },
    set(target, key, value) {
      target[key] = value;
      calls.push(['set', key, typeof value === 'object' ? 'gradient' : value]);
      return true;
    },
  });
  return {ctx, calls};
}
function render(motorPose, time, motion) {
  const recorded = recorder();
  const anchors = actor.draw(recorded.ctx, {
    motorPose, time, motion, reducedMotion: false, dopamine: 0.4, npf: 0.3,
  });
  return plain({calls: recorded.calls, anchors});
}
const poses = [
  {},
  null,
  {legs: {LF: {trochanter: 0.3, tibia: 0.2}, RH: {tibia: 0.1}},
   headAngle: 0.21, bodyRoll: 0.13, abdomenAngle: -0.1, wingAngles: {L: 0.2, R: -0.1}},
];
for (const pose of poses) {
  const first = render(pose, 0, 0);
  assert.deepEqual(render(pose, 17.31, 1), first,
                   'Explicit motor pose must ignore animation time and motion amplitude');
  assert.deepEqual(render(pose, 375.91, 0.73), first,
                   'No unmapped body part may revert to periodic animation');
  assert.equal(first.anchors.motionSource, 'explicit_motor_pose');
  for (const point of [...first.anchors.ports, first.anchors.head]) {
    assert.ok(Number.isFinite(point.x) && Number.isFinite(point.y), 'Invalid head/electrode anchor');
  }
}
console.log('Motor renderer invariants passed');
"""
