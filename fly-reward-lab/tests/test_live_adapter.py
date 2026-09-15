"""Exercise SSE freshness and validation without making network requests."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_live_adapter_rejects_fake_progress_and_handles_checkpoint_resume():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to exercise the browser live adapter")
    adapter = Path(__file__).resolve().parents[1] / "visualizer" / "live.js"
    checked = subprocess.run(
        [node, "-e", _CHECK_LIVE, str(adapter)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "Live adapter invariants passed" in checked.stdout


_CHECK_LIVE = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const ids = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
function frame(step, stream = 'process-a', run = 'persistent-run') {
  return {run_id: run, stream_id: stream, step, sim_time: step / 100,
          server_time: 1700000000 + step / 100,
          motorPose: {legs: Object.fromEntries(ids.map(id => [id, {trochanter: 0.01, tibia: 0.02}]))},
          channel_rates_hz: Array(24).fill(5), dopamine: 0.3, npf: 0.4,
          dopamine_rate_hz: 50, motor_mean_hz: 7};
}
async function flush() { for (let i = 0; i < 12; i++) await Promise.resolve(); }
function harness(deferState = false) {
  let now = 1000, nextTimer = 1, resolveState;
  const statuses = [], states = [], metas = [], fetches = [], streams = [];
  const intervals = new Map(), timeouts = new Map();
  const delayed = new Promise(resolve => {resolveState = resolve;});
  class EventSource {
    constructor(url, options) {this.url = url; this.options = options; this.readyState = 0; this.handlers = {}; streams.push(this);}
    addEventListener(name, fn) {(this.handlers[name] ||= []).push(fn);}
    emit(name, value) {
      if (name === 'open') this.readyState = 1;
      if (name === 'error') this.readyState = 0;
      if (name === 'closed-error') {this.readyState = 2; name = 'error';}
      const event = name === 'state' || name === 'message' ? {data: JSON.stringify(value)} : {};
      for (const fn of this.handlers[name] || []) fn(event);
    }
    close() {this.readyState = 2; this.wasClosed = true;}
  }
  const context = {
    URL, AbortController, Date: {now: () => now}, console,
    location: {origin: 'https://viewer.example', href: 'https://viewer.example/fly/index.html'},
    EventSource,
    fetch(url, options) {
      fetches.push({url, options});
      if (url.endsWith('/api/meta')) return Promise.resolve({ok: true, json: async () => ({channels: Array(24).fill({}), recorded_neuron_count: 815})});
      if (url.endsWith('/api/state')) return Promise.resolve({ok: true, json: () => deferState ? delayed : Promise.resolve(frame(10))});
      throw new Error('Unexpected network request: ' + url);
    },
    setInterval(fn) {const id = nextTimer++; intervals.set(id, fn); return id;},
    clearInterval(id) {intervals.delete(id);},
    setTimeout(fn) {const id = nextTimer++; timeouts.set(id, fn); return id;},
    clearTimeout(id) {timeouts.delete(id);},
  };
  context.window = context;
  vm.runInNewContext(source, context);
  const live = context.FlyLive.connect('https://backend.example', {
    staleAfterMs: 1000,
    onState: (state, receipt) => states.push({state, receipt}),
    onStatus: value => statuses.push(value),
    onMeta: value => metas.push(value),
  });
  return {live, statuses, states, metas, streams, fetches, intervals, timeouts, resolveState, api: context.FlyLive,
          fireTimeouts() {const pending = [...timeouts.values()]; timeouts.clear(); pending.forEach(fn => fn());},
          tick(ms) {now += ms; for (const fn of [...intervals.values()]) fn();}};
}

(async () => {
  const h = harness();
  await flush();
  const stream = h.streams[0];
  assert.equal(h.states.length, 1);
  assert.equal(h.states[0].receipt.source, 'state');
  assert.equal(h.live.snapshot().phase, 'connecting', 'An initial GET snapshot is not a live stream');
  assert.equal(h.metas.length, 1);
  assert.equal(stream.url, 'https://backend.example/api/stream');
  assert.equal(stream.options.withCredentials, false);
  assert.deepEqual(h.fetches.map(f => new URL(f.url).pathname).sort(), ['/api/meta', '/api/state']);
  assert.ok(h.fetches.every(f => f.options.method === 'GET' && f.options.cache === 'no-store' && f.options.credentials === 'omit'));

  stream.emit('open');
  stream.emit('state', frame(10));
  assert.equal(h.live.snapshot().phase, 'connecting', 'Socket open plus cached state cannot establish progress');
  assert.equal(h.states.length, 1, 'Duplicate snapshots must not generate animation callbacks');
  stream.emit('message', frame(11));
  assert.equal(h.live.snapshot().phase, 'connected');
  assert.equal(h.states.length, 2);
  assert.ok(Object.isFrozen(h.live.getState().motorPose.legs.LF));

  h.tick(600);
  stream.emit('state', frame(11));
  h.tick(500);
  stream.emit('state', frame(11));
  assert.equal(h.live.snapshot().phase, 'stale', 'Repeated cached packets must not reset the progress timer');
  assert.equal(h.states.length, 2);
  stream.emit('state', frame(12));
  assert.equal(h.live.snapshot().phase, 'connected');
  const received = h.live.snapshot().lastReceivedAt;
  stream.emit('state', frame(9));
  assert.equal(h.live.getState().step, 12, 'Out-of-order state must not rewind an active process');
  assert.equal(h.live.snapshot().lastReceivedAt, received);

  // Malformed coordinates/counters/time/channel arrays must never reach the
  // renderer, or make the last good frame appear freshly computed.
  const corruptions = [
    f => {delete f.motorPose.legs.LF;},
    f => {delete f.motorPose.legs.RH.trochanter;},
    f => {f.motorPose.legs.LM.tibia = NaN;},
    f => {f.motorPose.bodyRoll = Infinity;},
    f => {f.motorPose.bodyPositionMm = [0,0,0];},
    f => {f.motorPose.legPointsMm = {};},
    f => {f.motorPose.bodyPositionMm = [0,0,Infinity]; f.motorPose.legPointsMm = {};},
    f => {f.channel_rates_hz.pop();},
    f => {f.channel_rates_hz[5] = Infinity;},
    f => {f.channel_rates_hz[7] = -1;},
    f => {f.step = 1.5;},
    f => {f.step = Infinity;},
    f => {f.sim_time = NaN;},
    f => {f.server_time = Infinity;},
    f => {f.uptime_seconds = Infinity;},
    f => {f.realtime_factor = -1;},
    f => {f.dopamine = NaN;},
    f => {delete f.npf;},
    f => {f.motor_mean_hz = -1;},
    f => {f.run_id = '';},
    f => {f.stream_id = '';},
    f => {f.sim_time = 0.12;},
  ];
  const count = h.states.length;
  for (const corrupt of corruptions) {
    h.tick(1);
    const invalid = frame(13); corrupt(invalid); stream.emit('state', invalid);
    assert.equal(h.live.getState().step, 12);
    assert.equal(h.states.length, count);
    assert.equal(h.live.snapshot().lastReceivedAt, received);
    assert.match(h.live.snapshot().error, /Rejected live state/);
  }

  stream.emit('error');
  assert.equal(h.live.snapshot().phase, 'disconnected');
  assert.equal(h.live.getState().step, 12, 'Keep the actual last state, clearly labeled disconnected');
  stream.emit('open');
  stream.emit('state', frame(13));
  assert.equal(h.live.snapshot().phase, 'connected');

  // The engine run identity survives a checkpoint resume; the process epoch
  // changes. Its counters may legitimately roll back until computation resumes.
  stream.emit('state', frame(4, 'process-b'));
  assert.equal(h.live.getState().step, 4);
  assert.equal(h.live.snapshot().streamId, 'process-b');
  assert.equal(h.states.at(-1).receipt.resumed, true);
  assert.equal(h.states.at(-1).receipt.newRun, false);
  assert.equal(h.live.snapshot().phase, 'connecting');
  stream.emit('state', frame(5, 'process-b'));
  assert.equal(h.live.snapshot().phase, 'connected');
  stream.emit('state', frame(3, 'process-b'));
  assert.equal(h.live.getState().step, 5);
  assert.ok(h.live.snapshot().resumedAt !== null);

  assert.throws(() => h.api.connect('http://remote.example'), /HTTPS/);
  h.live.close();
  assert.equal(h.live.snapshot().phase, 'closed');
  assert.equal(stream.wasClosed, true);
  assert.equal(h.intervals.size, 0);
  const closedCount = h.states.length;
  stream.emit('state', frame(6, 'process-b'));
  assert.equal(h.states.length, closedCount);

  // A slow GET response must not overwrite the stream, even if the GET claims
  // a larger counter from a different process or a different engine run.
  const delayed = harness(true);
  await flush();
  delayed.streams[0].emit('open');
  assert.equal(delayed.live.getState(), null);
  delayed.tick(1100);
  assert.equal(delayed.live.snapshot().phase, 'stale');
  assert.equal(delayed.states.length, 0, 'Do not invent frames while waiting for the backend');
  delayed.streams[0].emit('state', frame(20));
  assert.equal(delayed.live.snapshot().phase, 'connecting');
  delayed.streams[0].emit('state', frame(21));
  assert.equal(delayed.live.snapshot().phase, 'connected');
  delayed.resolveState(frame(100, 'older-process', 'older-run'));
  await flush();
  assert.equal(delayed.live.getState().step, 21);
  assert.equal(delayed.live.getState().run_id, 'persistent-run');
  delayed.live.close();

  // A warming backend may return HTTP503, leaving EventSource permanently
  // closed instead of natively reconnecting. Maintain one cancellable retry.
  const warming = harness();
  await flush();
  const old = warming.streams[0];
  old.emit('closed-error');
  old.emit('closed-error');
  assert.equal(warming.live.snapshot().phase, 'disconnected');
  assert.equal(warming.timeouts.size, 1, 'Repeated errors must not accumulate retry timers');
  warming.fireTimeouts();
  assert.equal(warming.streams.length, 2);
  const replacement = warming.streams[1];
  replacement.emit('open');
  replacement.emit('state', frame(11));
  assert.equal(warming.live.snapshot().phase, 'connected');
  old.emit('state', frame(1000, 'obsolete-process'));
  assert.equal(warming.live.getState().step, 11, 'An obsolete EventSource cannot replace current state');
  replacement.emit('closed-error');
  assert.equal(warming.timeouts.size, 1);
  warming.live.close();
  assert.equal(warming.timeouts.size, 0, 'Closing must cancel the pending manual retry');
  warming.fireTimeouts();
  assert.equal(warming.streams.length, 2, 'Closed clients must not reconnect');
  console.log('Live adapter invariants passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
