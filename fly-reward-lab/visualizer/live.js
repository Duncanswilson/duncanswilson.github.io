/* Read-only consumer for the continuous simulation API.
 *
 * const live = FlyLive.connect("https://backend.example", {
 *   onState(state, receipt) {}, onStatus(status) {}, onMeta(meta) {},
 *   staleAfterMs: 15000
 * });
 * live.getState(); live.snapshot(); live.close();
 *
 * GET /api/meta, GET /api/state, EventSource /api/stream (message/state events).
 * State: {run_id, stream_id?, step, sim_time, server_time, motorPose:{legs:{...}},
 *         channel_rates_hz:[24], motor_rates_hz?:[...], dopamine, npf,
 *         dopamine_rate_hz, motor_mean_hz, ...}. server_time is Unix seconds.
 * A changed stream_id identifies a backend-process restart and permits loading
 * an earlier checkpoint without changing the persistent engine run_id.
 *
 * Only the latest real server state is retained. No replay, interpolation,
 * generated frames, reset requests, or stimulation controls exist here.
 * "connected" requires an observed advancing stream state. A cached initial
 * GET or an open socket alone cannot establish a running simulation.
 */
(function (global) {
  "use strict";

  const record = value => value !== null && typeof value === "object" && !Array.isArray(value);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const legIds = ["LF", "LM", "LH", "RF", "RM", "RH"];
  const freeze = value => {
    if (value && typeof value === "object" && !Object.isFrozen(value)) {
      Object.values(value).forEach(freeze);
      Object.freeze(value);
    }
    return value;
  };

  function validateState(value) {
    if (!record(value)) throw new Error("State must be an object");
    if (typeof value.run_id !== "string" || !value.run_id) throw new Error("State is missing run_id");
    if ("stream_id" in value && (typeof value.stream_id !== "string" || !value.stream_id)) {
      throw new Error("State stream_id must be a nonempty string when provided");
    }
    if (!Number.isSafeInteger(value.step) || value.step < 0) throw new Error("State step must be a nonnegative integer");
    if (!finite(value.sim_time) || value.sim_time < 0) throw new Error("State sim_time must be finite and nonnegative");
    if (!finite(value.server_time) || value.server_time < 0) throw new Error("State server_time must be finite Unix seconds");
    if (!record(value.motorPose) || !record(value.motorPose.legs)) throw new Error("State is missing motorPose.legs");
    for (const id of legIds) {
      const leg = value.motorPose.legs[id];
      if (!record(leg)) throw new Error("Invalid motor leg: " + id);
      for (const joint of ["trochanter", "tibia"]) {
        if (!finite(leg[joint])) throw new Error("Missing or nonfinite motor joint: " + id + "." + joint);
      }
      if ("coxa" in leg && !finite(leg.coxa)) throw new Error("Nonfinite motor joint: " + id + ".coxa");
    }
    for (const key of ["bodyRoll", "headAngle", "abdomenAngle"]) {
      if (key in value.motorPose && !finite(value.motorPose[key])) throw new Error("Nonfinite motor pose: " + key);
    }
    if ("wingAngles" in value.motorPose) {
      if (!record(value.motorPose.wingAngles)) throw new Error("Invalid wingAngles");
      for (const angle of Object.values(value.motorPose.wingAngles)) {
        if (!finite(angle)) throw new Error("Nonfinite wing angle");
      }
    }
    for (const key of ["channel_rates_hz", "motor_rates_hz"]) {
      if (key === "motor_rates_hz" && !(key in value)) continue;
      if (!Array.isArray(value[key]) || (key === "channel_rates_hz" && value[key].length !== 24) ||
          value[key].length > 100000 ||
          !value[key].every(rate => finite(rate) && rate >= 0)) {
        throw new Error("Invalid " + key);
      }
    }
    for (const key of ["dopamine", "npf", "dopamine_rate_hz", "motor_mean_hz"]) {
      if (!finite(value[key]) || value[key] < 0) throw new Error("Missing, negative or nonfinite " + key);
    }
    for (const key of ["realtime_factor", "uptime_seconds"]) {
      if (key in value && (!finite(value[key]) || value[key] < 0)) throw new Error("Negative or nonfinite " + key);
    }
    return freeze(value);
  }

  function connect(baseUrl, options) {
    options = options || {};
    if (typeof global.EventSource !== "function" || typeof global.fetch !== "function") {
      throw new Error("This browser needs EventSource and fetch to receive live simulation states");
    }
    const base = new URL(baseUrl || global.location.origin, global.location.href);
    if (!["http:", "https:"].includes(base.protocol) || base.username || base.password) {
      throw new Error("The public backend URL must use HTTP(S) without embedded credentials");
    }
    if (base.protocol === "http:" && !["localhost", "127.0.0.1", "[::1]"].includes(base.hostname)) {
      throw new Error("Remote live backends must use HTTPS; HTTP is allowed only for localhost");
    }
    if (base.search || base.hash) throw new Error("Use a backend base URL without a query or fragment");
    base.pathname = base.pathname.replace(/\/+$/, "") + "/";
    const endpoint = path => new URL("api/" + path, base).href;
    const staleAfterMs = options.staleAfterMs === undefined ? 15000 : options.staleAfterMs;
    if (!finite(staleAfterMs) || staleAfterMs <= 0) throw new Error("staleAfterMs must be a positive finite number");
    const requestTimeoutMs = options.requestTimeoutMs === undefined ? 15000 : options.requestTimeoutMs;
    if (!finite(requestTimeoutMs) || requestTimeoutMs <= 0) throw new Error("requestTimeoutMs must be a positive finite number");

    let closed = false;
    let transport = "connecting";
    let latest = null;
    let metadata = null;
    let lastReceivedAt = null;
    let lastProgressAt = null;
    let lastStreamReceivedAt = null;
    let resumedAt = null;
    let streamHasProgress = false;
    let transportError = null;
    let validationError = null;
    let metadataError = null;
    let metaPending = false;
    let openedAt = Date.now();
    const startedAt = openedAt;
    const controllers = new Set();
    let stream = null;
    let retryTimer = null;
    let retryDelayMs = 1000;

    function notify(name, ...args) {
      if (typeof options[name] !== "function") return;
      try { options[name](...args); }
      catch (error) { global.console?.error("FlyLive " + name + " callback failed", error); }
    }

    function snapshot() {
      const now = Date.now();
      const progressAgeMs = lastProgressAt === null ? null : Math.max(0, now - lastProgressAt);
      let phase;
      if (closed) phase = "closed";
      else if (transport !== "open") phase = transportError ? "disconnected" : "connecting";
      else if ((progressAgeMs !== null && progressAgeMs >= staleAfterMs) ||
               (!streamHasProgress && now - openedAt >= staleAfterMs)) phase = "stale";
      else phase = streamHasProgress ? "connected" : "connecting";
      return Object.freeze({
        phase,
        transport,
        baseUrl: base.href,
        streamUrl: endpoint("stream"),
        startedAt,
        lastReceivedAt,
        lastProgressAt,
        lastStreamReceivedAt,
        progressAgeMs,
        runId: latest?.run_id ?? null,
        streamId: latest?.stream_id ?? null,
        resumedAt,
        lastStep: latest?.step ?? null,
        simTime: latest?.sim_time ?? null,
        serverTime: latest?.server_time ?? null,
        metadataReady: metadata !== null,
        error: transportError || validationError || metadataError,
      });
    }

    function status() { notify("onStatus", snapshot()); }

    function accept(payload, source) {
      if (closed) return;
      // A delayed initial GET must never replace a state already received from
      // the stream (including a different run after a backend restart).
      if (source === "state" && lastStreamReceivedAt !== null) return;
      let state;
      try { state = validateState(payload); }
      catch (error) {
        validationError = "Rejected live state: " + error.message;
        status();
        return;
      }
      const now = Date.now();
      const sameRun = latest !== null && latest.run_id === state.run_id;
      const newEpoch = sameRun && typeof latest.stream_id === "string" &&
                       typeof state.stream_id === "string" && latest.stream_id !== state.stream_id;
      const sameEpoch = sameRun && !newEpoch;
      if (sameEpoch && state.step < latest.step) return;
      if (sameEpoch && state.sim_time < latest.sim_time) {
        validationError = "Rejected live state: simulated time moved backwards within a run";
        status();
        return;
      }
      const advances = sameEpoch && state.step > latest.step && state.sim_time > latest.sim_time;
      const changed = latest === null || !sameEpoch || state.step > latest.step;
      if (sameEpoch && state.step > latest.step && !advances) {
        validationError = "Rejected live state: step increased without simulated time advancing";
        status();
        return;
      }
      validationError = null;
      lastReceivedAt = now;
      if (source === "stream") {
        lastStreamReceivedAt = now;
        transportError = null;
        if (!sameEpoch) { streamHasProgress = false; openedAt = now; }
        if (advances) streamHasProgress = true;
      }
      if (changed) {
        if (!sameRun) resumedAt = null;
        if (newEpoch) resumedAt = now;
        latest = state;
        lastProgressAt = now;
        notify("onState", latest, Object.freeze({source, receivedAt: now, advances,
                                                 newRun: !sameRun, resumed: newEpoch}));
      }
      // Repeated packets may confirm the socket is open, but do not refresh
      // progressAgeMs or animate the same state again. A stalled backend becomes
      // stale even if it keeps sending its last cached snapshot.
      status();
    }

    async function get(path) {
      const controller = new AbortController();
      controllers.add(controller);
      const timer = global.setTimeout(() => controller.abort(), requestTimeoutMs);
      try {
        const response = await global.fetch(endpoint(path), {
          method: "GET", cache: "no-store", credentials: "omit", signal: controller.signal,
        });
        if (!response.ok) throw new Error("HTTP " + response.status + " for /api/" + path);
        return await response.json();
      } finally {
        global.clearTimeout(timer);
        controllers.delete(controller);
      }
    }

    async function loadMeta() {
      if (closed || metaPending) return;
      metaPending = true;
      try {
        const value = await get("meta");
        if (closed) return;
        if (!record(value)) throw new Error("/api/meta did not return an object");
        metadata = freeze(value);
        metadataError = null;
        notify("onMeta", metadata);
      } catch (error) {
        if (!closed) metadataError = "Metadata unavailable: " + (error.name === "AbortError" ? "request timed out" : error.message);
      } finally {
        metaPending = false;
        if (!closed) status();
      }
    }

    function retryClosedStream() {
      if (closed || retryTimer !== null) return;
      // Some HTTP failures permanently close EventSource. Retry those with one
      // bounded timer; leave readyState=0 retries to the native EventSource.
      retryTimer = global.setTimeout(() => {
        retryTimer = null;
        if (!closed) openStream();
      }, retryDelayMs);
      retryDelayMs = Math.min(retryDelayMs * 2, 30000);
    }

    function openStream() {
      if (closed) return;
      stream?.close();
      let current;
      try {
        current = new global.EventSource(endpoint("stream"), {withCredentials: false});
        stream = current;
      } catch (error) {
        transport = "closed";
        transportError = "Live stream unavailable; retrying: " + error.message;
        retryClosedStream();
        status();
        return;
      }
      current.addEventListener("open", () => {
        if (closed || current !== stream) return;
        if (retryTimer !== null) { global.clearTimeout(retryTimer); retryTimer = null; }
        retryDelayMs = 1000;
        transport = "open";
        transportError = null;
        streamHasProgress = false;
        openedAt = Date.now();
        if (metadata === null) loadMeta();
        status();
      });
      current.addEventListener("error", () => {
        if (closed || current !== stream) return;
        transport = current.readyState === 2 ? "closed" : "connecting";
        streamHasProgress = false;
        transportError = "Live stream disconnected; reconnecting";
        if (current.readyState === 2) retryClosedStream();
        status();
      });
      current.addEventListener("message", event => receive(event, current));
      current.addEventListener("state", event => receive(event, current));
    }

    function receive(event, current) {
      if (closed || current !== stream) return;
      try {
        if (typeof event.data !== "string" || event.data.length > 2000000) throw new Error("invalid or oversized state packet");
        accept(JSON.parse(event.data), "stream");
      } catch (error) {
        validationError = "Rejected live state: " + error.message;
        status();
      }
    }
    const timer = global.setInterval(status, Math.max(100, Math.min(1000, staleAfterMs / 4)));
    openStream();
    loadMeta();
    get("state").then(value => accept(value, "state")).catch(error => {
      if (!closed && lastStreamReceivedAt === null) {
        validationError = "Initial state unavailable: " + (error.name === "AbortError" ? "request timed out" : error.message);
        status();
      }
    });
    status();

    return Object.freeze({
      getState: () => latest,
      getMeta: () => metadata,
      snapshot,
      close() {
        if (closed) return;
        closed = true;
        transport = "closed";
        stream?.close();
        global.clearInterval(timer);
        if (retryTimer !== null) { global.clearTimeout(retryTimer); retryTimer = null; }
        controllers.forEach(controller => controller.abort());
        controllers.clear();
        status();
      },
    });
  }

  global.FlyLive = Object.freeze({connect});
})(window);
