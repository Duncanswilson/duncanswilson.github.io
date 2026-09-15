/* Fly illustration with an explicit motor-pose rendering mode.
 * The projected joint geometry and limits are illustrative assumptions, not
 * validated biomechanics. In motor-pose mode, geometry never depends on time.
 */
(function () {
  "use strict";

  const TAU = Math.PI * 2;
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const number = (value, fallback) => Number.isFinite(Number(value)) ? Number(value) : fallback;
  const portColors = ["#69d8bc", "#eda0a2", "#d8c184"];

  function ellipse(ctx, x, y, rx, ry, rotation, fill, stroke, width) {
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, rotation || 0, 0, TAU);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width || 1; ctx.stroke(); }
  }

  function line(ctx, points, color, width) {
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.lineWidth = width;
    ctx.strokeStyle = color;
    ctx.stroke();
  }

  function transformed(point, x, y, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    return {x: x + point.x * c - point.y * s, y: y + point.x * s + point.y * c};
  }

  function rotateVector(vector, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    return [vector[0] * c - vector[1] * s, vector[0] * s + vector[1] * c];
  }

  const subtract = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const object = value => value && typeof value === "object" && !Array.isArray(value) ? value : {};

  function flexionRotation(proximal, distal, offset) {
    // Interior angle = pi - |signed external turn|. Derive the screen rotation
    // sign from the neutral joint itself, so positive flexion closes both sides
    // of the fly instead of applying an arbitrary shared screen-space sign.
    const turn = Math.atan2(proximal[0] * distal[1] - proximal[1] * distal[0],
                            proximal[0] * distal[0] + proximal[1] * distal[1]);
    const bend = Math.abs(turn);
    const target = clamp(bend + number(offset, 0), 0, Math.PI - 0.02);
    return (Math.sign(turn) || 1) * (target - bend);
  }

  function interiorAngle(a, joint, b) {
    const u = subtract(a, joint), v = subtract(b, joint);
    const length = Math.hypot(...u) * Math.hypot(...v);
    return Math.acos(clamp((u[0] * v[0] + u[1] * v[1]) / length, -1, 1));
  }

  function motorLegPose(spec, input) {
    input = object(input);
    const base = spec.base.slice();
    const thigh = subtract(spec.knee, base);
    // A short coxa is mostly covered by the thorax. Its small neutral bend makes
    // extension/flexion distinct while preserving the authored knee and foot.
    const side = spec.near ? 1 : -1;
    const neutralHip = [base[0] + thigh[0] * 0.17 - side * thigh[1] * 0.06,
                        base[1] + thigh[1] * 0.17 + side * thigh[0] * 0.06];
    const coxaVector = subtract(neutralHip, base);
    const femurVector = subtract(spec.knee, neutralHip);
    const tibiaVector = subtract(spec.ankle, spec.knee);
    const tarsusVector = subtract(spec.foot, spec.ankle);
    const coxa = number(input.coxa, 0);
    const trochanter = flexionRotation(coxaVector, femurVector, input.trochanter);
    const tibia = flexionRotation(femurVector, tibiaVector, input.tibia);
    const hip = add(base, rotateVector(coxaVector, coxa));
    const knee = add(hip, rotateVector(femurVector, coxa + trochanter));
    const ankle = add(knee, rotateVector(tibiaVector, coxa + trochanter + tibia));
    const foot = add(ankle, rotateVector(tarsusVector, coxa + trochanter + tibia));
    return {base, hip, knee, ankle, foot,
            trochanterInteriorAngle: interiorAngle(base, hip, knee),
            kneeInteriorAngle: interiorAngle(hip, knee, ankle)};
  }

  function motorKinematics(input) {
    // Public pure API: motorKinematics({legs:{LF:{coxa,trochanter,tibia},...},
    // wingAngles:{L,R}, bodyRoll, abdomenAngle, headAngle}). Angles are radians.
    // Trochanter/tibia are positive-flexion offsets; other angles are signed
    // offsets from neutral. Coordinates are local [x,y] arrays before body roll.
    // Omitted/nonfinite inputs are neutral. No neuron-to-joint mapping is made
    // here: the caller must supply its documented motor-output bridge.
    input = object(input);
    const requestedLegs = object(input.legs), wingAngles = object(input.wingAngles);
    const posed = {};
    for (const spec of legs) posed[spec.id] = motorLegPose(spec, requestedLegs[spec.id]);
    return {
      legs: posed,
      wingAngles: {L: number(wingAngles.L, 0), R: number(wingAngles.R, 0)},
      bodyRoll: number(input.bodyRoll, 0),
      abdomenAngle: number(input.abdomenAngle, 0),
      headAngle: number(input.headAngle, 0)
    };
  }

  // Orthographic camera: x=head-to-tail, y=anatomical left, z=height.
  // The same projection is applied to every simulated endpoint and the floor.
  const projectBodyPoint = p => [p[0] * 85 + p[1] * 22, p[1] * 48 - p[2] * 85];
  function physicalKinematics(input) {
    const origin = projectBodyPoint(input.bodyPositionMm);
    const posed = {};
    for (const spec of legs) {
      const points = input.legPointsMm[spec.id];
      const local = key => subtract(projectBodyPoint(points[key]), origin);
      const hip = local('hip'), knee = local('knee'), foot = local('foot');
      // The distal link includes the tarsus in the reduced plant. Its painted
      // subdivision is collinear, not an additional animated joint.
      const ankle = [knee[0] + .78 * (foot[0] - knee[0]), knee[1] + .78 * (foot[1] - knee[1])];
      posed[spec.id] = {base:hip, hip, knee, ankle, foot};
    }
    return {legs:posed, origin};
  }

  function drawShadow(ctx, roll, amplitude) {
    ctx.save();
    ctx.translate(18 + roll * 18, 39);
    ctx.scale(1, 0.40);
    const shadow = ctx.createRadialGradient(0, 0, 14, 0, 0, 147);
    shadow.addColorStop(0, "rgba(0,0,0,.43)");
    shadow.addColorStop(0.48, "rgba(0,0,0,.21)");
    shadow.addColorStop(1, "rgba(0,0,0,0)");
    ellipse(ctx, 0, 0, 147 + amplitude * 3, 90, 0, shadow);
    ctx.restore();
  }

  function legPose(spec, t, motion) {
    const phase = spec.phase;
    const flex = Math.sin(t * 2.5 + phase) * motion;
    const curl = Math.sin(t * 3.8 + phase + 0.65) * motion;
    const knee = [spec.knee[0] + flex * 16, spec.knee[1] + curl * 12];
    const ankle = [spec.ankle[0] + flex * 24, spec.ankle[1] - curl * 20];
    const ankleToFoot = [spec.foot[0] - spec.ankle[0], spec.foot[1] - spec.ankle[1]];
    const turn = curl * (spec.near ? 0.60 : 0.42);
    const c = Math.cos(turn), s = Math.sin(turn);
    const foot = [
      ankle[0] + ankleToFoot[0] * c - ankleToFoot[1] * s,
      ankle[1] + ankleToFoot[0] * s + ankleToFoot[1] * c
    ];
    return {base: spec.base, knee, ankle, foot};
  }

  function drawLeg(ctx, pose, near) {
    const {base, knee, ankle, foot} = pose;
    const hip = pose.hip || base;
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    const dark = near ? "#191d1e" : "#293333";
    const lit = near ? "#566465" : "#3f4c4b";
    // Separate tapered segments make the joints visible even at small scale.
    if (pose.hip) line(ctx, [base, hip], dark, near ? 6.5 : 4.8);
    line(ctx, [hip, knee], dark, near ? 7.4 : 5.4);
    line(ctx, [[hip[0] - 1.1, hip[1] - 1.1], [knee[0] - 1.1, knee[1] - 1.1]], lit, 1.5);
    line(ctx, [knee, ankle], dark, near ? 4.5 : 3.4);
    line(ctx, [[knee[0] - 0.7, knee[1] - 0.7], [ankle[0] - 0.7, ankle[1] - 0.7]], lit, 0.85);
    line(ctx, [ankle, foot], dark, near ? 2.9 : 2.3);
    ellipse(ctx, knee[0], knee[1], near ? 4.4 : 3.5, 3.3, -0.3, "#232b2b", "#62716b", 0.7);
    ellipse(ctx, ankle[0], ankle[1], 2.6, 2.2, 0, "#252c2b");

    const dx = foot[0] - ankle[0], dy = foot[1] - ankle[1];
    const length = Math.hypot(dx, dy) || 1;
    const ux = dx / length, uy = dy / length;
    for (let i = 1; i < 5; i++) {
      const f = i / 6;
      const x = ankle[0] + dx * f, y = ankle[1] + dy * f;
      line(ctx, [[x - uy * 1.5, y + ux * 1.5], [x + uy * 1.5, y - ux * 1.5]], "#76837a", 0.65);
    }
    // Paired terminal claws, relaxed and curled rather than splayed straight.
    ctx.strokeStyle = dark;
    ctx.lineWidth = 1.25;
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(foot[0], foot[1]);
      ctx.quadraticCurveTo(foot[0] + ux * 7 + uy * side * 4,
                          foot[1] + uy * 7 - ux * side * 4,
                          foot[0] + ux * 5 + uy * side * 6,
                          foot[1] + uy * 5 - ux * side * 6);
      ctx.stroke();
    }
    // Sparse tibial bristles, fixed in segment space rather than frame noise.
    const sx = ankle[0] - knee[0], sy = ankle[1] - knee[1];
    const sl = Math.hypot(sx, sy) || 1;
    for (let i = 1; i < 8; i++) {
      const f = i / 9, side = i % 2 ? 1 : -1;
      const x = knee[0] + sx * f, y = knee[1] + sy * f;
      line(ctx, [[x, y], [x + side * sy / sl * 4.5 - sx / sl * 1.5,
                          y - side * sx / sl * 4.5 - sy / sl * 1.5]], near ? "#293331" : "#3a4942", 0.65);
    }
    ctx.restore();
  }

  function abdomenPath(ctx) {
    ctx.beginPath();
    ctx.moveTo(-69, -25);
    ctx.bezierCurveTo(-41, -50, 17, -48, 58, -27);
    ctx.bezierCurveTo(84, -14, 95, -3, 96, 5);
    ctx.bezierCurveTo(73, 41, 15, 52, -32, 35);
    ctx.bezierCurveTo(-63, 27, -77, 0, -69, -25);
    ctx.closePath();
  }

  function drawAbdomen(ctx, t, motion, motorAngle) {
    const driven = motorAngle !== undefined;
    const curl = driven ? 0 : Math.sin(t * 1.9 + 0.8) * motion;
    ctx.save();
    // The abdomen folds gently toward the thorax, then stretches into a twist.
    // Its attachment stays beneath the thorax so the motion reads as a curl.
    ctx.translate(driven ? 77 : 77 - (1 - Math.cos(t * 1.9 + 0.8)) * motion * 5, 20 + curl * 9);
    ctx.rotate(0.19 + (driven ? motorAngle : curl * 0.15));
    ctx.scale(1 - Math.abs(curl) * 0.045, 1 + Math.abs(curl) * 0.025);
    const shell = ctx.createLinearGradient(-16, -45, 20, 51);
    shell.addColorStop(0, "#56625f");
    shell.addColorStop(0.29, "#303a37");
    shell.addColorStop(0.64, "#202523");
    shell.addColorStop(1, "#101514");
    abdomenPath(ctx);
    ctx.fillStyle = shell;
    ctx.fill();
    ctx.save();
    ctx.clip();
    // Curved tergites wrap around the abdomen with a narrow reflective rim.
    for (let i = 0; i < 6; i++) {
      const x = -41 + i * 23;
      ctx.beginPath();
      ctx.moveTo(x - 11, -54);
      ctx.bezierCurveTo(x + 13, -26, x + 16, 16, x - 7, 52);
      ctx.lineWidth = 6.2;
      ctx.strokeStyle = "rgba(9,13,12,.72)";
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 14, -54);
      ctx.bezierCurveTo(x + 10, -26, x + 13, 16, x - 10, 52);
      ctx.lineWidth = 1.1;
      ctx.strokeStyle = "rgba(158,170,135,.36)";
      ctx.stroke();
    }
    const sheen = ctx.createRadialGradient(-26, -24, 2, -12, -19, 84);
    sheen.addColorStop(0, "rgba(173,188,155,.28)");
    sheen.addColorStop(1, "rgba(126,160,143,0)");
    ellipse(ctx, -1, -19, 86, 23, -0.08, sheen);
    ctx.restore();
    abdomenPath(ctx);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "#151b19";
    ctx.stroke();
    for (let i = 0; i < 23; i++) {
      const a = 0.12 + i / 22 * Math.PI;
      const x = 4 + Math.cos(a) * 78, y = Math.sin(a) * 40;
      line(ctx, [[x, y], [x + Math.cos(a) * 4, y + Math.sin(a) * 5]], "rgba(45,52,38,.7)", 0.65);
    }
    ctx.restore();
  }

  function wingPath(ctx) {
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.bezierCurveTo(35, -27, 119, -59, 173, -41);
    ctx.bezierCurveTo(199, -32, 192, -9, 167, 7);
    ctx.bezierCurveTo(124, 36, 47, 23, 0, 0);
    ctx.closePath();
  }

  function drawWing(ctx, x, y, angle, narrow, alpha, t, motion, phase, motorAngle) {
    const driven = motorAngle !== undefined;
    const flutter = driven ? 0 : Math.sin(t * 13 + phase) * Math.pow(Math.max(0, Math.sin(t * 1.1 + phase)), 8);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle + (driven ? motorAngle : flutter * motion * 0.045));
    ctx.scale(1, narrow + flutter * motion * 0.025);
    ctx.globalAlpha *= alpha;
    wingPath(ctx);
    const membrane = ctx.createLinearGradient(17, -43, 123, 24);
    membrane.addColorStop(0, "rgba(212,234,226,.37)");
    membrane.addColorStop(0.38, "rgba(164,198,187,.16)");
    membrane.addColorStop(0.76, "rgba(223,231,206,.32)");
    membrane.addColorStop(1, "rgba(150,180,168,.12)");
    ctx.fillStyle = membrane;
    ctx.fill();
    ctx.strokeStyle = "rgba(132,160,145,.65)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.save();
    ctx.clip();
    const veins = [
      [0, 0, 66, -23, 171, -38],
      [0, 0, 88, -5, 184, -23],
      [0, 0, 83, 7, 165, 8],
      [0, 0, 66, 19, 119, 22],
      [60, -20, 78, -7, 83, 7],
      [106, -17, 114, -3, 117, 16],
      [131, -29, 139, -19, 144, -8],
      [69, 3, 67, 12, 69, 20]
    ];
    ctx.lineCap = "round";
    for (let i = 0; i < veins.length; i++) {
      const v = veins[i];
      ctx.beginPath();
      ctx.moveTo(v[0], v[1]);
      ctx.quadraticCurveTo(v[2], v[3], v[4], v[5]);
      ctx.strokeStyle = i < 4 ? "rgba(83,110,96,.54)" : "rgba(98,126,108,.42)";
      ctx.lineWidth = i < 4 ? 1.1 : 0.8;
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(38, -19);
    ctx.bezierCurveTo(74, -38, 127, -44, 163, -37);
    ctx.strokeStyle = "rgba(249,250,222,.48)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
    ctx.restore();
  }

  function drawThorax(ctx) {
    ctx.save();
    ctx.rotate(0.14);
    const shell = ctx.createRadialGradient(-20, -30, 3, -2, -6, 62);
    shell.addColorStop(0, "#69746c");
    shell.addColorStop(0.25, "#404e46");
    shell.addColorStop(0.7, "#202b26");
    shell.addColorStop(1, "#111914");
    ellipse(ctx, -1, -8, 52, 42, 0, shell, "#141d17", 1.5);
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(-1, -8, 51, 41, 0, 0, TAU);
    ctx.clip();
    // Dorsal striping and fine longitudinal texture in the cuticle.
    for (let i = -2; i <= 2; i++) {
      ctx.beginPath();
      ctx.moveTo(-44, -17 + i * 11);
      ctx.bezierCurveTo(-12, -28 + i * 12, 19, -17 + i * 10, 49, -6 + i * 9);
      ctx.strokeStyle = i % 2 ? "rgba(12,23,17,.35)" : "rgba(175,184,147,.13)";
      ctx.lineWidth = i % 2 ? 4 : 1.3;
      ctx.stroke();
    }
    for (let i = 0; i < 50; i++) {
      const a = i * 2.3999632297;
      const radial = Math.sqrt((i + 0.5) / 50);
      const x = Math.cos(a) * 43 * radial;
      const y = -8 + Math.sin(a) * 32 * radial;
      line(ctx, [[x, y], [x - 3.5, y - 4.5]], "rgba(15,24,18,.46)", 0.7);
    }
    ctx.restore();
    for (let i = 0; i < 16; i++) {
      const a = -Math.PI + i / 15 * Math.PI * 1.4;
      const x = Math.cos(a) * 47, y = -8 + Math.sin(a) * 36;
      const length = 7 + (i % 3) * 2;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.quadraticCurveTo(x + Math.cos(a) * length, y + Math.sin(a) * length,
                          x + Math.cos(a) * length + 2, y + Math.sin(a) * length - 2);
      ctx.strokeStyle = "#28372b";
      ctx.lineWidth = 1.05;
      ctx.stroke();
    }
    // Small amber halteres are visible at the base of the flight apparatus.
    line(ctx, [[34, 17], [48, 26]], "#555345", 2.6);
    ellipse(ctx, 50, 28, 5, 3.7, 0.5, "#817456", "#333c2d", 1);
    ctx.restore();
  }

  function drawEye(ctx, x, y, rx, ry, angle, brightness) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    const eye = ctx.createRadialGradient(-rx * 0.36, -ry * 0.4, 1, 0, 1, ry * 1.2);
    eye.addColorStop(0, "#cc7160");
    eye.addColorStop(0.3, "#a8473c");
    eye.addColorStop(0.72, "#6e2625");
    eye.addColorStop(1, "#341c1d");
    ellipse(ctx, 0, 0, rx, ry, 0, eye, "#2b2721", 1.6);
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(0, 0, rx - 0.8, ry - 0.8, 0, 0, TAU);
    ctx.clip();
    const size = 2.45;
    for (let row = -9; row <= 9; row++) {
      for (let col = -9; col <= 9; col++) {
        const cx = col * size * 1.73 + (row % 2) * size * 0.866;
        const cy = row * size * 1.5;
        if ((cx / rx) ** 2 + (cy / ry) ** 2 > 1.2) continue;
        ctx.beginPath();
        for (let corner = 0; corner < 6; corner++) {
          const a = (corner + 0.5) / 6 * TAU;
          const px = cx + Math.cos(a) * size, py = cy + Math.sin(a) * size;
          if (corner === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.strokeStyle = "rgba(49,13,19,.33)";
        ctx.lineWidth = 0.58;
        ctx.stroke();
        if ((row + col) % 3 === 0) ellipse(ctx, cx - 0.5, cy - 0.6, 0.45, 0.45, 0, "rgba(255,168,120,.22)");
      }
    }
    ellipse(ctx, -rx * 0.34, -ry * 0.42, rx * 0.20, ry * 0.25, -0.55,
            "rgba(255,227,185," + (0.16 + brightness * 0.05) + ")");
    ctx.restore();
    ctx.restore();
  }

  function drawHead(ctx, t, motion, dopamine, npf, motorAngle) {
    const driven = motorAngle !== undefined;
    const headX = -78, headY = driven ? -22 : -22 + Math.sin(t * 2.2 + 0.5) * motion * 1.2;
    const headAngle = -0.055 + (driven ? motorAngle : Math.sin(t * 1.8 + 1) * motion * 0.028);
    const pads = [{x: 11, y: -28}, {x: 26, y: -15}, {x: 19, y: 4}];
    ctx.save();
    ctx.translate(headX, headY);
    ctx.rotate(headAngle);
    const cap = ctx.createLinearGradient(-16, -35, 17, 30);
    cap.addColorStop(0, "#647064");
    cap.addColorStop(0.38, "#364435");
    cap.addColorStop(1, "#18261d");
    ellipse(ctx, 0, 0, 38, 33, -0.12, cap, "#162317", 1.4);
    drawEye(ctx, -3, -24, 21, 13, -0.16, dopamine);
    drawEye(ctx, -17, 4, 23.5, 28.5, -0.21, dopamine);
    ellipse(ctx, 2, -3, 9, 20, -0.35, "rgba(117,128,94,.25)");

    // Antennae, aristae, and a small tucked proboscis complete the silhouette.
    const antennaTurn = driven ? 0 : Math.sin(t * 3.4) * motion * 2;
    line(ctx, [[-28, -10], [-43, -16], [-48, -13 + antennaTurn]], "#283927", 2.3);
    ellipse(ctx, -45, -13 + antennaTurn, 4.8, 3, -0.5, "#6a7351", "#2b3b25", 1);
    line(ctx, [[-45, -14 + antennaTurn], [-52, -27 + antennaTurn], [-65, -31 + antennaTurn]], "#34462c", 0.85);
    for (let i = 0; i < 5; i++) {
      const x = -52 - i * 2.4, y = -27 - i * 0.75 + antennaTurn;
      line(ctx, [[x, y], [x + 0.3, y - 4 + (i % 2)]], "#405138", 0.6);
      line(ctx, [[x, y], [x - 2, y + 3]], "#405138", 0.6);
    }
    line(ctx, [[-29, 18], [-39, 22]], "#263822", 3.5);
    ellipse(ctx, -40, 23, 5.3, 3.6, -0.12, "#536244", "#2b3a23", 1);
    for (let i = 0; i < 8; i++) {
      const x = 2 + i * 3.5, y = -25 + i * 1.4;
      line(ctx, [[x, y], [x + 2, y - 7]], "rgba(33,51,29,.7)", 0.75);
    }

    // Surface electrodes sit on the cuticle: small soft pads, no penetration.
    const ports = [];
    for (let i = 0; i < pads.length; i++) {
      const p = pads[i], color = portColors[i];
      ellipse(ctx, p.x + 1.3, p.y + 2.7, 7.7, 4.3, -0.3, "rgba(0,0,0,.25)");
      ellipse(ctx, p.x, p.y, 7.2, 4.4, -0.3, "#c8cabb", "#4d6352", 0.9);
      ellipse(ctx, p.x, p.y - 0.5, 4.8, 2.7, -0.3, color);
      line(ctx, [[p.x, p.y - 1], [p.x, p.y - 6]], "#cfdbc3", 1.7);
      ellipse(ctx, p.x, p.y - 6, 2.25, 1.9, 0, "#e5ede0", "#485d4d", 0.7);
      const level = i === 0 ? dopamine : i === 1 ? npf : (dopamine + npf) / 2;
      if (level > 0.015) {
        ctx.save();
        const light = ctx.createRadialGradient(p.x, p.y - 1, 0, p.x, p.y - 1, 11);
        light.addColorStop(0, i === 0 ? "rgba(105,216,188,.19)" : i === 1 ? "rgba(237,160,162,.19)" : "rgba(216,193,132,.19)");
        light.addColorStop(1, "rgba(255,255,255,0)");
        ctx.globalAlpha *= level;
        ellipse(ctx, p.x, p.y - 1, 11, 8, 0, light);
        ctx.restore();
      }
      const location = transformed({x: p.x, y: p.y - 6}, headX, headY, headAngle);
      ports.push({x: location.x, y: location.y, color});
    }
    ctx.restore();
    return {ports, head: {x: headX, y: headY}};
  }

  // Camera convention: head faces screen-left. Anatomical LEFT is the lower,
  // near row; RIGHT is the upper, far row. F/M/H mean front/middle/hind.
  const legs = [
    {id: "RH", base: [27, -22], knee: [80, -76], ankle: [151, -83], foot: [193, -49], phase: 0.2, near: false},
    {id: "RM", base: [-1, -30], knee: [8, -99], ankle: [53, -139], foot: [104, -134], phase: 2.2, near: false},
    {id: "RF", base: [-31, -28], knee: [-79, -99], ankle: [-143, -113], foot: [-173, -78], phase: 4.0, near: false},
    {id: "LH", base: [28, 7], knee: [98, 72], ankle: [135, 115], foot: [180, 101], phase: 3.2, near: true},
    {id: "LM", base: [-2, 14], knee: [-12, 81], ankle: [29, 132], foot: [80, 137], phase: 5.0, near: true},
    {id: "LF", base: [-34, 7], knee: [-103, 52], ankle: [-149, 94], foot: [-194, 80], phase: 0.8, near: true}
  ];

  function draw(ctx, options) {
    options = options || {};
    // Supplying even an empty/null motorPose selects neutral explicit geometry;
    // missing groups NEVER fall back to the authored oscillatory animation.
    const hasMotorPose = Object.prototype.hasOwnProperty.call(options, "motorPose");
    const motor = hasMotorPose ? motorKinematics(options.motorPose) : null;
    const physical = hasMotorPose && options.motorPose?.legPointsMm ? physicalKinematics(options.motorPose) : null;
    const reduced = Boolean(options.reducedMotion);
    const t = hasMotorPose || reduced ? 0 : number(options.time, 0);
    const motion = hasMotorPose || reduced ? 0 : clamp(number(options.motion, 0.35), 0, 1);
    const dopamine = clamp(number(options.dopamine, 0), 0, 1);
    const npf = clamp(number(options.npf, 0), 0, 1);
    const roll = hasMotorPose ? motor.bodyRoll : motion * (Math.sin(t * 1.9) * 0.19 + Math.sin(t * 0.77) * 0.055);
    const bodyY = physical ? physical.origin[1] : hasMotorPose ? 0 : motion * (Math.cos(t * 1.9) - 1) * 3.2;
    const bodyX = physical ? physical.origin[0] : hasMotorPose ? 0 : motion * Math.sin(t * 1.9) * 2.4;
    ctx.save();
    if (!physical) drawShadow(ctx, roll, motion);
    else {
      // A body shadow projected onto z=0; no fake contact patches under lifted feet.
      const center = projectBodyPoint([options.motorPose.bodyPositionMm[0],options.motorPose.bodyPositionMm[1],0]);
      ellipse(ctx,center[0],center[1],95,20,0,'rgba(0,0,0,.14)');
    }
    ctx.translate(bodyX, bodyY);
    ctx.rotate(roll);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const poses = legs.map(spec => physical ? physical.legs[spec.id] : hasMotorPose ? motor.legs[spec.id] : legPose(spec, t, motion));
    // Soft contact patches under feet keep the moving anatomy on the floor.
    for (let i = 0; i < poses.length; i++) {
      const foot = poses[i].foot;
      if (!physical) ellipse(ctx, foot[0] + 2, foot[1] + 4, 9, 3, 0, "rgba(0,0,0,.12)");
      else if (options.motorPose.legPointsMm[legs[i].id].foot[2] <= 0) {
        const floor = projectBodyPoint([...options.motorPose.legPointsMm[legs[i].id].foot.slice(0,2),0]);
        ellipse(ctx,floor[0]-bodyX,floor[1]-bodyY,5,2,0,'rgba(0,0,0,.3)');
      }
    }
    for (let i = 0; i < 3; i++) drawLeg(ctx, poses[i], false);
    drawWing(ctx, 7, -30, -0.36, 0.85, 0.75, t, motion, 1.5, hasMotorPose ? motor.wingAngles.R : undefined);
    drawAbdomen(ctx, t, motion, hasMotorPose ? motor.abdomenAngle : undefined);
    drawWing(ctx, 10, -20, 0.045, 0.93, 0.92, t, motion, 0, hasMotorPose ? motor.wingAngles.L : undefined);
    for (let i = 3; i < 6; i++) drawLeg(ctx, poses[i], true);
    drawThorax(ctx);
    const anchors = drawHead(ctx, t, motion, dopamine, npf, hasMotorPose ? motor.headAngle : undefined);
    ctx.restore();
    const legJoints = {};
    for (let i = 0; i < legs.length; i++) {
      const pose = poses[i], points = {};
      for (const key of ["base", "hip", "knee", "ankle", "foot"]) {
        if (pose[key]) points[key] = transformed({x: pose[key][0], y: pose[key][1]}, bodyX, bodyY, roll);
      }
      if (hasMotorPose) {
        points.trochanterInteriorAngle = pose.trochanterInteriorAngle;
        points.kneeInteriorAngle = pose.kneeInteriorAngle;
      }
      legJoints[legs[i].id] = points;
    }
    return {
      ports: anchors.ports.map(port => Object.assign(transformed(port, bodyX, bodyY, roll), {color: port.color})),
      head: transformed(anchors.head, bodyX, bodyY, roll),
      legJoints,
      motionSource: physical ? "simulated_body_geometry" : hasMotorPose ? "explicit_motor_pose" : "authored_animation"
    };
  }

  window.FlyActor = Object.freeze({draw, motorKinematics, physicalKinematics, projectBodyPoint});
})();
