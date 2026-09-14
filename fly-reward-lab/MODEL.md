# What this model computes

This package simulates a bounded rate model whose sparse wiring is constrained
by the downloaded connectome. It is a **hypothesis model**, not a reproduction
of Shiu et al., a calibrated biophysical emulation, or a measurement of pleasure.
All physiological time constants, gains, and initial states below are choices
made for this implementation. They have not been fitted to male-fly recordings.

## Recorded motor output and joint motion

The visualizer uses the individual rate states of identified motor neurons, recorded at every integration step, including the initial and final states. These recordings do not modify neural dynamics. [MOTOR_MAPPING.md](MOTOR_MAPPING.md) describes the annotation rules mapping 164 of 815 motor neurons to 24 labeled muscle groups. Remaining motor outputs are still saved, but have no invented movement assigned to them.

For each of the six legs and the trochanter/tibia joints, the bridge averages rates within the annotated flexor and extensor groups. With `r_max = config.max_rate`, it computes:

```
normalized_input = mean(group_motor_rates_hz) / r_max
activation[t] = activation[t-1] + (normalized_input[t] - activation[t-1]) * (1 - exp(-dt / 0.04))
joint_target = 1.2 * (flexor_activation - extensor_activation)  # radians
joint_offset[t] = joint_offset[t-1] + (joint_target[t] - joint_offset[t-1]) * (1 - exp(-dt / 0.12))
```

Initial activation equals the initial normalized neural rate; initial joint offsets are zero. Positive offsets mean flexion. The renderer preserves limb segment lengths, mirrors flexion correctly between anatomical sides, and limits illustrated interior angles to 0.02–π radians. Head, abdomen, body root and wings remain fixed. Motor mode has no time-driven poses, gait oscillator, motion noise, or dopamine-to-wriggle function. Decorative input-cable pulses do not move the body.

The normalization, equal motor-unit weighting, activation time constant, angular gain, relaxation time, neutral geometry and joint limits are **uncalibrated mechanical assumptions**. They are not muscle forces, measured physiological transfer functions or validated body dynamics. The playback is one-way and omits proprioceptive feedback, ground contact, body translation and physical torque. Sustained balanced antagonist activity can settle into a static pose. The delivered best-tested replay does so: its largest joint offset is approximately 2.88°, with zero recorded joint change during the final second.

## Fast network dynamics

Let C[i,j] be the nonnegative count of synapses from neuron j to neuron i.
Rows are targets and columns are sources. Define

```
W[i,j] = C[i,j] * sign[j] / sum_k C[i,k]
```

A row with no incoming edges is zero. Source signs are +1 for acetylcholine,
-1 for GABA, -1 for glutamate, and -1 for histamine. The glutamate sign is an
explicit assumption: the dataset does not supply a complete receptor-specific
sign map. Histamine inhibition is supported by fly Ort/HisCl1 histamine-gated
chloride-channel physiology, including inhibitory signaling in visual circuits;
extending this sign to every histaminergic target remains an assumption without
a complete receptor map. See [Gengs et al., 2002](https://pubmed.ncbi.nlm.nih.gov/12196539/)
and [Alejevski et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6335465/).
Dopamine,
serotonin, octopamine, and unknown labels receive zero fast coupling. Zero here
means the corresponding fast action is omitted, not that its biological effect
is absent or has a known sign. Synapse counts provide structural weights; they
are not measurements of conductance or effective synaptic strength.

The model accepts full transmitter names and common abbreviations, including
HIS and HA for histamine. Missing/unknown annotations remain `unknown`.
Unexpected labels retain their normalized names, receive zero fast coupling,
and appear in `unmapped_transmitter_counts`; they are not silently relabeled
as missing annotations. The actual loaded graph has 5,910 histaminergic cells
and 3,602 cells with unknown transmitter, both explicitly distinguished.

For each neuron, a fixed seeded baseline b[i] is sampled uniformly from 0.03 to
0.07. The target rate is

```
target[i] = Rmax * clip(b[i] + g * sum_j W[i,j] * r[j]/Rmax
                       + 0.10 * occupancy[i] * sensitivity[i] + 0.15 * npf, 0, 1)
dr[i]/dt = (target[i] - r[i]) / 0.05 seconds
```

The rate update uses the exact first-order filter for a frozen target each
step. All rates remain in [0, Rmax]. Default Rmax is 100 Hz: an arbitrary
refractory-inspired numerical ceiling, not a measured universal fly limit.
Network gain g must lie in [0, 0.95]. The incoming normalization bounds the
absolute row sum by one and keeps recurrent amplification controlled.

Positive `dopamine_drive` clamps every neuron identified by its dopamine
transmitter label to `dopamine_drive * Rmax` throughout the run, including the
initial state. A zero drive means no clamp, rather than silencing dopamine
neurons. Non-dopamine cells are never directly clamped by that parameter.

## Dopamine, clearance, and optional adaptation

Dopamine is a dimensionless per-target state D[i] in [0,1], not a measured
concentration in molar units. The anatomical projection P[i,j] retains only
edges whose source j has a dopamine transmitter label, and normalizes counts
by each target's total incoming DA synapse count. For a target with no such
incoming edges, its row is zero and it receives no modeled DA modulation.
The states follow

```
P[i,j] = DA_source[j] * C[i,j] / sum_k DA_source[k] * C[i,k]
release[i] = 2 * sum_j P[i,j] * r[j]/Rmax / second
clearance = 2 * reuptake_factor / second
dD[i]/dt = release[i] * (1 - D[i]) - clearance * D[i]
occupancy[i] = D[i] / (D[i] + 0.25)
```

The update integrates this equation exactly with rates held fixed for one
step. Smaller `reuptake_factor` slows abstract clearance. Zero disables this
clearance mechanism. These equations do not model a particular transporter,
drug, receptor subtype, or delivery method. DA synapse counts supply a
locality and relative-weight prior. Extrasynaptic diffusion is omitted, and
each anatomical target is assigned the same hypothetical receptor dynamics
without evidence of its actual receptor distribution. Absolute DA input count
does not scale release because each recipient's DA weights are normalized.

With adaptation enabled, per-target receptor sensitivity S[i] in [0,1] follows

```
dS[i]/dt = 0.2 * (1-S[i]) - 0.8 * occupancy[i] * S[i]
```

Rates in that equation are per second, and the update is exact for frozen
occupancy. The first term allows recovery when occupancy falls, so the model
represents reversible desensitization rather than permanent damage. Setting
`tolerance=False` holds sensitivity at one. This switch disables one modeled
adaptation mechanism; it does not establish the biological elimination of all
tolerance mechanisms.

`dopamine_reward_proxy = mean(occupancy[i] * sensitivity[i])` over anatomical
DA recipients is an explicitly chosen readout. Concentration and sensitivity
traces also average over anatomical recipients. If there are no recipients,
concentration and reward proxy are zero, and sensitivity is reported as one.
The identical receptor response assigned to those targets is hypothetical. A
greater scalar is not evidence of greater biological reward or pleasure.
Dopamine cell classes can in fact carry different and opposite reinforcement
signals; this model does not resolve those distinctions.

## NPF-associated abstract drive

The supplied transmitter classes do not authenticate NPF cell identities or
an NPF receptor map. The package therefore does not invent cell IDs, infer NPF
from dopamine labels, or substitute a human opioid receptor. It uses a declared
global variable N:

```
dN/dt = (npf_drive - N) / 0.3 seconds
```

The exact first-order filter starts at zero and remains in [0,1]. Its value is
reported as `npf_exposure_proxy`. The additive global gain in the network rate
equation is an unvalidated modeling choice, not an established cellular NPF
mechanism. This component is useful for comparing assumptions only.

## Independent response and saturation readouts

An independent shadow trajectory begins at 60% of the run, initialized from
the primary trajectory and sharing its dopamine and NPF state. During the
60%-to-80% interval, the shadow receives an added drive of 0.02 on a seeded 5%
of non-dopamine neurons (at least one when available). The input then stops;
the shadow continues to the end. The primary trajectory never receives this
probe. The response readout is the mean absolute rate difference over all
non-dopamine neurons, and its mean metric uses only the active probe interval.

This tests responsiveness to a numerical input, not odor discrimination,
learned preference, behavior, or consciousness. `saturation_fraction` is the
fraction of neurons at or above 95% of Rmax. These readouts help expose trivial
high-output or unresponsive regimes; they do not validate the reward proxy.

## Presets and reproducibility

| Preset | DA clamp fraction | Clearance factor | NPF drive | Adaptation |
|---|---:|---:|---:|---|
| baseline | 0, no clamp | 1 | 0 | on |
| dopamine | 0.5 | 0.25 | 0 | on |
| npf | 0, no clamp | 1 | 0.6 | on |
| combined | 0.5 | 0.25 | 0.6 | on |
| combined_no_tolerance | 0.5 | 0.25 | 0.6 | off |
| maximal_dopamine | 1 | 0.05 | 0 | off |

The last preset combines a maximal DA firing clamp, slow clearance, and disabled
adaptation. It changes all three dopamine assumptions; it is not an isolated
test of firing alone. `maximal` refers to the configured clamp ceiling, not a
demonstration of a global biological optimum.

The time step must be in (0, 0.02] seconds, and duration must contain at least
five whole steps. Results record the configuration and all constants. The
default seed is fixed, so repeated identical inputs reproduce the same output.
Aggregate metrics are integrated at every step and do not depend on trace
downsampling. All reported states are checked for finiteness.

The model omits morphology, spike timing, electrical synapses, authentic
receptor distributions, cell-specific dopamine/NPF actions, measured resting
states, synaptic learning, metabolism, and embodied sensory feedback. Optimizing
its proxies explores this implementation's assumptions. It does not establish
that the downloaded wiring supports the modeled physiological states or any
subjective experience.
