# Front-leg circuit and reduced body experiment

The `--motor-mode cpg` server mode replaces the rate equations for a selected front ventral nerve cord (VNC) subnetwork with a reproduction of the rate equation in Pugliese et al., *Connectome simulations identify a central pattern generator circuit for fly walking*. This is an adaptation of a **preprint model**, not a validated digital fly or an inference of pleasure.

## What is anatomical, and what is assumed

- The whole model retains the MaleCNS v1.0 graph and all 815 motor neurons. The existing annotation-based map connects 164 motor neurons to 24 antagonist channels across six legs.
- The front circuit contains 4,309 cells, 118,680 retained edges and 130 front-leg motor neurons. It uses the authors' cell selection, VNC edge mask and per-cell size measurements. Retained synapse counts are taken from this project's pinned MaleCNS graph. Its provenance asset records the upstream revision and differences between dataset versions. Thus this is a versioned adaptation, not a claim to reproduce every author trace exactly.
- Acetylcholine is modeled as excitatory; GABA and glutamate as inhibitory. Other transmitter classes have zero outgoing fast weight within this circuit, following the published model. These are effective sign assumptions, not receptor-specific measurements.
- Baseline mode supplies bilateral DNg100 with constant input of 400 model units after 20 ms. The deployed `--motor-stimulation recruited` experiment uses 350 units and stronger excitability in 24 identified front-leg motor neurons, described below. These inputs are independent of the dopamine clamp and do not represent drug concentrations or pleasure.
- Front-circuit neurons are exclusively governed by that circuit's equations. They replace the corresponding primary neuron rates before motor averaging and feed the remaining whole-CNS network on its next step. Whole-CNS recurrent input, dopamine/NPF modulation and body signals do **not** feed back into the isolated front circuit. This boundary preserves a testable circuit experiment, but omits biological interactions.
- Both front legs have circuit-generated rhythmic motor activity in the regression experiment. Middle and hind legs retain their own original whole-CNS motor activity; there is no copied rhythm, six-leg phase controller or imposed gait.

## Equations and clocks

For each selected cell, `s` is its size divided by the authors' reference median. With `tau=0.020 s`, gain `a=1/s`, threshold `theta=7.5*s`, rate ceiling `R=200 Hz`, and signed anatomical counts `W` scaled by 0.03:

```
target = max(200 * tanh((a / 200) * (0.03 * W @ rates + input - theta)), 0)
d rates / dt = (target - rates) / 0.020
```

The production circuit uses deterministic mean parameters and fixed RK4 integration at 1 ms. No sine wave, prescribed phase, repeated recording or ongoing random excitation generates its rhythm. All circuit rates and its integer clock are checkpointed. The surrounding whole-CNS rate model advances at 10 ms. Muscle activation follows the actual channel means through the existing 40 ms filter.

The hybrid mode uses a 200 Hz ceiling and dopamine drive 0.5, so dopamine neurons remain at 100 Hz. This **changes normalized dopamine release and other normalization** relative to the older 100 Hz model. It is a new experimental run, not seamless continuation of the old trajectory. NPF remains the existing abstract global drive outside the isolated front circuit.

## Stronger motor recruitment

The active intervention selects actual front-circuit neurons annotated `Tr flexor MN`, `Acc. tr flexor MN`, or `Ti extensor MN`: 24 cells in total. For these cells only, the input-to-rate gain is multiplied by 100 and the threshold is set to zero. No extra tonic motor current is added. Their effective equation is:

```
target = max(200 * tanh((100 / (200*s)) * (0.03 * W @ rates)), 0)
d rates / dt = (target - rates) / 0.020
```

The other cells retain their baseline gain, threshold and time constant. Constant DNg100 input is 350 model units, selected because both front circuits maintain rhythms at this setting. Larger descending input or indiscriminate motor stimulation can saturate cells or coactivate antagonists, reducing movement. The selected intervention was chosen by numerical sweeps for larger sustained movement; these excitability values are **assumptions, not measured physiology or a validated pharmacological effect**.

The anatomical synapse counts, 200 Hz ceiling, channel averages, 40 ms muscle filter, torque gains, contact physics, body geometry and renderer are unchanged. Increased movement comes from changed neural rates. There are no per-neuron prescribed phases, synthetic gait signals, angle multipliers or repeating trajectories. `/api/meta` lists every targeted body ID and the intervention parameters. Saved checkpoints contain the complete intervention configuration and reject a mismatched configuration. `--motor-stimulation baseline` retains the original circuit parameters for comparison.

## From neurons to the painted fly

Filtered flexor/extensor activation differences generate torque. Coactivation increases stiffness. A reduced body solver integrates six two-hinge legs, lumped segment masses, body translation, gravity, passive joint springs/damping, compliant floor contact and friction. The browser projects the server's three-dimensional hip, knee and foot coordinates; it never generates movement from elapsed browser time.

Geometry, masses, torque gains, activation-to-force conversion, joint limits and friction are explicitly **uncalibrated engineering parameters**. Body orientation and leg azimuths are constrained. Head, abdomen, wings and coxa have no independent actuation. The painted shell is illustrative; the leg endpoints and body translation follow the reduced solver. This can yield small twitches or a resting pose rather than dramatic writhing.

The solver reports position, velocity and contact/load diagnostics. They are not assigned to sensory neurons. The downloaded annotations do not establish the required receptor tuning, so a closed sensory feedback loop would add an unsupported mapping at this stage.

## Verification

`tests/test_cpg.py` checks sustained raw motor oscillations, no-drive and circuit ablations, numerical refinement and anatomical identity. `tests/test_live_cpg.py` checks that the same rates appear in the primary graph, raw motor output, channel means and downstream network; checkpoint continuation is exact. `tests/test_body.py` checks contact forces, weight support, input-dependent motion, convergence and checkpoint validation. Browser tests check projection and time independence.

The original full-graph test over simulated seconds 2–5 had a largest joint excursion of only 0.00785 degrees, almost invisible at the normal pixel scale. With stronger recruitment, the full-graph test over simulated seconds 4–12 produces roughly 4.7 degrees of joint excursion and 2.5 actor pixels of projected joint movement. Both front legs move; feet mostly remain in contact and body translation stays small. This is visible leg articulation, not whole-body rolling or a validated gait. Raw motor rates and channel values remain available through the read-only API; the minimal viewer shows the room scene.

Removing descending input or silencing either tested core excitatory cell type (`IN17A001` or `INXXX466`) abolishes sustained recruited motor oscillations. Holding the actual muscle activations constant lets the body settle, rather than continuing a built-in movement pattern. Tests also check the stronger circuit against a 0.5 ms reference integration, exact raw-neuron/channel agreement, and exact checkpoint continuation with non-default stimulation.

See `CPG_VERIFICATION.json` for the real-data integration measurements and limitations. A test that motor rates oscillate is evidence about these equations, not evidence of subjective experience or a natural gait.

## Sources and implementation

- [Pugliese et al. preprint record](https://pubmed.ncbi.nlm.nih.gov/42094485/)
- [Authors' code and data](https://github.com/smpuglie/Pugliese_2026)
- [MaleCNS dataset](https://male-cns.janelia.org/download/), CC BY 4.0, MaleCNS collaboration
- [Circuit implementation](flyreward/cpg.py), [hybrid integration](flyreward/cpg_live.py), [mechanical plant](flyreward/body.py)

The NumPy/SciPy solver here is independently written from the declared equations. No JAX runtime or trained movement controller is required.

## Reproduce the real-data integration check

From this package directory, with its live dependencies installed and the MaleCNS data already downloaded, run:

```sh
python scripts/verify_cpg_integration.py --data-dir /path/to/malecns-data --output-dir /path/to/new-verification --seconds 12 --warmup-seconds 4 --motor-stimulation recruited
```

The script runs a local model, writes `CPG_VERIFICATION.json` and `hybridtrace.npz` to the explicitly selected output directory, and checks exact checkpoint continuation. It measures neural, channel, joint and projected-body variability after the warmup. It does not connect to or alter the running backend. Timing and peak memory depend on the host; the source hashes record the tested implementation.
