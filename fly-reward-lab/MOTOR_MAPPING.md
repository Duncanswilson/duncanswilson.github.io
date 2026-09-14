# Motor outputs used by the body adapter

The manifest [results/motor/motor_mapping.json](results/motor/motor_mapping.json) identifies **815 motor neurons already present in the 165,122-neuron MaleCNS simulation**. Their individual simulated firing rates can be recorded. **164 neurons** have sufficiently explicit labels for this adapter to assign them to **24 channels**: flexion and extension at the trochanter and tibia joints of all six legs. The other **651 motor neurons remain recorded but unmapped to movement**.

The identity map comes from the downloaded, checksum-verified [MaleCNS v1.0 annotations](https://male-cns.janelia.org/download/). It does not assign movement from dopamine concentration, a random neuron number, or an animation phase. Joint mechanics are a separate hypothesis.

## Identity rules

`build_motor_mapping(data_dir, graph)` in [flyreward/motor_mapping.py](flyreward/motor_mapping.py) returns a JSON-ready dictionary. `motor_neuron_ids` contains all retained motor IDs, sorted by their observed IDs, and can be passed directly to `run_simulation(..., record_neuron_ids=...)`. `motor_graph_indices` gives their positions in the supplied graph. Each `channels` entry contains `name`, `leg`, `joint`, `action`, `body_ids`, `graph_indices`, `motor_types`, `count`, and `available`. The manifest preserves each motor neuron's original annotations and explains every exclusion.

Only these exact type strings assign an action:

| Published neuron type | Joint | Channel action |
|---|---|---|
| `Tr flexor MN`, `Acc. tr flexor MN` | Coxa–trochanter | Flexor |
| `Tr extensor MN` | Coxa–trochanter | Extensor |
| `Ti flexor MN`, `Acc. ti flexor MN` | Femur–tibia | Flexor |
| `Ti extensor MN` | Femur–tibia | Extensor |

Mapped cells must have `status == Traced`, `superclass == vnc_motor`, matching `type`/`mancType`, a consistent instance suffix and `somaSide`, and consistent leg subclass, soma neuromere, and exit nerve. `fl/T1`, `ml/T2`, and `hl/T3` designate front, middle, and hind legs. MANC's published motor atlas explains these categories and assigns muscle targets through anatomical matching and serial homology; accessory and main flexors share a basic joint action. These are anatomical assignments rather than measured force calibrations. [Cheong et al., 2026, Figures 5, 7 and 13](https://elifesciences.org/articles/96084)

**Laterality is an explicit anatomical inference.** These files provide `somaSide`, but no independent exit-nerve-side field. The `instance` suffix is derived from soma side and is only a consistency check. [MaleCNS typing methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/) The adapter uses the documented projection of leg motor neurons to the corresponding ipsilateral leg. [Venkatasubramanian et al., 2019, Figure 1 and results](https://elifesciences.org/articles/42692)

## Coverage of the actual graph

| Leg | Trochanter flexors | Trochanter extensors | Tibia flexors | Tibia extensors |
|---|---:|---:|---:|---:|
| LF: left front | 11 | 2 | 15 | 2 |
| LM: left middle | 9 | 2 | 11 | 2 |
| LH: left hind | 9 | 1 | 17 | 2 |
| RF: right front | 10 | 2 | 14 | 2 |
| RM: right middle | 9 | 2 | 11 | 2 |
| RH: right hind | 9 | 2 | 16 | 2 |

For example, left-front tibia extensors are the observed MaleCNS IDs **800636** and **815344**. Right-front tibia extensors are **804257** and **815678**. Every other ID is listed in the manifest, with its graph index and source row.

The full recording inventory includes 708 VNC motor neurons and 107 central-brain motor neurons. By the source's subclass labels it includes 381 leg, 214 abdominal, 67 wing, 44 neck and 16 haltere motor neurons, plus other head or unclassified motor categories. A cell being unmapped by this adapter does not imply that its target is unknown to biology: this adapter intentionally supports only two leg joints. Wing, neck, abdomen, tarsal, and coxal-rotation activity receives no invented action.

## What the mapping does not establish

The source files contain no measured force curves, muscle sizes, moment arms, mechanical loads, or complete proprioceptive loop. A motor atlas can establish targets using genetic lines and peripheral imaging, but those measurements do not automatically calibrate this simulation. [Azevedo et al., 2024](https://www.nature.com/articles/s41586-024-07389-x) Physiological measurements also show substantial differences in force per spike among tibia motor units. [Azevedo et al., 2020](https://elifesciences.org/articles/56754)

Consequently, averaging flexor and extensor rates, applying an activation time constant, and converting their difference into a joint angle are declared mechanical approximations. A constant motor command should settle to a fixed pose rather than generate a scripted gait. The mapping module itself generates no oscillation, force, angle, or movement trajectory.

Validation: 23 focused mapping tests cover exact labels, conflicting side/segment/nerve/type annotations, missing labels, unknown motor types, graph ID ordering, graph exclusions, duplicates, and rejection of synthetic graphs at the public data entry point. The actual manifest includes all 815 annotated motor IDs with zero motor IDs excluded from the supplied full graph.
