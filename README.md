# WebGPU EANNs

Browser port of [Applying_EANNs](https://github.com/ArztSamuel/Applying_EANNs): 2D top-down
cars that learn to drive a track using a feedforward neural network trained by a genetic
algorithm — no gradient descent, no PyTorch, no training backend. The physics simulation and
neural-network forward passes for the whole population run in a **WebGPU compute shader**;
the genetic algorithm itself (selection, mutation over 30×47 floats) runs on the CPU.

## Run it

```sh
npm install
npm run dev
```

Open the printed URL in a current Chrome/Edge (WebGPU required). `npm run build` type-checks
and builds to `dist/`; `npm run preview` serves the build.

## How it works

### The learning loop

Each car is driven by a small neural network, topology `[5, 4, 3, 2]` with the SoftSign
activation `x / (1 + |x|)` on every layer:

- **Inputs**: distances from 5 forward raycast sensors (±45°, ±20.6°, 0° relative to the
  car's heading), in raw track units (max 10).
- **Outputs**: turn and engine, clamped to ±1.

A genome is the network's 47 weights laid out flat. One generation:

1. Every car in the population drives simultaneously until it dies (wall hit, or 7 s without
   reaching a checkpoint).
2. Fitness = fraction of the track completed (checkpoints give distance-proportional partial
   credit).
3. The CPU genetic algorithm builds the next generation, reproducing the Unity original's
   *effective* behavior: the best 2 genomes pass through unmodified, the rest of the
   population is filled with clones of parents drawn at random from the top 3, then each
   clone is mutated per-parameter with probability 0.3 by a uniform amount in ±2.0.

> Note: Unity's crossover operator is accidentally a no-op (`Random.Next() < 0.6` compares a
> non-negative `int` against `0.6`), which is what makes the original converge so fast — it
> is effectively a mutation-only evolution strategy. This port reproduces that deliberately;
> an earlier version that "fixed" the crossover converged dramatically slower.

### WebGPU architecture

The simulation is **GPU-resident**: car states live in GPU storage buffers for the entire
generation, the compute pass advances them in place, and the render pass reads the *same*
buffers directly — there is no per-frame CPU↔GPU roundtrip for simulation data.

```
            ┌──────────────────────── GPU ────────────────────────┐
            │                                                     │
 genomes ──►│  compute pass (sim.wgsl): one thread per car        │
            │    sensors → NN forward → physics → fitness         │
            │       │ writes in place                             │
            │       ▼                                             │
 car state ─│─► render pass: reads the same car/sensor buffers    │
  buffers   │    in the vertex shader (instanced quads/lines)     │
            │                                                     │
            └─────────────────────────────────────────────────────┘
                          │ copyBufferToBuffer (once per frame)
                          ▼
            CPU: readback → generation over? → GA → upload genomes
```

#### Buffers (`src/demos/track/buffers.ts`)

All sim data is in one bind group (`src/demos/track/sim.wgsl.ts`):

| Binding | Buffer | Type | Contents |
|---------|--------|------|----------|
| 0 | `params` | uniform | carCount, wallCount, cpCount (u32), dt (f32) — 16 B |
| 1 | `genomes` | storage, read | population × 47 f32, uploaded once per generation |
| 2 | `cars` | storage, read_write | population × 48 B car states (also read by the renderer) |
| 3 | `walls` | storage, read | wall segments, 4 f32 each (x1, y1, x2, y2) |
| 4 | `checkpoints` | storage, read | 32 B per checkpoint: position, distToPrev, reward, accReward |
| 5 | `sensors` | storage, read_write | 5 raw sensor distances per car, for the renderer's crosses |

Plus a `readback` buffer (`MAP_READ | COPY_DST`), target of the per-frame state copy.

`CarState` is 12 f32 (48 B stride), mirrored exactly in WGSL:
`pos(2), angleDeg, vel, alive(u32), cpIndex(u32), timeSinceCp, fitness, outputs(2), pad(2)`.

#### Compute pass (`src/demos/track/sim.wgsl.ts`, driven by `src/demos/track/evolution.ts`)

- `@workgroup_size(64)`, one invocation per car, `ceil(population / 64)` workgroups.
- Each dispatch = one physics substep at a fixed `dt = 1/50` s (Unity's FixedUpdate rate):
  raycast 5 sensors against every wall segment (nearest hit), run the SoftSign forward pass
  reading the car's genome, integrate the kinematic car model, then handle checkpoint capture
  (fitness credit), the 7 s checkpoint timeout, and wall death.
- `Evolution.substeps(k)` records k dispatches in one command buffer per rendered frame —
  `k` is the GUI's "Sim speed", so training runs up to 64× realtime without touching the CPU.
- **Generation-end detection**: once per frame the cars buffer is copied to the readback
  buffer and `mapAsync`'d; concurrent consumers (`isGenerationOver`, `readFitness`,
  `readBestCarState`) share that single in-flight readback. When every car's `alive` flag is
  0, `evolve()` runs the CPU GA, uploads the new genomes with `queue.writeBuffer`, and
  rewrites the initial car states — the next frame starts the new generation.

#### Render pass (`src/demos/track/renderer.ts`)

A single instanced pipeline draws everything in 3 draw calls:

1. **Walls** — one elongated quad per wall strip plus two round joint discs per segment
   (8-triangle fans) so corners join smoothly. The render wall list is preprocessed by
   `buildRenderWalls` (`src/demos/track/track.ts`): the two collider edges of each extracted Unity
   wall strip are paired back into a centerline, and near-coincident endpoints are snapped
   together.
2. **Cars** — instanced 1×2 quads (the Unity `Car.prefab` size) rotated by the car-state
   angle read straight from the GPU buffer; red, with the current best car green (a camera
   uniform carries `bestIndex`). Dead cars emit degenerate triangles.
3. **Sensor crosses** — 5 per alive car, positioned in the vertex shader from car pose +
   the GPU sensor-distance buffer.

The camera is an orthographic uniform (center + half-size) that lerp-follows the best car
like Unity's `CameraMovement.cs`; wheel zooms, dragging pans (and auto-disables follow).

#### CPU side

- `src/demos/track/network.ts` — forward pass, genome↔weights mapping in Unity's exact order.
- `src/demos/track/ga.ts` — seeded RNG (mulberry32), population init (uniform ±1), `nextGeneration`.
- `src/demos/track/model.ts` — model save/load, `localStorage` autosave, Unity genotype import.
- `src/demos/track/car.ts` — CPU reference implementation of the car physics, used by the selftest
  as a parity oracle for the WGSL sim (and nothing else — training is GPU-only).

## Simulation fidelity

Constants matched to the Unity original (`CarMovement.cs`, `Sensor.cs`, `Checkpoint.cs`,
`GeneticAlgorithm.cs`):

| Parameter | Value |
|---|---|
| Physics timestep | 1/50 s |
| Max velocity / acceleration / friction | 20 / 8 / 2 (friction only at zero throttle) |
| Turn speed | 100 °/s |
| Sensors | 5 rays at ±45°, ±20.6° (atan(2.7/7.2)), 0°; max 10, min 0.01, raw units |
| Checkpoint capture radius / timeout | 3 units / 7 s |
| Car size | 1×2 units (wall death at 0.5 from center; point approximation of Unity's box collider) |
| Genome init | uniform ±1 |
| Mutation | per-parameter p=0.3, additive uniform ±2.0, all but best 2 |

Known deliberate simplification: the car is a point with a 0.5 half-width for wall
collision, not a full box collider (marked with a `ponytail:` comment in `src/demos/track/car.ts`).

## Controls (GUI panel, top-right)

- **mode**: Train (population evolves) / Test (one loaded model reruns the track, GA off).
- **track**: track selector (`public/tracks/<name>.json`).
- **population**: cars per generation (applies on reload; buffers are sized at init).
- **Sim speed**: physics substeps per rendered frame, 1–64.
- **Follow best car**: camera lerp-follows the green car; dragging the canvas pans and
  unchecks this automatically; the mouse wheel zooms.
- **Save best model**: downloads the best car's genome as JSON.
- **Load model file**: loads a model and switches to Test mode with it.
- **Load saved best**: reseeds the population with the autosaved best genotype.

HUD: top-left shows the alive population count and the best car's Turn/Engine/Eval;
bottom-left the generation counter; bottom-right a live diagram of the best car's network
(edge green/red = positive/negative weight, width ∝ |weight|), like the Unity original.

The best genotype per track is autosaved to `localStorage` after every generation
(`eanns:best:<trackName>`), so training survives reloads.

## Model format

```json
{ "topology": [5, 4, 3, 2], "weights": [/* 47 floats */],
  "meta": { "track": "Track1", "generation": 42, "eval": 0.97 } }
```

Weights are in the Unity genome order (layer-major, row-major, bias row last per layer).
Unity's native genotype files (`p0;p1;…;p46` plain text, from `Genotype.SaveToFile`) load
directly via **Load model file** — you can train in Unity and test in the browser, or vice
versa.

## Tracks

Tracks are pure data. The real Unity Track1–4 geometry was extracted from the binary scene
files and ships in `public/tracks/`; `practice.json` is a hand-authored circuit.

To add a track:

1. Add `public/tracks/mytrack.json`:
   `{ "name", "start": {x, y, angleDeg}, "checkpoints": [[x,y],…], "walls": [[x1,y1,x2,y2],…] }`.
   Checkpoints are ordered racing-line points (index 0 = start line); walls are segments.
   Coordinates are Unity convention: x right, y up, angle CCW-positive with 0 = +Y.
2. Add `"mytrack"` to `TRACKS` in `src/demos/track/main.ts`.

To export tracks from the Unity project yourself, drop `tools/TrackExporter.cs` into
`Applying_EANNs/UnityProject/Assets/Editor/`, then use menu **Tools/EANNs/Export Open Track
Scene** (or **Export All Track Scenes**). JSON lands in `<UnityProject>/ExportedTracks/`.

## Project layout

```
src/
  demos/
    track/     the car demo, self-contained: main.ts, network.ts (forward pass),
               ga.ts (genetic algorithm), model.ts (persistence), selftest.ts
               (parity/unit asserts), sim.wgsl.ts (compute shader), buffers.ts,
               evolution.ts (generation driver: dispatch, readback, GA hand-off),
               track.ts (track data + checkpoint math), car.ts (CPU reference
               physics), renderer.ts (instanced render pass, pan/zoom), hud.ts
    flappy/    the Flappy Bird demo, same shape: main.ts, flappy.wgsl.ts,
               flappy_buffers.ts, flappy_evolution.ts, flappy_renderer.ts
  gui/       controls_gui.ts (demo-agnostic lil-gui panel, URL/localStorage settings)
  ui/        networkPanel.ts (2D-canvas network diagram, topology passed in)
  webgpu/    utils.ts (device/context init, buffer helper, DPI-aware resize)
  utils/     dom.ts, rng.ts (seeded RNG)
public/tracks/         track1–4.json (extracted from Unity), practice.json
public/assets/flappy/  sprites from the source repo
tools/                 TrackExporter.cs (Unity editor script)
```

A demo directory holds everything specific to that training project (sim shader, GA,
persistence, renderer); adding a new project means adding a new `src/demos/<name>/`
directory plus an HTML entry — the shared code is only `gui/`, `ui/`, `webgpu/`, `utils/`.

## Selftest

Open the app URL with `?selftest` appended: runs hand-computed NN asserts, GA determinism
checks, a straight-line car check, model-parser round-trips, and a GPU-vs-CPU 500-step
parity comparison (max |Δfitness| ≤ 1e-4). Result is shown on screen and in the console.
