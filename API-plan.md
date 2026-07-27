# WebGPU EANN — Public API Plan

Goal: give the project a TorchGA-style API layer so demos are built by *configuring*
a shared evolution engine instead of hand-rolling `*_evolution.ts` / `*_buffers.ts` /
`main.ts` each time. A new demo should be: one WGSL sim shader + one descriptor object.

This document only defines the APIs and the migration plan. No implementation yet.

---

## 1. What we mirror from TorchGA / PyGAD

TorchGA's whole design is four idioms (`torchga.py`, 91 lines):

| TorchGA idiom | What we adopt |
|---|---|
| Genome = one flat vector per solution; population = list of vectors | Already true here (`Float64Array[]`, layer-major, input-major, bias-row-last). We make it a *documented, computed* contract instead of hand-counted `*_GENOME_SIZE` constants. |
| Bidirectional genome↔model mapping (`model_weights_as_vector` / `model_weights_as_dict`) | `flattenWeights` / `unflattenWeights` + a per-layer offset table derived from the topology. |
| Population init: `(model, num_solutions)` → solution 0 pristine, rest perturbed | `createPopulation(network, count, opts)`. |
| Fitness is fully user-owned; library never knows what the network does | Stays true. The difference: TorchGA evaluates one solution per CPU call; we evaluate the *whole population per GPU dispatch*, so fitness becomes a batched reducer over read-back agent states. |
| Config-by-kwargs GA with string-enum operators (`parent_selection_type="tournament"`) | `GAConfig` with `selection: 'roulette' \| 'tournament' \| 'layered-crossover'` etc., delegating to the existing `utils/ga.ts` functions. |
| Lifecycle callbacks receiving the GA instance (`on_generation`, `on_fitness`, …) | Same names, camelCase: `onGeneration`, `onFitness`, `onNewBest`, `onReset`. |

What we deliberately do **not** mirror:

- PyGAD's blocking `run()` loop. Games need an interactive rAF loop with speed
  control and rendering between steps, so our driver is step-based
  (`substeps(k)` + `checkAndEvolve()`), matching the existing demos.
- `pygad.load/save` pickle files. We keep the existing JSON `{topology, weights, meta}`
  model format, consolidated into one module.

---

## 2. Current duplication (why the API looks like this)

From the codebase audit, the near-identical code across the 5 demos:

- Generation driver: `substeps / readStates / checkAndEvolve`, readback dedup,
  `isEvolving` guard, f64→f32 genome flatten+upload — 5 copies.
- Population init (uniform ±1) — 5 copies.
- `genomeSize(topology)` math — hand-computed per demo (`740`, `82`, `65`, `57`, `47`).
- Model persistence (`model.ts`: JSON format, localStorage autosave, test slot) —
  3 near-identical copies (snake, pacman, track); flappy/dino have none.
- `main.ts` bootstrap: settings → WebGPU init → evolution → renderer → HUD →
  NetworkPanel → settings panel → rAF loop with fixed-Hz accumulator — 5 copies.
- HUD overlay, `showMessage/showError` — 5 copies.
- Dense NN forward pass — inlined 5× in WGSL (+ 2 CPU copies).

What is genuinely per-demo and stays per-demo: the WGSL simulation shader
(environment physics + perception + action), the agent-state layout, the fitness
formula, the renderer, and (for dino/flappy) a shared CPU world tick.

The API seam is therefore:

```
demo = Simulation (shader + state layout + fitness)  ⊕  GAConfig  ⊕  Renderer
core = network + population + evolution driver + model store + runDemo bootstrap
```

---

## 3. Proposed API

New directory: `src/core/`. Existing `src/utils/ga.ts`, `src/utils/evaluation.ts`,
`src/webgpu/utils.ts`, `src/ui/*` stay and are reused, not rewritten.

### 3.1 `core/network.ts` — network definition (the "model")

```ts
type Activation = 'identity' | 'relu' | 'sigmoid' | 'tanh' | 'softsign';

interface Network {
  topology: readonly number[];     // e.g. [24, 16, 16, 4]
  activations: readonly Activation[]; // one per layer after input
  genomeSize: number;              // computed: Σ (in+1)*out
  layerOffsets: readonly { weights: number; biases: number; in: number; out: number }[];
}

function defineNetwork(
  topology: readonly number[],
  opts?: { hidden?: Activation; output?: Activation } // defaults relu / identity
): Network;
```

Genome layout is the existing convention, now the single source of truth:
layer-major, input-major weight matrix, bias row last per layer.

Mapping helpers (TorchGA's `model_weights_as_vector` / `_as_dict`):

```ts
function flattenWeights(layers: { weights: Float64Array; biases: Float64Array }[]): Float64Array;
function unflattenWeights(net: Network, genome: Float64Array): { weights: Float64Array; biases: Float64Array }[];
function forwardCPU(net: Network, genome: Float64Array, inputs: ArrayLike<number>): Float64Array;
```

`forwardCPU` replaces the hand-written CPU passes (snake's `decisionFor`, track's
`network.ts forward`) and feeds `NetworkPanel.draw()`.

### 3.2 `core/population.ts` — population creation (TorchGA's `TorchGA`)

```ts
function createPopulation(
  net: Network,
  count: number,
  opts?: { seed?: number; initRange?: number; seedGenome?: Float64Array }
): Float64Array[];
// If seedGenome given: solution 0 = seedGenome verbatim, rest = seedGenome + U(-r, r)
// (exactly TorchGA's create_population). Otherwise all = U(-r, r), r default 1.
```

### 3.3 `core/ga.ts` — GA config-by-kwargs over `utils/ga.ts`

```ts
type GAConfig =
  | { selection: 'roulette';    mutateRate?: number; sigma?: number; eliteCount?: number }
  | { selection: 'tournament';  tournamentSize?: number; eliteFraction?: number;
      crossoverRate?: number; mutateRate?: number; clamp?: number; /* …existing opts */ }
  | { selection: 'layered-crossover'; eliteCount?: number; mutateRate?: number;
      sigma?: number; clamp?: number };   // per-layer single-point, snake-style

function nextGeneration(
  population: Float64Array[],
  fitnesses: Float64Array,
  rng: Rng,
  net: Network,
  config: GAConfig
): Float64Array[];   // thin dispatch over utils/ga.ts
```

Track's custom GA (remainder stochastic sampling + BLX) slots in as a fourth
variant, `selection: 'remainder-blend'` (implemented: `nextRemainderBlendGeneration`
in utils/ga.ts).

### 3.4 `core/evolution.ts` — the shared generation driver

**Status: implemented** (all 5 demos migrated). The sketch below is the design
contract; the shipped `Simulation` grew a few hooks during migration —
`isAgentDone`, `paramsBytes`/`writeParams` (pacman's 48-byte params),
`maintain` (mid-generation display restart), `beforeGaStep` (adaptive mutation),
`onNewGeneration` (env re-roll/seed rotation), `trainingSeed`/`displaySeed`,
`rankBest`/`bestMeta` (pacman's level-first best ranking), and `probe`.
`Evolution.init` is synchronous, and the driver also exposes `resetAgent` /
`restartAgentIfDone` / `resetIfOver` / `countAlive` / `rngForHooks`.

The demo supplies a `Simulation`; the driver owns everything else.

```ts
interface Simulation {
  /** Agent-state stride and initial contents. */
  agentFloats: number;
  initialStates(count: number, seed: number): Float32Array;

  /** WGSL compute shader. Receives genomes + agents at bindings 0/1 (fixed contract),
      plus demo-owned buffers appended after. One thread per agent. */
  shader: string;
  /** Demo-owned extra buffers/bindings (maze, pipes, walls, …) and their layout. */
  createExtraBuffers?(device: GPUDevice, ctx: { populationSize: number; episodes: number }): GPUBindingResource[];
  /** Per-tick hook for shared-CPU worlds (dino obstacles, flappy pipes):
      tick world state and queue uploads before this step's dispatch. */
  beforeStep?(queue: GPUQueue, step: number): void;

  /** Generation lifecycle. */
  isGenerationOver(states: Float32Array): boolean;     // usually "all agents dead"
  fitness(states: Float32Array, ctx: FitnessContext): Float64Array; // batched reducer
  resetExtra?(rng: Rng): void;                          // pacman env re-roll, maze state

  /** Optional per-agent probe for the NetworkPanel/HUD: reconstruct the input
      vector from read-back state (e.g. snake's 24-ray vision). The driver runs
      `forwardCPU` on it to get outputs; stats go to the panel's stats table. */
  probe?(states: Float32Array, agentIndex: number):
    { inputs?: Float32Array; stats?: [string, string | number][] };
}

interface EvolutionConfig {
  populationSize?: number;          // default 300
  seed?: number;                    // default 1
  episodes?: number;                // default 1; uses utils/evaluation.ts layout
  ga: GAConfig;
  displayAgent?: boolean;           // trailing replay agent (default true)
  callbacks?: {
    onGeneration?(evo: Evolution): void | 'stop';
    onFitness?(evo: Evolution, fitnesses: Float64Array): void;
    onNewBest?(evo: Evolution, genome: Float64Array, fitness: number): void;
    onReset?(evo: Evolution): void;
  };
}

class Evolution {
  static init(device: GPUDevice, net: Network, sim: Simulation, cfg: EvolutionConfig): Promise<Evolution>;

  readonly net: Network;
  readonly buffers: { params: GPUBuffer; genomes: GPUBuffer; agents: GPUBuffer; readback: GPUBuffer };
  readonly populationSize: number;
  readonly generation: number;
  readonly bestGenome: Float64Array | null;
  readonly bestFitness: number;

  genomeAt(i: number): Float64Array;
  injectBest(genome: Float64Array, meta?: unknown): void;   // load-into-display-slot
  displayGenome(): Float64Array | null;

  substeps(k: number): void;              // k dispatches in one encoder
  readStates(): Promise<Float32Array>;    // deduped readback (existing logic)
  checkAndEvolve(): Promise<void>;        // isGenerationOver → fitness → GA → upload
  setMutationRate?(rate: number): void;   // adaptive-mutation knob (pacman)
}
```

Notes:

- The genome-buffer contract (binding 0 = params uniform, 1 = genomes, 2 = agents,
  extras after) is the one all 5 shaders already use; it becomes official.
- Multi-episode scoring: when `episodes > 1`, the driver applies the
  `agentIndex = genomeIndex * episodes + episodeIndex` layout and
  `aggregateEpisodes()` from `utils/evaluation.ts`; the demo's `fitness()` returns
  per-*agent* scores and the driver reduces to per-*genome*.
- Adaptive mutation (pacman's stagnation logic) is expressed via `setMutationRate`
  from `onGeneration`, keeping the policy demo-owned.

### 3.5 `core/modelStore.ts` — persistence (consolidates 3× `model.ts`)

```ts
interface SavedModel { topology: number[]; weights: number[]; meta?: Record<string, unknown> }

function saveBestModel(namespace: string, net: Network, genome: Float64Array, meta?: object): void;
function loadBestModel(namespace: string): SavedModel | null;
function saveTestModel(namespace: string, net: Network, genome: Float64Array, meta?: object): void;
function loadTestModel(namespace: string): SavedModel | null;
function parseModelFile(text: string): SavedModel;   // JSON, plus legacy CSV/genotype parsers
```

localStorage keys unified as `eanns:best:<namespace>` / `eanns:testModel:<namespace>`
(track's current un-namespaced keys get migrated). Wiring `onNewBest` →
`saveBestModel` in `runDemo` gives flappy/dino autosave for free.

### 3.6 `core/runDemo.ts` — the shared `main.ts` bootstrap

**Status: implemented.** Beyond the sketch, the shipped descriptor also has:
`stepMode: 'per-frame'` (track steps `speed` substeps per rAF, no accumulator),
`play` (pacman's keyboard mode), `testTick` (pacman replays agent 0),
`modelNamespaces` (track keys best models per track name; `''` = the legacy
global `eanns:testModel` slot), `autosave: false` + per-generation autosave via
`callbacks.onFitness` (track), `actions` overrides (track's Save/Load buttons),
`beforeStart` (track's `?selftest`), `afterFrame` (track's camera follow),
`hudColor`/`hudChipStyle` (pacman/track HUD styling), and function-form
`toggles`/`sliders` (they receive the evolution and renderer).

```ts
interface DemoDescriptor {
  namespace: string;                                   // 'snake' — settings/model keys
  network: Network;
  simulation: Simulation;
  ga: GAConfig;
  episodes?: number;
  createRenderer(canvas: HTMLCanvasElement, gpu: WebGPUState, evo: Evolution):
    Promise<{ render(dt: number): void; setBestIndex?(i: number): void }>;
  networkPanel?: { variant?: 'compact' | 'snake'; inputLabels?: string[]; outputLabels?: string[] };
  panel?: { toggles?: ToggleDef[]; sliders?: SliderDef[] };   // passed to createDemoSettingsPanel
  hud?: (evo: Evolution) => string;                    // one-line status → shared HUD element
  stepsPerSecond?: number;                             // fixed-Hz accumulator rate
}

function runDemo(descriptor: DemoDescriptor): Promise<void>;
```

`runDemo` does what every `main.ts` does today: `currentSettings()` →
`initializeWebGPU` → `Evolution.init` → renderer → `NetworkPanel` →
`createDemoSettingsPanel` (with Save/Load wired to `modelStore`) → HUD → rAF loop
with speed control → `showError` on failure. Each demo's `main.ts` shrinks to:

```ts
runDemo({ namespace: 'snake', network, simulation, ga, createRenderer, ... });
```

### 3.7 (Optional, later) `core/wgslForward.ts` — generated WGSL forward pass

The dense forward pass is inlined 5× in WGSL with only topology/activation
differences. If wanted, `wgslForward(net)` can codegen the
`fn nn_forward(genomes_ptr, inputs) -> outputs` snippet from the `Network`
definition, interpolated into demo shaders the same way state-layout constants
already are. **Deferred**: it touches every shader and the per-demo output handling
(argmax vs sampled-softmax vs sigmoid) differs. Migrate first, dedup after the API
proves stable.

---

## 4. What a new demo looks like (extension contract)

1. Write `mygame.wgsl.ts`: sim shader honoring the buffer contract
   (params/genomes/agents at bindings 0–2), one thread per agent, writes a
   fitness-relevant field into agent state.
2. Define state layout constants (the `A = { … }` map), `AGENT_FLOATS`,
   `initialStates`.
3. Write `simulation`: `fitness()` reducer + `isGenerationOver()` (+ optional
   `beforeStep` / `createExtraBuffers`).
4. Write the renderer (unchanged — reads the same GPU buffers).
5. `main.ts` = one `runDemo({...})` call.

No evolution loop, no GA code, no readback logic, no persistence code, no
bootstrap glue. Adding a demo = shader + descriptor + renderer.

---

## 5. Migration plan

**Status: complete.** All phases landed; `npm run build` passes after each.
Deviations from the phase notes: track's GA was ported (variant
`remainder-blend`), flappy/dino gained model save/load + Test mode via the
shared bootstrap, and pacman's old bare-array test models still load
(`loadTestModel` accepts both formats).

Migrate one demo at a time, keeping each demo playable after its step. Suggested
order — simplest shape first, oddballs last:

**Phase 0 — core primitives (no demo changes yet)**
- `core/network.ts` (`defineNetwork`, `genomeSize`, `flatten/unflatten`, `forwardCPU`)
- `core/population.ts` (`createPopulation`)
- `core/ga.ts` (dispatch over existing `utils/ga.ts`)
- `core/modelStore.ts` (consolidate snake/pacman/track `model.ts` formats; keep the
  legacy CSV/Unity genotype parsers)
- Verify: `tsc` build passes; existing demos untouched.

**Phase 1 — Snake** (single episode, per-thread world, simplest GA-consumer)
- Build `core/evolution.ts` against snake: extract `SnakeEvolution`'s
  substeps/readback/checkAndEvolve/upload logic into the driver, snake keeps only
  shader + layout + fitness (`lifetime² × 2^score`) + GA choice.
- Replace snake's `model.ts` and `decisionFor` with `modelStore` + `forwardCPU`.
- Build `core/runDemo.ts` against snake's `main.ts`.
- Verify: snake page trains, autosaves, network panel renders identically.

**Phase 2 — Dino & Flappy** (adds the `beforeStep` shared-CPU-world hook)
- Express pipe/obstacle ticking + upload as `beforeStep`; both were already the
  thinnest evolution classes.
- They gain model save/load and namespaced settings for free (currently `notWired`
  stubs).
- Verify: training curves and speed control behave as before.

**Phase 3 — Pac-Man** (the feature-complete one: episodes, adaptive mutation,
env re-roll, play/test modes)
- `episodes: 6` via driver; per-generation env re-roll via `resetExtra`; adaptive
  mutation via `setMutationRate` in `onGeneration`; play/test mode knobs stay as
  thin demo methods calling into the evolution's params buffer.
- Verify: 6-episode aggregate fitness, stagnation-triggered rate bump, saved-model
  playback.

**Phase 4 — Track** (the odd one out)
- Decide explicitly: port its custom GA (`ga.ts` remainder-sampling + BLX) in as a
  fourth `GAConfig.selection` variant, or switch it to an existing operator. Default:
  port it — zero behavior change.
- Migrate `network.ts forward` → `forwardCPU`, localStorage keys → namespaced
  (`eanns:best:track:<trackName>`), keep `?selftest` working (it uses
  `simulatePopulationGpu`, which stays a track-owned utility).
- Verify: selftest passes, per-track saved models still load (one-time key
  migration), HUD/checkpoint behavior unchanged.

**Phase 5 — cleanup**
- Delete the extracted `*_evolution.ts`, `model.ts` copies, and per-demo
  bootstrap code; delete legacy lil-gui `setupControls` if nothing references it.
- Re-tally `*_GENOME_SIZE` constants — all should now come from `defineNetwork`.
- (Optional) evaluate `core/wgslForward.ts` codegen.

Each phase ends with `npm run build` + a manual smoke of that demo's page; there is
no test suite in the project today, and this plan doesn't add one — the
`?selftest` mechanism and per-demo visual verification are the existing convention.

---

## 6. Explicitly out of scope

- Moving the GA to GPU (compute-shader selection/crossover). CPU GA over 300×~1k
  genomes per generation is not a bottleneck; revisit only if profiling says so.
- A generic `fitness_func(solution, idx)` per-solution CPU callback — incompatible
  with batched GPU evaluation and not needed by any current demo.
- PyGAD-style blocking `run()`, `plot_fitness`, pickle persistence.
- Backwards-compat shims for the old per-demo classes; demos are migrated, not
  dual-maintained.

---

## 7. Example: Snake built on the API

What `src/demos/snake/` shrinks to after migration. Three pieces: the state layout
and initial states (unchanged from today), a `Simulation`, and a one-call `main.ts`.
The WGSL shader (`snake.wgsl.ts`) is untouched — it already honors the
params/genomes/agents binding contract.

### `snake_sim.ts` — the Simulation (the only new demo code)

```ts
import { A, AGENT_FLOATS, GRID, initialAgentStates } from './snake_buffers';
import { snakeShader } from './snake.wgsl';
import type { Simulation } from '../../core/evolution';

export const snakeSim: Simulation = {
  agentFloats: AGENT_FLOATS,
  initialStates: initialAgentStates,
  shader: snakeShader,

  // Generation ends when every snake is dead.
  isGenerationOver(states) {
    for (let i = 0; i < states.length; i += AGENT_FLOATS) {
      if (states[i + A.gameOver] < 0.5) return false;
    }
    return true;
  },

  // SnakeAI fitness: lifetime^2 x 2^score, 2^10 bonus per score past 9.
  fitness(states, { populationSize }) {
    const out = new Float64Array(populationSize);
    for (let i = 0; i < populationSize; i++) {
      const o = i * AGENT_FLOATS;
      const score = states[o + A.score];
      const lifetimeSq = states[o + A.moves] ** 2;
      out[i] = score < 10 ? lifetimeSq * 2 ** score : lifetimeSq * 1024 * (score - 9);
    }
    return out;
  },

  // Rebuild the 24-ray vision vector so runDemo can light up the NetworkPanel
  // (replaces SnakeEvolution.visionFor; the driver runs forwardCPU for outputs).
  probe(states, agentIndex) {
    return { inputs: visionFor(states, agentIndex) }; // same 8-dir x 3 raycast as today
  },
};
```

### `main.ts` — the whole bootstrap

```ts
import { defineNetwork, runDemo } from '../../core';
import { initialAgentStates, A, AGENT_FLOATS } from './snake_buffers';
import { snakeSim } from './snake_sim';
import { SnakeRenderer } from './snake_renderer';

const network = defineNetwork([24, 16, 16, 4], { hidden: 'relu', output: 'relu' });
// network.genomeSize === 740 — replaces the hand-counted SNAKE_GENOME_SIZE

runDemo({
  namespace: 'snake',           // model keys: eanns:best:snake / eanns:testModel:snake
  network,
  simulation: snakeSim,
  ga: { selection: 'layered-crossover', eliteCount: 1, mutateRate: 0.05, sigma: 0.2, clamp: 1 },
  stepsPerSecond: 100,          // SnakeAI's 100 moves/sec, x sim-speed slider

  createRenderer: (canvas, gpu, evo) => new SnakeRenderer(canvas, gpu, evo.buffers),

  networkPanel: { variant: 'snake', outputLabels: ['UP', 'DOWN', 'LEFT', 'RIGHT'] },

  hud: (evo, probe) =>
    `Apples: ${probe.inputs ? applesOf(probe) : 0}  Alive: ${evo.aliveCount}  Gen: ${evo.generation}`,

  callbacks: {
    onNewBest: (evo, genome, fitness) => {
      // autosave is already wired by runDemo via modelStore; this is for extras
      console.log(`gen ${evo.generation}: new best ${fitness}`);
    },
  },
});
```

That's it — no evolution loop, no readback, no GA call, no panel/settings/bootstrap
glue. `SnakeEvolution` (306 lines), `main.ts` (148 lines), and `model.ts` collapse
into the descriptor above plus the demo-owned files that remain: `snake_buffers.ts`
(layout + initial states), `snake.wgsl.ts`, `snake_renderer.ts`.

### Manual path (TorchGA-style, no `runDemo`)

For notebooks/tests/headless training, the pieces compose directly, mirroring
TorchGA's "create population → hand to GA → run → extract best":

```ts
import { defineNetwork, createPopulation, Evolution, nextGeneration, saveBestModel } from '../../core';

const net = defineNetwork([24, 16, 16, 4], { hidden: 'relu' });
const device = (await navigator.gpu.requestAdapter())!.requestDevice(...);

// TorchGA: torch_ga = TorchGA(model, num_solutions=10)
const seedGenome = createPopulation(net, 1, { seed: 1 })[0];

const evo = await Evolution.init(await device, net, snakeSim, {
  populationSize: 300,
  seed: 1,
  ga: { selection: 'layered-crossover' },
});

// pygad: ga_instance.run()  →  here: step the world until generations roll over
for (let gen = 0; gen < 250; gen++) {
  while (evo.generation === gen + 1 - 1) {  // step until the generation flips
    evo.substeps(600);
    await evo.checkAndEvolve();
  }
}

// pygad: solution, fitness, idx = ga_instance.best_solution()
saveBestModel('snake', net, evo.bestGenome!, { fitness: evo.bestFitness });
```
