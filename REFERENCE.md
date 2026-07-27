# API reference

Every public symbol of `src/core/`, the TorchGA-style API all demos are built
on. For a guided tour with examples see [API.md](API.md).

All symbols are re-exported from `src/core/index.ts` (`import { … } from '../../core'`).

### `core/network.ts` — the "model"

```ts
type Activation = 'identity' | 'relu' | 'sigmoid' | 'tanh' | 'softsign';

interface Network {
  topology: readonly number[];        // e.g. [24, 16, 16, 4]
  genomeSize: number;                 // Σ (in + 1) * out, computed
  layers: readonly LayerLayout[];     // per-layer { offset, in, out, activation }
}

defineNetwork(topology, opts?: { hidden?: Activation; output?: Activation }): Network
```

Defines the network and the canonical flat-genome layout shared by the GA, the
WGSL shaders, and the NetworkPanel: **layer-major; within a layer, the weight
matrix is input-major with the bias row last** — `genome[offset + k*out + j]` is
the weight from input `k` to output `j`, `genome[offset + in*out + j]` is the
bias of output `j`. `hidden` defaults to `'relu'`, `output` to `'identity'`.

```ts
forwardCPU(net, genome, inputs): Float64Array
```

CPU forward pass (panel display, debugging, CPU-side replay). Applies each
layer's activation; throws if `inputs.length !== topology[0]`.

```ts
unflattenWeights(net, genome): LayerWeights[]   // TorchGA: model_weights_as_dict
flattenWeights(net, layers): Float64Array       // TorchGA: model_weights_as_vector
// LayerWeights = { weights: Float64Array /* in*out, input-major */, biases: Float64Array }
```

Bidirectional genome↔per-layer-tensors mapping. Both validate shapes and throw
on mismatch.

### `core/population.ts`

```ts
createPopulation(net, count, opts?: {
  seed?: number;          // default 1 (mulberry32)
  initRange?: number;     // default 1
  seedGenome?: Float64Array;
}): Float64Array[]
```

TorchGA's `create_population`: with `seedGenome`, solution 0 is the seed
verbatim and the rest are `seed + U(-range, range)`; without it, every solution
is `U(-range, range)`.

### `core/ga.ts` — GA config-by-name

```ts
type GAConfig =
  | { selection: 'roulette';          // fitness-proportional pick, copy, gaussian mutate
      mutateRate?: number; sigma?: number; eliteCount?: number }            // 0.1, 1, 0
  | { selection: 'tournament';        // best-of-k draws; reset/drift mutation
      tournamentSize?: number; eliteFraction?: number; crossoverRate?: number;
      mutateRate?: number; mutateRange?: number; resetShare?: number;
      driftSigma?: number; clamp?: number }              // 3, 0.02, 0.8, 0.05, 1, 0.5, 0.5, 8
  | { selection: 'layered-crossover'; // roulette parents + per-layer single-point crossover
      eliteCount?: number; mutateRate?: number; sigma?: number; clamp?: number } // 1, 0.05, 0.2, 1
  | { selection: 'remainder-blend' }; // remainder stochastic sampling + BLX/uniform crossover
                                      // + dynamic elite scaling (the track demo's Unity GA)

nextGeneration(population, fitnesses, rng, net, config): Float64Array[]
```

`nextGeneration` is a thin dispatch over `src/utils/ga.ts`; `Evolution` calls it
for you. All operators work on flat `Float64Array` genomes and return a new
population (elites cloned, everything else fresh).

### `core/evolution.ts` — the generation driver

Owns the genome store, the standard GPU buffers, dispatch, readback,
generation-end detection, the GA step, and best-genome tracking. The demo owns
the environment, supplied as a `Simulation`.

#### The GPU contract

| Binding | Buffer | Type | Contents |
|---------|--------|------|----------|
| 0 | `params` | uniform | 16 B by default (`u32[0]` = total agent count); `paramsBytes` + `writeParams` customize |
| 1 | `genomes` | storage, read | `(populationSize + display?) × genomeSize` f32, re-uploaded each generation |
| 2 | `agents` | storage, read_write | `agentCount × agentFloats` f32; the renderer reads this too |
| 3+ | extras | your choice | `extraBuffers`, in declaration order |

One thread per agent, `@workgroup_size(64)`. Agent count is
`populationSize × episodes`, plus one trailing **display agent** (a replay slot
always running the best-so-far genome) when `displayAgent` is on. With
`episodes > 1` the layout follows `src/utils/evaluation.ts`:
`agentIndex = genomeIndex * episodes + episodeIndex`, so per-genome aggregation
is integer division.

#### `Simulation`

Required:

| Field | Meaning |
|---|---|
| `agentFloats: number` | Floats per agent in the state buffer (u32 fields stored as bit patterns). |
| `shader: string` | WGSL compute shader, entry point `main`, honoring the contract above. |
| `initialStates(count, seed, episodes): Float32Array` | Initial contents for `count` agent slots. |
| `isAgentDone(states, agentIndex): boolean` | True when one agent's run has ended. |
| `isGenerationOver(states, evo): boolean` | True when the generation should end (usually: every training agent done). |
| `fitness(states, evo): Float64Array` | Per-**agent** scores, length `trainingAgents`; the driver aggregates episodes. |

Optional hooks, in the order they fire:

| Hook | When / use |
|---|---|
| `paramsBytes?: number` | Params uniform size (default 16). |
| `writeParams?(data, evo)` | Fill the params buffer; called at init and on every `evo.writeParams()`. |
| `extraBuffers?(device, evo): Record<string, GPUBuffer>` | Demo buffers bound at 3+, merged into `evo.buffers`. |
| `beforeStep?(evo, step)` | Shared-CPU-world tick (dino obstacles, flappy pipes) before each substep's dispatch. When present, each substep gets its own encoder/submit; otherwise k substeps share one encoder. |
| `maintain?(evo, states)` | Mid-generation upkeep after readback (pacman restarts its display agent). |
| `rankBest?(states, fitnesses, evo): number` | Genome index to record as the new best, or -1. Default: argmax, recorded when it beats `evo.bestFitness`. |
| `bestMeta?(states, index, fitness, evo)` | Extra meta recorded with a new best (goes to autosave and the HUD). |
| `beforeGaStep?(evo, fitnesses)` | After fitness, before the GA step — adaptive mutation via `evo.setMutationRate()`. |
| `onNewGeneration?(evo)` | After the GA step, before agents reset — env re-roll, seed rotation, `evo.writeParams()`. |
| `trainingSeed?(evo): number` | Seed for training agents' `initialStates` each generation (default 0). |
| `displaySeed?(evo): number` | Seed for the display agent (default: the best genome's index, for deterministic replays). |
| `probe?(states, agentIndex): { inputs?: Float32Array; stats?: [string, string \| number][] }` | Rebuild an agent's network inputs from state; `runDemo` runs `forwardCPU` on them for the NetworkPanel. |

#### `EvolutionConfig` / `EvolutionCallbacks`

```ts
Evolution.init(device, net, sim, {
  populationSize?: number;        // default 300
  seed?: number;                  // default 1
  episodes?: number;              // default 1
  episodeAggregate?: 'mean' | 'min' | 'median';  // default 'mean'
  displayAgent?: boolean;         // default true
  ga: GAConfig;                   // required
  callbacks?: {
    onFitness?(evo, fitnesses): void;                          // after scoring, before the GA
    onNewBest?(evo, genome, fitness, meta): void;              // new recorded best
    onGeneration?(evo): void;                                  // after reset, generation already incremented
  },
}): Evolution
```

#### `Evolution` instance API

State: `generation`, `populationSize`, `episodes`, `trainingAgents`,
`displayAgentIndex`, `buffers` (`{ params, genomes, agents, readback, …extras }`),
`bestGenome`, `bestFitness`, `bestGeneration`, `bestMetaValue`, `lastBestIndex`,
`lastScores`, `lastFitnesses`, `device`, `net`, `rngForHooks`.

| Method | What it does |
|---|---|
| `substeps(k)` | Dispatch k sim steps (one encoder, unless `beforeStep` is set). |
| `readStates(): Promise<Float32Array>` | Copy of the agents buffer; concurrent callers share one in-flight readback. |
| `checkAndEvolve()` | The generation machine: `maintain` → over? → `fitness` → aggregate → `onFitness` → best tracking (`onNewBest`) → `beforeGaStep` → GA → `onNewGeneration` → upload genomes → reset agents → `onGeneration`. Guarded against re-entry. |
| `genomeAt(agentIndex)` | CPU-side genome driving an agent slot (episode-aware). |
| `displayGenome()` | Genome in the display slot (null without a display agent). |
| `injectBest(genome, meta?)` | Load a genome into slot 0 + the display slot and reset. Without `meta.eval`, `bestFitness` becomes `Infinity` so autosave never overwrites a hand-loaded model. |
| `setMutationRate(rate)` | Adaptive-mutation override for the next GA step. |
| `writeParams()` | Re-run `writeParams` and re-upload the params buffer. |
| `countAlive(states)` | Live training agents. |
| `resetAgents()` / `resetAgent(i, seed)` / `resetDisplayAgent()` | Respawn all / one / the display agent from `initialStates`. |
| `restartDisplayIfDead()` / `restartAgentIfDone(i, seed)` | Test-mode replay loops. |
| `resetIfOver()` | Test mode without a display agent: reset everything once the run ends. |

### `core/modelStore.ts` — persistence

Model format is JSON `{ topology, weights, meta? }`; localStorage keys are
`eanns:best:<namespace>` and `eanns:testModel:<namespace>` (namespace `''` maps
to the track demo's legacy global `eanns:testModel`).

| Function | What it does |
|---|---|
| `parseModelText(text, net, legacyParser?)` | Validate a model file against `net` (topology + genome size); throws on invalid. Non-JSON text goes to `legacyParser` (e.g. SnakeAI CSV, Unity genotypes). |
| `downloadModel(namespace, net, genome, meta?)` | Download the genome as `eanns-<namespace>-gen<N>.json`. |
| `autosaveBestModel(namespace, net, genome, fitness, meta?)` | Save to localStorage unless a better `eval` is already stored (a previous session's champion survives a fresh page). Adds `eval: fitness` to `meta`. |
| `loadBestModel(namespace, net)` | The stored `SavedModel`, or null (corrupt/mismatched entries count as absent). |
| `saveTestModel(namespace, net, genome)` / `loadTestModel(namespace, net)` | The Test-mode slot; `loadTestModel` also accepts the legacy bare-weight-array format. |

### `core/runDemo.ts` — the shared bootstrap

`startDemo(descriptor)` = `runDemo(descriptor).catch(showError)`. `runDemo`
does what every demo's `main.ts` used to hand-roll: URL/localStorage settings →
WebGPU init → `Evolution.init` → renderer → HUD chips → NetworkPanel → settings
panel (Mode/Track/Population/Sim speed/FPS + your toggles/sliders + Save/Load
buttons wired to `modelStore`) → the rAF loop with speed control and
Train/Test/Play modes.

#### `DemoDescriptor<R>`

Required:

| Field | Meaning |
|---|---|
| `namespace: string` | Settings/model key prefix. |
| `network: Network` | From `defineNetwork`. |
| `simulation: Simulation` | See above. |
| `ga: GAConfig` | See above. |
| `createRenderer(canvas, gpu, evo): R \| Promise<R>` | Your renderer; `R` must have `render(): void`, may have `setBestIndex(i)`. Receives `evo.buffers`. |

Commonly used:

| Field | Default | Meaning |
|---|---|---|
| `stepsPerSecond` | 60 | Sim steps per real second at speed 1 (accumulator mode). |
| `maxStepsPerFrame` | 600 | Catch-up cap per rAF. |
| `stepMode` | `'accumulator'` | `'per-frame'`: exactly `speed` substeps per rAF (track). |
| `episodes` / `episodeAggregate` | 1 / `'mean'` | Multi-episode evaluation. |
| `displayAgent` | true | Trailing best-genome replay agent. |
| `seed` / `testPopulation` | 1 / 1 | GA seed; population in Test mode. |
| `tracks` | — | Track names for the settings selector (`?track=`). |
| `pickShownAgent(states, evo)` | display agent | Which agent the camera/panel follows. |
| `hud(evo, states, shown, mode)` | — | Top-left stats chip text (`\n`-separated); no chip when omitted. |
| `networkPanel` | — | `{ variant, outputLabels, onToggle }` for the live network diagram. |
| `panelStats(evo, shown, inputs, states, mode)` | probe stats | Rows for the panel's stats table. |
| `toggles` / `sliders` | — | `(evo, renderer) => [...]` extra settings controls. |
| `legacyModelParser` | — | Non-JSON model file formats. |
| `bodyClass` / `hudColor` / `hudChipStyle` | — | Page/HUD styling hooks. |

Escape hatches:

| Field | Meaning |
|---|---|
| `play` | `{ waiting, setup?, onKeydown? }` — human-playable mode (pacman). While `waiting`, substeps pause. |
| `testTick(evo)` | Custom per-frame Test-mode behavior (default: restart the display agent, or `resetIfOver` without one). |
| `autosave` | Set `false` to own autosave yourself via `callbacks.onFitness` (track's per-generation autosave). |
| `modelNamespaces(track)` | Per-track/localized storage keys; `{ best, test }`, `test: ''` = legacy global slot. |
| `actions.onSaveModel(evo)` / `actions.onLoadSavedBest(evo)` | Override the panel's Save/Load buttons. |
| `beforeStart({ gpu, showMessage })` | Runs before settings; return true to take over the page (track's `?selftest`). |
| `afterFrame(renderer, evo, states, shown)` | Per-frame hook after the shown agent is picked (camera follow). |
| `callbacks` | Extra `EvolutionCallbacks`, chained after the built-in autosave. |
