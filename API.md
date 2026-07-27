# Using the API

Guide to building demos on `src/core/`. For the full symbol-by-symbol
documentation see [REFERENCE.md](REFERENCE.md); the design rationale is in
[API-plan.md](API-plan.md).

Every demo is built on `src/core/` (a TorchGA-style API: the genome is one flat
vector per solution, fitness is fully demo-owned, GA behavior is config-by-name).
A demo provides three things — a **network**, a **simulation** (one WGSL compute
shader + state layout + fitness reducer), and a **renderer** — and `runDemo`
supplies everything else.

### Quickstart: a whole demo in one file

```ts
// src/demos/mygame/main.ts
import '../../style.css';
import { defineNetwork, startDemo, type Simulation } from '../../core';
import { mygameShader } from './mygame.wgsl';   // your sim: one thread per agent
import { MyRenderer } from './my_renderer';

// 1. The network. genomeSize is computed — never hand-count weights again.
const network = defineNetwork([8, 8, 2], { hidden: 'relu', output: 'tanh' });

// 2. The simulation: state layout, generation rule, fitness. That's all the
//    driver needs — it owns buffers, dispatch, readback, the GA, and autosave.
const AGENT_FLOATS = 4; // e.g. x, y, alive(u32 bits), score

const simulation: Simulation = {
  agentFloats: AGENT_FLOATS,
  shader: mygameShader,
  initialStates: (count) => {
    const states = new Float32Array(count * AGENT_FLOATS);
    for (let i = 0; i < count; i++) new Uint32Array(states.buffer)[i * AGENT_FLOATS + 2] = 1;
    return states;
  },
  isAgentDone: (states, i) => new Uint32Array(states.buffer)[i * AGENT_FLOATS + 2] !== 1,
  isGenerationOver(states, evo) {
    for (let i = 0; i < evo.trainingAgents; i++) if (!this.isAgentDone(states, i)) return false;
    return true;
  },
  fitness(states, evo) {
    const out = new Float64Array(evo.trainingAgents);
    for (let i = 0; i < evo.trainingAgents; i++) out[i] = states[i * AGENT_FLOATS + 3];
    return out;
  },
};

// 3. The descriptor. Settings panel, HUD, network panel, model save/load,
//    Train/Test modes, and the rAF loop are all wired up by runDemo.
startDemo({
  namespace: 'mygame',           // localStorage keys: eanns:best:mygame, …
  network,
  simulation,
  ga: { selection: 'tournament', mutateRate: 0.05 },
  stepsPerSecond: 60,
  createRenderer: (canvas, gpu, evo) => new MyRenderer(canvas, gpu, evo.buffers),
});
```

Add a `mygame.html` entry (copy `dino.html`, point its script at
`src/demos/mygame/main.ts`) and register it in `vite.config.ts` — done.

The shader side of the contract: the driver binds `params` (uniform) at 0,
`genomes` (storage read) at 1, `agents` (storage read_write) at 2, and your
`extraBuffers` at 3+; one thread per agent at `@workgroup_size(64)`; each agent's
genome starts at `agentGenomeIndex * genomeSize` floats.

### Headless / manual path

For tests or custom loops, the pieces compose directly (like TorchGA's
"create population → hand to GA → run → extract best"):

```ts
import { createPopulation, defineNetwork, Evolution, saveTestModel } from '../../core';

const net = defineNetwork([8, 8, 2], { hidden: 'relu', output: 'tanh' });
const seedGenome = createPopulation(net, 1, { seed: 42 })[0]; // TorchGA: TorchGA(model, 1)

const evo = Evolution.init(device, net, simulation, {
  populationSize: 300,
  seed: 42,
  ga: { selection: 'layered-crossover', eliteCount: 1 },
});

for (let gen = 1; gen <= 250; gen++) {
  while (evo.generation === gen) {       // step until the generation flips
    evo.substeps(600);
    await evo.checkAndEvolve();
  }
}
saveTestModel('mygame', net, evo.bestGenome!); // pygad: best_solution()
```

### Adding a demo, end to end

1. `src/demos/<name>/<name>.wgsl.ts` — sim shader honoring the binding contract,
   one thread per agent, writing fitness-relevant fields into agent state.
2. `<name>_buffers.ts` — the `A` state-layout map, `AGENT_FLOATS`, `initial*States`.
3. `<name>_net.ts` — `defineNetwork(...)`.
4. `<name>_sim.ts` — the `Simulation` (`isAgentDone`, `isGenerationOver`,
   `fitness`, plus hooks as needed).
5. `<name>_renderer.ts` — reads the same GPU buffers.
6. `main.ts` — one `startDemo({...})` call; `<name>.html` entry + `vite.config.ts` input.
