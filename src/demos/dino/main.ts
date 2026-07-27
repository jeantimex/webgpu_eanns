import '../../style.css';
import { startDemo } from '../../core';
import { DINO_FLOATS } from './dino_buffers';
import { dinoNetwork } from './dino_net';
import { dinoSim, dinoWorld } from './dino_sim';
import { DinoRenderer } from './dino_renderer';

/** Highest-score dino (the original highlights the leader). */
function pickShown(states: Float32Array): number {
  const u32 = new Uint32Array(states.buffer);
  let best = 0;
  for (let i = 1; i < states.length / DINO_FLOATS; i++) {
    if (u32[i * DINO_FLOATS + 3] > u32[best * DINO_FLOATS + 3]) best = i;
  }
  return best;
}

startDemo({
  namespace: 'dino',
  network: dinoNetwork,
  simulation: dinoSim,
  // The source repo's geneticAlgorithm.js: roulette pick, copy, gaussian mutation.
  ga: { selection: 'roulette' },
  displayAgent: false,
  stepsPerSecond: 60, // the original's frame rate
  maxStepsPerFrame: 240,
  bodyClass: 'snake-layout',

  createRenderer: async (canvas, gpu, evo) => {
    const renderer = await DinoRenderer.create(canvas, gpu, evo.buffers);
    return {
      render: () => renderer.render(dinoWorld.obstacle, dinoWorld.groundScroll, dinoWorld.runFrame),
      setBestIndex: (i) => renderer.setBestIndex(i),
    };
  },

  pickShownAgent: (states) => pickShown(states),

  hud: (evo, states, shown) => {
    const o = shown * DINO_FLOATS;
    const u32 = new Uint32Array(states.buffer);
    return (
      `Score:   ${Math.floor(u32[o + 3] / 7)}\n` + // the original's units: one point per ~7 frames
      `Cleared: ${dinoWorld.cleared}\n` +
      `Alive:   ${evo.countAlive(states)}\n` +
      `Fitness: ${states[o + 4].toFixed(1)}`
    );
  },

  networkPanel: {
    variant: 'snake',
    outputLabels: ['JUMP'],
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  },

  panelStats: (evo, shown, _inputs, states) => {
    const o = shown * DINO_FLOATS;
    const u32 = new Uint32Array(states.buffer);
    return [
      ['GEN', evo.generation],
      ['SCORE', Math.floor(u32[o + 3] / 7)],
      ['CLEARED', dinoWorld.cleared],
      ['POP LEFT', evo.countAlive(states)],
      ['FITNESS', states[o + 4].toFixed(1)],
      ['SPEED', dinoWorld.gamespeed.toFixed(1)],
    ];
  },
});
