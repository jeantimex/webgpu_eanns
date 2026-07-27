import '../../style.css';
import { startDemo } from '../../core';
import { BIRD_FLOATS } from './flappy_buffers';
import { flappyNetwork } from './flappy_net';
import { flappySim, flappyWorld } from './flappy_sim';
import { FlappyRenderer } from './flappy_renderer';

/** Highest-score bird. */
function pickShown(states: Float32Array): number {
  const u32 = new Uint32Array(states.buffer);
  let best = 0;
  for (let i = 1; i < states.length / BIRD_FLOATS; i++) {
    if (u32[i * BIRD_FLOATS + 4] > u32[best * BIRD_FLOATS + 4]) best = i;
  }
  return best;
}

startDemo({
  namespace: 'flappy',
  network: flappyNetwork,
  simulation: flappySim,
  // The source repo's geneticAlgorithm.js: roulette pick, copy, gaussian mutation.
  ga: { selection: 'roulette' },
  displayAgent: false,
  stepsPerSecond: 60, // the original's frame rate
  maxStepsPerFrame: 240,
  bodyClass: 'snake-layout',

  createRenderer: async (canvas, gpu, evo) => {
    const renderer = await FlappyRenderer.create(canvas, gpu, evo.buffers);
    return {
      render: () => renderer.render(flappyWorld.pipesList.length),
      setBestIndex: (i) => renderer.setBestIndex(i),
    };
  },

  pickShownAgent: (states) => pickShown(states),

  hud: (evo, states, shown) => {
    const o = shown * BIRD_FLOATS;
    return (
      `Pipes:   ${flappyWorld.pipesPassed}\n` +
      `Alive:   ${evo.countAlive(states)}\n` +
      `VelY:    ${states[o + 2].toFixed(3)}\n` +
      `Fitness: ${states[o + 5].toFixed(1)}`
    );
  },

  networkPanel: {
    variant: 'snake',
    outputLabels: ['FLAP'],
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  },

  panelStats: (evo, shown, _inputs, states) => {
    const o = shown * BIRD_FLOATS;
    const u32 = new Uint32Array(states.buffer);
    return [
      ['GEN', evo.generation],
      ['PIPES', flappyWorld.pipesPassed],
      ['POP LEFT', evo.countAlive(states)],
      ['FITNESS', states[o + 5].toFixed(1)],
      ['SCORE', u32[o + 4]],
      ['VEL Y', states[o + 2].toFixed(2)],
    ];
  },
});
