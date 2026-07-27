import '../../style.css';
import { startDemo } from '../../core';
import { A, AGENT_FLOATS } from './snake_buffers';
import { snakeNetwork } from './snake_net';
import { snakeSim } from './snake_sim';
import { parseSnakeAiCsv } from './model';
import { SnakeRenderer } from './snake_renderer';

startDemo({
  namespace: 'snake',
  network: snakeNetwork,
  simulation: snakeSim,
  // CodeBullet's SnakeAI GA: roulette selection + per-layer single-point crossover.
  ga: { selection: 'layered-crossover', eliteCount: 1, mutateRate: 0.05, sigma: 0.2, clamp: 1 },
  stepsPerSecond: 100, // SnakeAI runs at 100 moves/sec, multiplied by sim speed
  bodyClass: 'snake-layout',
  hudChipStyle: true, // LCD-colored chips, readable over the light board
  legacyModelParser: parseSnakeAiCsv,

  createRenderer: (canvas, gpu, evo) => new SnakeRenderer(canvas, gpu, evo.buffers),

  hud: (evo, states, shown) => {
    const o = shown * AGENT_FLOATS;
    return (
      `Apples:  ${states[o + A.apples]}\n` +
      `Length:  ${states[o + A.length]}\n` +
      `Moves:   ${states[o + A.moves]}\n` +
      `Life:    ${states[o + A.sinceEat]}\n` +
      `Alive:   ${evo.countAlive(states)}`
    );
  },

  networkPanel: {
    variant: 'snake',
    outputLabels: ['UP', 'DOWN', 'LEFT', 'RIGHT'],
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  },

  panelStats: (evo, shown, _inputs, states) => {
    const o = shown * AGENT_FLOATS;
    return [
      ['GEN', evo.generation],
      ['BEST FITNESS', Math.floor(Number.isFinite(evo.bestFitness) ? evo.bestFitness : 0)],
      ['POP LEFT', evo.countAlive(states)],
      ['MOVES LEFT', Math.max(0, Math.floor(states[o + A.sinceEat]))],
      ['MUTATION RATE', '0.05'],
      ['SCORE', Math.floor(states[o + A.score])],
      ['BEST SCORE', Math.floor(evo.bestMetaValue.score ?? 0)],
      ['BEST GEN', evo.bestGeneration],
    ];
  },
});
