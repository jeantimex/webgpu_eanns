import type { Simulation } from '../../core';
import { A, AGENT_FLOATS, GRID, initialAgentStates } from './snake_buffers';
import { snakeShader } from './snake.wgsl';

function bodyBit(states: Float32Array, agentIndex: number, cell: number): boolean {
  const u32 = new Uint32Array(states.buffer);
  const o = agentIndex * AGENT_FLOATS;
  return ((u32[o + A.bodyMask + (cell >>> 5)] >>> (cell & 31)) & 1) === 1;
}

/** SnakeAI's 24-ray vision: food/body/1-over-distance in each of 8 directions. */
function visionFor(states: Float32Array, agentIndex: number): Float32Array {
  const o = agentIndex * AGENT_FLOATS;
  const hx = states[o + A.headX];
  const hy = states[o + A.headY];
  const ax = states[o + A.appleX];
  const ay = states[o + A.appleY];
  const dirs = [
    [-1, 0], [-1, -1], [0, -1], [1, -1],
    [1, 0], [1, 1], [0, 1], [-1, 1],
  ] as const;
  const vision = new Float32Array(24);
  for (let d = 0; d < dirs.length; d++) {
    let px = hx;
    let py = hy;
    let dist = 0;
    let food = 0;
    let body = 0;
    while (true) {
      px += dirs[d][0];
      py += dirs[d][1];
      dist++;
      if (px < 0 || px >= GRID || py < 0 || py >= GRID) break;
      if (!food && px === ax && py === ay) food = 1;
      if (!body && bodyBit(states, agentIndex, py * GRID + px)) body = 1;
    }
    vision[d * 3] = food;
    vision[d * 3 + 1] = body;
    vision[d * 3 + 2] = 1 / dist;
  }
  return vision;
}

/** Snake environment: each agent plays one game inside the compute shader. */
export const snakeSim: Simulation = {
  agentFloats: AGENT_FLOATS,
  shader: snakeShader,
  initialStates: (count, seed) => initialAgentStates(count, seed),

  isAgentDone: (states, i) => states[i * AGENT_FLOATS + A.gameOver] > 0.5,

  // Generation ends when every snake is dead, the display replay included.
  isGenerationOver(states, evo) {
    for (let i = 0; i < evo.trainingAgents; i++) {
      if (!this.isAgentDone(states, i)) return false;
    }
    return this.isAgentDone(states, evo.displayAgentIndex);
  },

  // SnakeAI fitness: lifetime^2 x 2^score, with a 2^10 bonus per score past 9
  // (score = length; lifetime = moves survived).
  fitness(states, evo) {
    const out = new Float64Array(evo.trainingAgents);
    for (let i = 0; i < evo.trainingAgents; i++) {
      const o = i * AGENT_FLOATS;
      const score = states[o + A.score];
      const lifetimeSq = states[o + A.moves] ** 2;
      out[i] = score < 10 ? lifetimeSq * 2 ** score : lifetimeSq * 1024 * (score - 9);
    }
    return out;
  },

  bestMeta: (states, bestIndex) => ({ score: states[bestIndex * AGENT_FLOATS + A.score] }),

  probe: (states, agentIndex) => ({ inputs: visionFor(states, agentIndex) }),
};
