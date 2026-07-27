import type { Simulation } from '../../core';
import { mulberry32 } from '../../utils/rng';
import {
  BIRD_FLOATS,
  BIRD_X,
  initialBirdStates,
  MAX_PIPES,
  PIPE_GAP,
  PIPE_SPAWN_FRAMES,
  PIPE_SPEED,
  PIPE_TOP_MAX,
  PIPE_TOP_MIN,
  PIPE_W,
  uploadFlappyPipes,
  WORLD_W,
  type PipeState,
} from './flappy_buffers';
import { flappyShader } from './flappy.wgsl';

/**
 * Pipes are shared across birds, so they tick on the CPU in beforeStep (one pipe
 * state upload per frame); bird physics and the NN forward passes run in the
 * compute shader.
 */
const rng = mulberry32(1);
export const flappyWorld = {
  pipesList: [] as PipeState[],
  /** Bars passed this generation (same for every bird: they never move horizontally). */
  pipesPassed: 0,
  frameCounter: 0,
};

function spawnPipe(): void {
  const topY = PIPE_TOP_MIN + rng() * (PIPE_TOP_MAX - PIPE_TOP_MIN);
  flappyWorld.pipesList.push({ x: WORLD_W, topY, bottomY: topY + PIPE_GAP, width: PIPE_W });
}

const paramsData = new ArrayBuffer(16);

export const flappySim: Simulation = {
  agentFloats: BIRD_FLOATS,
  shader: flappyShader,
  initialStates: (count) => initialBirdStates(count),

  extraBuffers: (device) => ({
    // Pipe struct: x, topY, bottomY, width (f32) = 16 bytes per pipe.
    pipes: device.createBuffer({ label: 'flappy pipes', size: MAX_PIPES * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
  }),

  beforeStep(evo) {
    const world = flappyWorld;
    for (let p = world.pipesList.length - 1; p >= 0; p--) {
      const wasAhead = world.pipesList[p].x + world.pipesList[p].width >= BIRD_X;
      world.pipesList[p].x -= PIPE_SPEED;
      if (wasAhead && world.pipesList[p].x + world.pipesList[p].width < BIRD_X) world.pipesPassed++;
      if (world.pipesList[p].x + world.pipesList[p].width < 0) world.pipesList.splice(p, 1);
    }
    if (world.frameCounter % PIPE_SPAWN_FRAMES === 0) spawnPipe();
    world.frameCounter++;
    if (world.pipesList.length > MAX_PIPES) world.pipesList.shift();
    uploadFlappyPipes(evo.device, evo.buffers.pipes, world.pipesList);

    new Uint32Array(paramsData)[0] = evo.populationSize;
    new Uint32Array(paramsData)[1] = world.pipesList.length;
    evo.device.queue.writeBuffer(evo.buffers.params, 0, paramsData);
  },

  isAgentDone: (states, i) => new Uint32Array(states.buffer)[i * BIRD_FLOATS + 3] !== 1,

  isGenerationOver(states, evo) {
    for (let i = 0; i < evo.trainingAgents; i++) {
      if (!this.isAgentDone(states, i)) return false;
    }
    return true;
  },

  fitness(states, evo) {
    const out = new Float64Array(evo.trainingAgents);
    for (let i = 0; i < evo.trainingAgents; i++) out[i] = states[i * BIRD_FLOATS + 5];
    return out;
  },

  /** resetGame in the original: clear the pipes (bird states are the driver's). */
  onNewGeneration() {
    flappyWorld.frameCounter = 0;
    flappyWorld.pipesList = [];
    flappyWorld.pipesPassed = 0;
  },
};
