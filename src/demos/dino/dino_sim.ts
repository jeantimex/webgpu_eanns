import type { Simulation } from '../../core';
import { mulberry32, type Rng } from '../../utils/rng';
import {
  BASE_SPEED,
  DINO_FLOATS,
  initialDinoStates,
  MAX_SPEED,
  WORLD_W,
  type ObstacleState,
} from './dino_buffers';
import { dinoShader } from './dino.wgsl';

/** Small cactus: 34x70 at y=230, variants at sheet x=446/548. Big: 49x100 at y=201, x=652/802. */
function makeObstacle(small: boolean, rng: Rng): ObstacleState {
  const multi = Math.floor(rng() * 3) + 1;
  const baseW = small ? 34 : 49;
  return {
    x: 0,
    y: small ? 230 : 201,
    w: baseW * multi,
    h: small ? 70 : 100,
    picX: small ? 446 + Math.floor(rng() * 2) * 102 : 652 + Math.floor(rng() * 2) * 150,
    scroll: small ? -100 : -200,
    baseW,
    small,
    multi,
  };
}

/**
 * The world (one alternating obstacle, speed ramp, ground scroll, run animation)
 * is shared across dinos, so it ticks on the CPU in beforeStep; dino physics and
 * the NN forward passes run in the compute shader.
 */
const rng = mulberry32(1);
export const dinoWorld = {
  obstacle: makeObstacle(true, rng),
  /** Obstacles cleared this generation (same for every dino: shared world). */
  cleared: 0,
  /** Original-style score ticks (one per ~7 frames); drives the speed ramp. */
  tickScore: 0,
  gamespeed: BASE_SPEED,
  groundScroll: 0,
  /** Sprite sheet x of the current run frame (1514/1602); jump frame is 1338. */
  runFrame: 1514,
  scoreInterval: 0,
  frameInterval: 0,
};

/** gameover() in the original: respawn the world (dino states are the driver's). */
function resetWorld(): void {
  dinoWorld.obstacle = makeObstacle(true, rng);
  dinoWorld.cleared = 0;
  dinoWorld.tickScore = 0;
  dinoWorld.gamespeed = BASE_SPEED;
  dinoWorld.scoreInterval = 0;
}

const paramsData = new ArrayBuffer(16);

export const dinoSim: Simulation = {
  agentFloats: DINO_FLOATS,
  shader: dinoShader,
  initialStates: (count) => initialDinoStates(count),

  extraBuffers: (device) => ({
    // One active obstacle: x, y, w, h (f32) = 16 bytes.
    obstacle: device.createBuffer({ label: 'dino obstacle', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
  }),

  beforeStep(evo) {
    // Speed ramp: 7 + score/100, capped at 17 (score ticks every ~7 frames).
    dinoWorld.scoreInterval++;
    if (dinoWorld.scoreInterval > 6) {
      dinoWorld.tickScore++;
      dinoWorld.scoreInterval = 0;
    }
    dinoWorld.gamespeed = Math.min(MAX_SPEED, BASE_SPEED + dinoWorld.tickScore / 100);

    // Run animation toggles every 5 frames.
    dinoWorld.frameInterval++;
    if (dinoWorld.frameInterval > 5) {
      dinoWorld.runFrame = dinoWorld.runFrame === 1514 ? 1602 : 1514;
      dinoWorld.frameInterval = 0;
    }

    // Obstacle advance + recycle, alternating small/big like the original.
    const obs = dinoWorld.obstacle;
    obs.scroll += dinoWorld.gamespeed;
    dinoWorld.groundScroll += dinoWorld.gamespeed;
    if (obs.scroll > WORLD_W + obs.w * 3) {
      dinoWorld.cleared++;
      dinoWorld.obstacle = makeObstacle(!obs.small, rng);
    }
    dinoWorld.obstacle.x = WORLD_W - dinoWorld.obstacle.scroll;

    const cur = dinoWorld.obstacle;
    evo.device.queue.writeBuffer(evo.buffers.obstacle, 0, new Float32Array([cur.x, cur.y, cur.w, cur.h]));
    new Uint32Array(paramsData)[0] = evo.populationSize;
    new Float32Array(paramsData)[1] = dinoWorld.gamespeed;
    evo.device.queue.writeBuffer(evo.buffers.params, 0, paramsData);
  },

  isAgentDone: (states, i) => new Uint32Array(states.buffer)[i * DINO_FLOATS + 2] !== 1,

  isGenerationOver(states, evo) {
    for (let i = 0; i < evo.trainingAgents; i++) {
      if (!this.isAgentDone(states, i)) return false;
    }
    return true;
  },

  fitness(states, evo) {
    const out = new Float64Array(evo.trainingAgents);
    for (let i = 0; i < evo.trainingAgents; i++) out[i] = states[i * DINO_FLOATS + 4];
    return out;
  },

  onNewGeneration: () => resetWorld(),
};
