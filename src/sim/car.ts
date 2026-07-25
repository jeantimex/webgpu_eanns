import { forward } from '../ai/network';
import { buildCheckpointTable, wallsFlat, CAPTURE_RADIUS, type CheckpointTable, type Track } from './track';

/** Unity fixed timestep: 50 Hz. */
export const DT = 1 / 50;

// CarMovement.cs constants, verbatim.
const MAX_VEL = 20;
const ACCELERATION = 8;
const VEL_FRICT = 2; // applied only when throttle == 0
const TURN_SPEED = 180; // degrees/sec

const MAX_CHECKPOINT_DELAY = 7; // seconds without a checkpoint before death

const SENSOR_MAX_DIST = 25;
const SENSOR_MIN_DIST = 0.01;
const BRAKE_ACCEL = 24;
// Sensor angles relative to forward and car-local origins (x=right, y=forward),
// from Car.prefab: sensors at local (±0.3,0.27),(±0.3,0.42),(0,0.42) under car scale
// (1,2) → y doubled. Order matches Unity's hierarchy (GetComponentsInChildren):
// right side first → [−45°, −atan(2.7/7.2)≈−20.56°, 0°, +20.56°, +45°] in our
// CCW-positive convention — Unity-trained genomes depend on the pairing.
const SENSOR_HALF_ANGLE = Math.atan(2.7 / 7.2);
export const SENSOR_ANGLES = [-Math.PI / 4, -SENSOR_HALF_ANGLE, 0, SENSOR_HALF_ANGLE, Math.PI / 4];
export const SENSOR_ORIGINS = [
  [-0.3, 0.54],
  [-0.3, 0.84],
  [0, 0.84],
  [0.3, 0.84],
  [0.3, 0.54],
];

// ponytail: car approximated as a point that dies within half-width of a wall;
// Unity used a 1x2 box collider (Car.prefab: 1x1 collider, transform scale (1,2)).
export const CAR_HALF_WIDTH = 0.5;

const DEG2RAD = Math.PI / 180;
// State is carried in f32 to mirror the WGSL compute shader (GPU==CPU selftest).
const f32 = Math.fround;

export interface CarState {
  x: number;
  y: number;
  /** CCW-positive degrees; forward = (-sin, cos). */
  angleDeg: number;
  vel: number;
  alive: boolean;
  /** Index of the next checkpoint to capture; starts at 1 ([0] is the start line). */
  cpIndex: number;
  timeSinceCp: number;
  fitness: number;
  /** Last NN outputs: [turn, engine]. */
  outputs: [number, number];
}

export function initCarState(track: Track): CarState {
  return {
    x: track.start.x,
    y: track.start.y,
    angleDeg: track.start.angleDeg,
    vel: 0,
    alive: true,
    cpIndex: 1,
    timeSinceCp: 0,
    fitness: 0,
    outputs: [0, 0],
  };
}

/** Ray (origin, dir, t >= 0) vs segment [a, b]; returns t or -1. Mirrors sim.wgsl. */
function raySegment(ox: number, oy: number, dx: number, dy: number, ax: number, ay: number, bx: number, by: number): number {
  const sx = bx - ax;
  const sy = by - ay;
  const denom = dx * sy - dy * sx;
  if (Math.abs(denom) < 1e-9) return -1;
  const apx = ax - ox;
  const apy = ay - oy;
  const t = (apx * sy - apy * sx) / denom;
  const u = (apx * dy - apy * dx) / denom;
  return t >= 0 && u >= 0 && u <= 1 ? t : -1;
}

function pointSegmentDist(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const abx = bx - ax;
  const aby = by - ay;
  const t = Math.max(0, Math.min(1, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)));
  return Math.hypot(px - (ax + abx * t), py - (ay + aby * t));
}

/** 5 raw sensor distances (not normalized, like Unity's Sensor.Output). */
export function sense(walls: Float32Array, x: number, y: number, angleDeg: number): number[] {
  const rad = angleDeg * DEG2RAD;
  const fx = -Math.sin(rad);
  const fy = Math.cos(rad);
  const rx = Math.cos(rad);
  const ry = Math.sin(rad);
  const out = new Array<number>(5);
  for (let s = 0; s < 5; s++) {
    const ca = Math.cos(SENSOR_ANGLES[s]);
    const sa = Math.sin(SENSOR_ANGLES[s]);
    const dx = fx * ca - fy * sa;
    const dy = fx * sa + fy * ca;
    // Raycast from the sensor's front-mounted origin, not the car center (Sensor.cs:51).
    const ox = x + rx * SENSOR_ORIGINS[s][0] + fx * SENSOR_ORIGINS[s][1];
    const oy = y + ry * SENSOR_ORIGINS[s][0] + fy * SENSOR_ORIGINS[s][1];
    let best = SENSOR_MAX_DIST;
    for (let w = 0; w < walls.length; w += 4) {
      const t = raySegment(ox, oy, dx, dy, walls[w], walls[w + 1], walls[w + 2], walls[w + 3]);
      if (t >= 0 && t < best) best = t;
    }
    out[s] = f32(Math.max(best, SENSOR_MIN_DIST));
  }
  return out;
}

/**
 * One deterministic simulation step of a single car. Mirrors CarMovement.FixedUpdate
 * (input -> velocity -> friction), CarController death rules and
 * TrackManager.GetCompletePerc for fitness. `outputs` = [turn, engine].
 */
export function stepCar(
  state: CarState,
  outputs: [number, number],
  walls: Float32Array,
  table: CheckpointTable,
  cps: [number, number][],
  dt: number = DT,
): void {
  if (!state.alive) return;
  const turn = f32(Math.max(-1, Math.min(1, outputs[0])));
  const engine = f32(Math.max(-1, Math.min(1, outputs[1])));
  state.outputs = [turn, engine];

  // ApplyInput: engine & braking force.
  if (engine < 0) {
    if (state.vel > 0) {
      // Active braking while moving forward
      state.vel = f32(Math.max(0, state.vel + f32(engine * BRAKE_ACCEL * dt)));
    } else if (state.vel > engine * MAX_VEL) {
      state.vel = f32(Math.max(-MAX_VEL, state.vel + f32(engine * ACCELERATION * dt)));
    }
  } else if (engine > 0) {
    if (state.vel < 0) {
      // Active braking while moving backward
      state.vel = f32(Math.min(0, state.vel + f32(engine * BRAKE_ACCEL * dt)));
    } else if (state.vel < engine * MAX_VEL) {
      state.vel = f32(Math.min(MAX_VEL, state.vel + f32(engine * ACCELERATION * dt)));
    }
  }
  // ApplyInput: rotation (negative sign matches CarMovement's AngleAxis(-input * ...)).
  state.angleDeg = f32(state.angleDeg + f32(-turn * TURN_SPEED * dt));

  // ApplyVelocity.
  const rad = state.angleDeg * DEG2RAD;
  state.x = f32(state.x + f32(-Math.sin(rad) * state.vel * dt));
  state.y = f32(state.y + f32(Math.cos(rad) * state.vel * dt));

  // ApplyFriction: only when throttle == 0.
  if (engine === 0) {
    if (state.vel > 0) state.vel = f32(Math.max(0, state.vel - VEL_FRICT * dt));
    else if (state.vel < 0) state.vel = f32(Math.min(0, state.vel + VEL_FRICT * dt));
  }

  // Wall death.
  for (let w = 0; w < walls.length; w += 4) {
    if (pointSegmentDist(state.x, state.y, walls[w], walls[w + 1], walls[w + 2], walls[w + 3]) < CAR_HALF_WIDTH) {
      state.alive = false;
      break;
    }
  }

  // Checkpoint timeout & off-track grace period with grass friction.
  const currentIdx = state.cpIndex;
  if (currentIdx > 0 && currentIdx < cps.length) {
    const cpPrev = cps[currentIdx - 1];
    const cpCurr = cps[currentIdx];
    const corrDist = pointSegmentDist(state.x, state.y, cpPrev[0], cpPrev[1], cpCurr[0], cpCurr[1]);
    if (corrDist > 5.5) {
      state.vel = f32(state.vel * 0.95);
      state.timeSinceCp = f32(state.timeSinceCp + dt * 3);
      if (corrDist > 12) {
        state.alive = false;
      }
    }
  }
  state.timeSinceCp = f32(state.timeSinceCp + dt);
  if (state.timeSinceCp > MAX_CHECKPOINT_DELAY) state.alive = false;

  if (!state.alive) {
    state.vel = 0;
    // Strip partial credit for uncaptured segment on crash so suicidal wall hits are penalized!
    const idx = state.cpIndex;
    if (idx > 0 && idx <= cps.length) {
      state.fitness = table.accReward[idx - 1];
    } else {
      state.fitness = 0;
    }
    return;
  }

  // Checkpoint capture + fitness (TrackManager.GetCompletePerc).
  let idx = state.cpIndex;
  if (idx < cps.length) {
    let dist = Math.hypot(state.x - cps[idx][0], state.y - cps[idx][1]);
    let captured = false;
    while (dist <= CAPTURE_RADIUS) {
      idx++;
      captured = true;
      if (idx >= cps.length) break;
      dist = Math.hypot(state.x - cps[idx][0], state.y - cps[idx][1]);
    }
    if (captured) state.timeSinceCp = 0;
    state.cpIndex = idx;
    if (idx >= cps.length) {
      state.fitness = 1;
    } else {
      const complete = Math.max(0, (table.distToPrev[idx] - dist) / table.distToPrev[idx]);
      state.fitness = f32(table.accReward[idx - 1] + complete * table.reward[idx]);
    }
  } else {
    state.fitness = 1;
  }
}

/** CPU reference simulation of a whole population; oracle for the GPU==CPU selftest. */
export function simulatePopulation(genomes: Float64Array[], track: Track, steps: number): number[] {
  const walls = wallsFlat(track);
  const table = buildCheckpointTable(track);
  const states = genomes.map(() => initCarState(track));
  for (let step = 0; step < steps; step++) {
    for (let i = 0; i < genomes.length; i++) {
      if (!states[i].alive) continue;
      const sensors = sense(walls, states[i].x, states[i].y, states[i].angleDeg);
      const out = forward(genomes[i], sensors);
      stepCar(states[i], [f32(out[0]), f32(out[1])], walls, table, track.checkpoints);
    }
  }
  return states.map((s) => s.fitness);
}
