/**
 * GPU mirror of src/sim/car.ts — same constants, same math, one thread per car.
 * Any change here must be mirrored in car.ts (the selftest compares the two).
 */
export const simShader = /* wgsl */ `
struct SimParams {
  carCount: u32,
  wallCount: u32,
  cpCount: u32,
  dt: f32,
};

// 48-byte stride (12 f32): see CAR_FLOATS in src/gpu/buffers.ts.
struct CarState {
  pos: vec2f,        // offset 0
  angle: f32,        // offset 8, degrees CCW-positive, forward = (-sin, cos)
  vel: f32,          // offset 12
  alive: u32,        // offset 16
  cpIndex: u32,      // offset 20, next checkpoint to capture
  timeSinceCp: f32,  // offset 24
  fitness: f32,      // offset 28
  outputs: vec2f,    // offset 32, [turn, engine]
  pad: vec2f,        // offset 40
};

// 32-byte stride (8 f32): x, y, distToPrev, reward, accReward, 3x pad.
// (Storage-space array stride is roundUp(align, size), so the padding is required
// to match the 8-float packing in checkpointsFlat.)
struct Checkpoint {
  pos: vec2f,
  distToPrev: f32,
  reward: f32,
  accReward: f32,
  pad0: f32,
  pad1: vec2f,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> genomes: array<f32>;
@group(0) @binding(2) var<storage, read_write> cars: array<CarState>;
@group(0) @binding(3) var<storage, read> walls: array<vec4f>;
@group(0) @binding(4) var<storage, read> checkpoints: array<Checkpoint>;
// 5 raw sensor distances per car, for the renderer's sensor-hit crosses.
@group(0) @binding(5) var<storage, read_write> sensors: array<f32>;

// CarMovement.cs constants, verbatim.
const MAX_VEL = 20.0;
const ACCELERATION = 8.0;
const VEL_FRICT = 2.0;
const TURN_SPEED = 180.0;
const MAX_CHECKPOINT_DELAY = 7.0;

const SENSOR_MAX_DIST = 25.0;
const SENSOR_MIN_DIST = 0.01;
const BRAKE_ACCEL = 24.0;
// Sensor angles relative to forward and car-local origins (x=right, y=forward),
// matching Car.prefab (local (±0.3,0.27),(±0.3,0.42),(0,0.42), car scale (1,2))
// and Unity's hierarchy order (right side first). Mirror of car.ts.
const SENSOR_ANGLES = array<f32, 5>(-0.7853981633974483, -0.3587706702705722, 0.0, 0.3587706702705722, 0.7853981633974483);
const SENSOR_ORIGINS = array<vec2f, 5>(vec2f(-0.3, 0.54), vec2f(-0.3, 0.84), vec2f(0.0, 0.84), vec2f(0.3, 0.84), vec2f(0.3, 0.54));

const CAPTURE_RADIUS = 4.0;
// ponytail: point car with half-width; Unity used a 1x2 box collider.
const CAR_HALF_WIDTH = 0.5;

const DEG2RAD = 0.017453292519943295;
// FNN topology [5, 4, 3, 2], genome 47 floats, layer-major row-major with bias rows.
const TOPOLOGY = array<u32, 4>(5u, 4u, 3u, 2u);

fn raySegment(o: vec2f, d: vec2f, a: vec2f, b: vec2f) -> f32 {
  let s = b - a;
  let denom = d.x * s.y - d.y * s.x;
  if (abs(denom) < 1e-9) { return -1.0; }
  let ap = a - o;
  let t = (ap.x * s.y - ap.y * s.x) / denom;
  let u = (ap.x * d.y - ap.y * d.x) / denom;
  if (t >= 0.0 && u >= 0.0 && u <= 1.0) { return t; }
  return -1.0;
}

fn pointSegmentDist(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ab = b - a;
  let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
  return distance(p, a + ab * t);
}

fn softSign(x: f32) -> f32 {
  return x / (1.0 + abs(x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.carCount) { return; }
  var car = cars[i];
  if (car.alive == 0u) { return; }

  // 5 sensor raycasts, raw distances (not normalized), from front-mounted origins.
  let rad = car.angle * DEG2RAD;
  let fwd = vec2f(-sin(rad), cos(rad));
  let right = vec2f(cos(rad), sin(rad));
  var cur: array<f32, 5>;
  for (var s = 0u; s < 5u; s++) {
    let ca = cos(SENSOR_ANGLES[s]);
    let sa = sin(SENSOR_ANGLES[s]);
    let dir = vec2f(fwd.x * ca - fwd.y * sa, fwd.x * sa + fwd.y * ca);
    let origin = car.pos + right * SENSOR_ORIGINS[s].x + fwd * SENSOR_ORIGINS[s].y;
    var best = SENSOR_MAX_DIST;
    for (var w = 0u; w < params.wallCount; w++) {
      let wall = walls[w];
      let t = raySegment(origin, dir, wall.xy, wall.zw);
      if (t >= 0.0 && t < best) { best = t; }
    }
    let dist = max(best, SENSOR_MIN_DIST);
    cur[s] = dist;
    sensors[i * 5u + s] = dist;
  }

  // SoftSign forward pass over the genome (bias row last, input 1.0).
  var next: array<f32, 5>;
  var offset = i * 47u;
  var inCount = 5u;
  for (var l = 0u; l < 3u; l++) {
    let outCount = TOPOLOGY[l + 1u];
    for (var j = 0u; j < outCount; j++) {
      var sum = 0.0;
      for (var k = 0u; k < inCount; k++) {
        sum += cur[k] * genomes[offset + k * outCount + j];
      }
      sum += genomes[offset + inCount * outCount + j]; // bias row
      next[j] = softSign(sum);
    }
    offset += (inCount + 1u) * outCount;
    inCount = outCount;
    for (var j = 0u; j < 5u; j++) {
      cur[j] = select(cur[j], next[j], j < inCount);
    }
  }

  let turn = clamp(cur[0], -1.0, 1.0);
  let engine = clamp(cur[1], -1.0, 1.0);
  car.outputs = vec2f(turn, engine);
  let dt = params.dt;

  // ApplyInput: engine & braking force.
  if (engine < 0.0) {
    if (car.vel > 0.0) {
      // Active braking while moving forward
      car.vel = max(0.0, car.vel + engine * BRAKE_ACCEL * dt);
    } else if (car.vel > engine * MAX_VEL) {
      car.vel = max(-MAX_VEL, car.vel + engine * ACCELERATION * dt);
    }
  } else if (engine > 0.0) {
    if (car.vel < 0.0) {
      // Active braking while moving backward
      car.vel = min(0.0, car.vel + engine * BRAKE_ACCEL * dt);
    } else if (car.vel < engine * MAX_VEL) {
      car.vel = min(MAX_VEL, car.vel + engine * ACCELERATION * dt);
    }
  }
  // ApplyInput: rotation (negative sign matches CarMovement).
  car.angle += -turn * TURN_SPEED * dt;

  // ApplyVelocity.
  let rad2 = car.angle * DEG2RAD;
  car.pos += vec2f(-sin(rad2), cos(rad2)) * car.vel * dt;

  // ApplyFriction: only when throttle == 0.
  if (engine == 0.0) {
    if (car.vel > 0.0) { car.vel = max(car.vel - VEL_FRICT * dt, 0.0); }
    else if (car.vel < 0.0) { car.vel = min(car.vel + VEL_FRICT * dt, 0.0); }
  }

  // Wall death.
  for (var w = 0u; w < params.wallCount; w++) {
    let wall = walls[w];
    if (pointSegmentDist(car.pos, wall.xy, wall.zw) < CAR_HALF_WIDTH) {
      car.alive = 0u;
    }
  }

  // Checkpoint timeout & off-track grace period with grass friction.
  let currentIdx = car.cpIndex;
  if (currentIdx > 0u && currentIdx < params.cpCount) {
    let cpPrev = checkpoints[currentIdx - 1u].pos;
    let cpCurr = checkpoints[currentIdx].pos;
    let corrDist = pointSegmentDist(car.pos, cpPrev, cpCurr);
    if (corrDist > 5.5) {
      // Off-track grass friction slows car down
      car.vel *= 0.95;
      // Accelerate timeout while off-track (gives ~0.8s grace period)
      car.timeSinceCp += dt * 3.0;
      if (corrDist > 12.0) {
        car.alive = 0u;
      }
    }
  }
  car.timeSinceCp += dt;
  if (car.timeSinceCp > MAX_CHECKPOINT_DELAY) { car.alive = 0u; }

  if (car.alive == 0u) {
    car.vel = 0.0;
    // Strip partial credit for uncaptured segment on crash so suicidal wall hits are penalized!
    let idx = car.cpIndex;
    if (idx > 0u && idx <= params.cpCount) {
      car.fitness = checkpoints[idx - 1u].accReward;
    } else {
      car.fitness = 0.0;
    }
    cars[i] = car;
    return;
  }

  // Checkpoint capture + fitness (TrackManager.GetCompletePerc).
  var idx = car.cpIndex;
  if (idx < params.cpCount) {
    var dist = distance(car.pos, checkpoints[idx].pos);
    var captured = false;
    while (dist <= CAPTURE_RADIUS) {
      idx++;
      captured = true;
      if (idx >= params.cpCount) { break; }
      dist = distance(car.pos, checkpoints[idx].pos);
    }
    if (captured) { car.timeSinceCp = 0.0; }
    car.cpIndex = idx;
    if (idx >= params.cpCount) {
      car.fitness = 1.0;
    } else if (idx == 0u) {
      car.fitness = 0.0; // not yet across the start line (distToPrev[0] is 0)
    } else {
      let cp = checkpoints[idx];
      let complete = max((cp.distToPrev - dist) / cp.distToPrev, 0.0);
      car.fitness = checkpoints[idx - 1u].accReward + complete * cp.reward;
    }
  } else {
    car.fitness = 1.0;
  }

  cars[i] = car;
}
`;
