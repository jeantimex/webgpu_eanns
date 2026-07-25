/** Track definition: pure data, loaded from JSON (see public/tracks/). */
export interface Track {
  name: string;
  /** Start pose. angleDeg is CCW-positive degrees; forward = +Y at 0 (Unity convention). */
  start: { x: number; y: number; angleDeg: number };
  /** Ordered racing line, order matters. checkpoints[0] is the start line. */
  checkpoints: [number, number][];
  /** Wall segments as [x1, y1, x2, y2]. */
  walls: [number, number, number, number][];
}

/** Unity Checkpoint.CaptureRadius (3.0) + car length (1.0) = 4.0 effective capture radius. */
export const CAPTURE_RADIUS = 4.0;

/** Per-checkpoint derived geometry, mirroring TrackManager.CalculateCheckpointPercentages. */
export interface CheckpointTable {
  /** Distance to the previous checkpoint ([0] = 0). */
  distToPrev: number[];
  /** Share of total track length this checkpoint awards ([0] = 0). */
  reward: number[];
  /** Sum of rewards of all checkpoints up to this one. */
  accReward: number[];
  trackLength: number;
}

export function buildCheckpointTable(track: Track): CheckpointTable {
  const cps = track.checkpoints;
  const n = cps.length;
  const distToPrev = new Array<number>(n).fill(0);
  const accDistance = new Array<number>(n).fill(0);
  for (let i = 1; i < n; i++) {
    distToPrev[i] = Math.hypot(cps[i][0] - cps[i - 1][0], cps[i][1] - cps[i - 1][1]);
    accDistance[i] = accDistance[i - 1] + distToPrev[i];
  }
  const trackLength = accDistance[n - 1];
  const reward = new Array<number>(n).fill(0);
  const accReward = new Array<number>(n).fill(0);
  for (let i = 1; i < n; i++) {
    reward[i] = accDistance[i] / trackLength - accReward[i - 1];
    accReward[i] = accReward[i - 1] + reward[i];
  }
  return { distToPrev, reward, accReward, trackLength };
}

/** Walls flattened for GPU upload: 4 f32 per wall, 16 bytes (one vec4f each). */
export function wallsFlat(track: Track): Float32Array<ArrayBuffer> {
  return new Float32Array(track.walls.flat());
}

/**
 * Checkpoints flattened for GPU upload: 8 f32 per checkpoint (32-byte struct stride):
 * [x, y, distToPrev, reward, accReward, pad, pad, pad].
 */
export function checkpointsFlat(track: Track, table: CheckpointTable): Float32Array<ArrayBuffer> {
  const out = new Float32Array(track.checkpoints.length * 8);
  for (let i = 0; i < track.checkpoints.length; i++) {
    out.set(
      [track.checkpoints[i][0], track.checkpoints[i][1], table.distToPrev[i], table.reward[i], table.accReward[i]],
      i * 8,
    );
  }
  return out;
}

function isNumberArray(value: unknown, length: number): boolean {
  return Array.isArray(value) && value.length === length && value.every((v) => typeof v === 'number');
}

export function parseTrack(json: unknown): Track {
  const t = json as Track;
  if (typeof t?.name !== 'string') throw new Error('Track: missing "name"');
  if (!t.start || !isNumberArray([t.start.x, t.start.y, t.start.angleDeg], 3))
    throw new Error('Track: "start" must be {x, y, angleDeg} numbers');
  if (!Array.isArray(t.checkpoints) || t.checkpoints.length < 2 || !t.checkpoints.every((c) => isNumberArray(c, 2)))
    throw new Error('Track: "checkpoints" must be at least two [x, y] points');
  if (!Array.isArray(t.walls) || t.walls.length < 1 || !t.walls.every((w) => isNumberArray(w, 4)))
    throw new Error('Track: "walls" must be [x1, y1, x2, y2] segments');
  return t;
}

export async function loadTrack(url: string): Promise<Track> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load track ${url}: ${response.status}`);
  return parseTrack(await response.json());
}

/**
 * Render-only wall representation. Extracted Unity strips are stored as their two
 * long collider edges 0.5 units apart; pair those back up and emit the strip
 * centerline so the renderer can draw one 0.5-thick strip (like Unity's sprite)
 * instead of two separate edge lines. Unpaired segments pass through unchanged.
 * ponytail: O(n²) pair search, fine for a few hundred segments done once per track.
 */
export function buildRenderWalls(track: Track): number[][] {
  const walls = track.walls;
  const used = new Array(walls.length).fill(false);
  const out: number[][] = [];
  const mid = (w: number[]) => [(w[0] + w[2]) / 2, (w[1] + w[3]) / 2];
  for (let i = 0; i < walls.length; i++) {
    if (used[i]) continue;
    used[i] = true;
    const a = walls[i];
    const da = [a[2] - a[0], a[3] - a[1]];
    const la = Math.hypot(da[0], da[1]);
    const ma = mid(a);
    let paired = false;
    for (let j = i + 1; j < walls.length; j++) {
      if (used[j]) continue;
      const b = walls[j];
      const db = [b[2] - b[0], b[3] - b[1]];
      const lb = Math.hypot(db[0], db[1]);
      // Parallel (cross product ~ 0) and midpoints ~0.5 apart (strip width).
      const cross = Math.abs(da[0] * db[1] - da[1] * db[0]) / (la * lb || 1);
      const mb = mid(b);
      const gap = Math.hypot(ma[0] - mb[0], ma[1] - mb[1]);
      if (cross < 0.02 && gap > 0.35 && gap < 0.65) {
        // Edge b may be flipped relative to a; pick the endpoint pairing that
        // keeps the centerline length close to the originals.
        const s1 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, (a[3] + b[3]) / 2];
        const s2 = [(a[0] + b[2]) / 2, (a[1] + b[3]) / 2, (a[2] + b[0]) / 2, (a[3] + b[1]) / 2];
        const l1 = Math.hypot(s1[2] - s1[0], s1[3] - s1[1]);
        const l2 = Math.hypot(s2[2] - s2[0], s2[3] - s2[1]);
        out.push(Math.abs(l1 - la) <= Math.abs(l2 - la) ? s1 : s2);
        used[j] = true;
        paired = true;
        break;
      }
    }
    if (!paired) out.push(a);
  }
  // Adjacent strips rarely share exact endpoints, which leaves one joint disc per
  // endpoint (visible bumps). Snap endpoints within a strip-width of each other to
  // their average so a corner gets a single disc. ponytail: O(n²), once per track.
  const SNAP = 0.75;
  const endpoints: Array<[number, number]> = [];
  const owner: Array<[number, number]> = []; // [segment index, 0=start|1=end]
  out.forEach((s, i) => {
    endpoints.push([s[0], s[1]], [s[2], s[3]]);
    owner.push([i, 0], [i, 1]);
  });
  for (let i = 0; i < endpoints.length; i++) {
    for (let j = i + 1; j < endpoints.length; j++) {
      if (owner[i][0] === owner[j][0]) continue;
      const dx = endpoints[i][0] - endpoints[j][0];
      const dy = endpoints[i][1] - endpoints[j][1];
      if (dx * dx + dy * dy < SNAP * SNAP) {
        const mx = (endpoints[i][0] + endpoints[j][0]) / 2;
        const my = (endpoints[i][1] + endpoints[j][1]) / 2;
        endpoints[i][0] = endpoints[j][0] = mx;
        endpoints[i][1] = endpoints[j][1] = my;
      }
    }
  }
  out.forEach((s, i) => {
    s[0] = endpoints[2 * i][0];
    s[1] = endpoints[2 * i][1];
    s[2] = endpoints[2 * i + 1][0];
    s[3] = endpoints[2 * i + 1][1];
  });
  return out;
}
