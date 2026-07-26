# Flappy Bird — Evolved Neural Networks on the GPU

A port of Flappy Bird with a population of birds evolved by a genetic algorithm.
One GPU thread per bird, all sharing **one pipe field** — the birds are a flock
flying the same course, not independent games.

The network shape and fitness follow the well-known TensorFlow.js implementation
of this demo; the physics follows the source repo's `Bird.update()`.

---

## 1. At a glance

| Piece | Value |
|---|---|
| Inputs | 5 |
| Network | `5 → 8 (ReLU) → 1 (sigmoid)` |
| Genome | 57 weights |
| Output | flap when sigmoid > 0.5 |
| Selection | roulette wheel (`nextRouletteGeneration`) |
| Fitness | `score²` |
| Bird state | 8 floats (32 bytes) |
| World | shared by the whole population |

---

## 2. The world

| Constant | Value |
|---|---|
| Canvas | 288 × 512 (`bg.png`) |
| Ground line | y = 394 |
| Bird | x = 50, 38 × 26 |
| Pipe width / gap | 52 / 75 |
| Pipe speed | 6 px per frame |
| Spawn interval | every 50 frames |
| First gap top | 59 … 242 |

Up to 16 pipes are live at once. The gap position is drawn randomly per pipe, so
the course differs between generations but is identical for every bird within one.

A generation ends when every bird is dead.

---

## 3. Inputs

Five values, each clamped to [0, 1], matching the original's `map()` calls:

| # | Input | Normalisation |
|---|---|---|
| 0 | Horizontal gap to the next pipe | `(pipe.x − BIRD_X) / (288 − BIRD_X)` |
| 1 | Top pipe's lower edge | `pipe.topY / GROUND_Y` |
| 2 | Bottom pipe's upper edge | `pipe.bottomY / GROUND_Y` |
| 3 | Bird's height | `y / GROUND_Y` |
| 4 | Vertical velocity | `(velY + 12) / 24` |

Inputs 1 and 2 give the gap's **edges** rather than its centre, which lets the
network learn its own safety margin instead of aiming at a midpoint.

The "next pipe" is the first whose right edge is still ahead of the bird. When no
pipe is on screen the inputs fall back to `(1.0, 0.5, 0.5)` — maximum distance,
gap centred — so an empty screen reads as "nothing to do", not as a pipe at zero
distance.

---

## 4. Output and physics

One sigmoid output; **flap when it exceeds 0.5**. Unlike the dino, there is no
ground gate — a bird may flap at any time, so a network that saturates its output
hovers near the ceiling and dies there.

```
if (action > 0.5) { velY += JUMP_UPLIFT; velY *= 0.9 }
velY += GRAVITY
velY *= 0.9
y += velY
```

The 0.9 damping is applied **twice** when flapping — once inside the jump, once
after gravity. That is faithful to the original's `Bird.jump()` followed by
`Bird.update()`, and it makes a flap noticeably weaker than raw uplift suggests.

Death: touching the ceiling or ground, or overlapping any pipe.

---

## 5. Network and genome

`5 → 8 (ReLU) → 1 (sigmoid)`, 57 weights, layer-major and input-major with each
layer's bias row last:

| Segment | Indices | Count |
|---|---|---|
| Layer 1 weights | `[k * 8 + h]`, k ∈ 0..4, h ∈ 0..7 | 40 |
| Layer 1 biases | `[40 + h]` | 8 |
| Layer 2 weights | `[48 + h]` | 8 |
| Layer 2 bias | `[56]` | 1 |
| **Total** | | **57** |

---

## 6. Fitness and selection

```
fitness = score²
```

`score` counts frames survived, and the square is the original's
`normalizeFitness` behaviour. Roulette selection is proportional to fitness, so
squaring is what separates a bird that clears two pipes from one that clears
three — without it, mediocre birds breed nearly as often as good ones.

Selection is **roulette wheel** (`nextRouletteGeneration`): parents chosen with
probability proportional to fitness, cloned, then per-weight gaussian mutation.
No crossover.

---

## 7. Files

| File | Responsibility |
|---|---|
| `flappy_buffers.ts` | Genome/state layout, world constants, GPU buffers |
| `flappy.wgsl.ts` | Per-bird inputs, forward pass, physics, collision |
| `flappy_evolution.ts` | Shared pipe field, generation driver |
| `flappy_renderer.ts` | Sprite rendering |
| `main.ts` | Settings panel, HUD, main loop |
| `../../utils/ga.ts` | **Shared** GA — roulette selection lives here |

World constants are duplicated as literals in `flappy.wgsl.ts` because WGSL
cannot import; **the header comments in both files say to keep them in sync.**
Interpolating the TypeScript constants into the shader template (as the pacman
demo does) would remove the risk of drift.

---

## 8. Notes and possible improvements

- **Flappy Bird is close to a one-decision problem**, which is why a 5-input
  network solves it quickly. It makes a good baseline for the shared GA rather
  than a hard problem in its own right.
- **The shared pipe field caps diversity.** Every bird sees the identical course,
  so between-bird variance comes only from their own trajectories. Independent
  per-bird courses would give selection a broader signal.
- **Only the next pipe is visible.** The bird cannot see the pipe after it, so it
  cannot set up its height in advance for a hard consecutive pair.
- **No crossover.** Roulette + mutation only. `nextCrossoverGeneration` and
  `nextTournamentGeneration` in the shared GA are drop-in alternatives worth
  measuring.
