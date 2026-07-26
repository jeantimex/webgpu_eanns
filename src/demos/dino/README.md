# Chrome Dino — Evolved Neural Networks on the GPU

A port of the Chrome offline dinosaur game, with a population of dinos evolved by
a genetic algorithm. Every dino runs in its own GPU thread, but unlike the other
demos here **they all share one world**: a single obstacle stream, one speed ramp,
one ground scroll. The population is a crowd racing the same course, not many
independent games.

Physics follows the source repo's `update()` frame-for-frame — per-frame integer
steps at 60 fps, no delta time.

---

## 1. At a glance

| Piece | Value |
|---|---|
| Inputs | 6 |
| Network | `6 → 8 (ReLU) → 1 (sigmoid)` |
| Genome | 65 weights |
| Output | jump when sigmoid > 0.5 |
| Selection | roulette wheel (`nextRouletteGeneration`) |
| Fitness | `score²` |
| Dino state | 8 floats (32 bytes) |
| World | shared by the whole population |

---

## 2. The world

| Constant | Value |
|---|---|
| Canvas | 1000 × 400 |
| Platform line | y = 300 |
| Dino | x = 100, 89 × 94 |
| Gravity | 0.6 per frame |
| Jump velocity | −15 |
| Speed | `7 + score/100`, capped at **17** |

One obstacle is alive at a time. When it leaves the screen the next spawns,
**alternating small and large** (`makeObstacle(!obs.small)`), with a random
1–3× width multiplier on a 34 px (small) or 49 px (large) base. Score ticks once
every 7 frames and drives the speed ramp, so the game gets harder over time.

A generation ends when every dino is dead; the world then resets to base speed
with a fresh obstacle.

---

## 3. Inputs

Six values, each clamped to [0, 1]:

| # | Input | Normalisation |
|---|---|---|
| 0 | Gap to the obstacle | `(obs.x − (DINO_X + DINO_W)) / 1000` |
| 1 | Obstacle width | `obs.w / 150` |
| 2 | Obstacle height | `obs.h / 100` |
| 3 | Height above the ground | `(PLAT_Y − (y + DINO_H)) / 200` |
| 4 | Vertical velocity | `(velY − JUMP_VEL) / 30` |
| 5 | Current game speed | `gamespeed / 17` |

The set is deliberately minimal. Inputs 0–2 say *what is coming and how big*,
3–4 say *where I am in a jump*, and 5 says *how much time I have*, which is what
makes a fixed jump distance insufficient as the speed ramps up.

Because every dino shares one world, inputs 0–2 and 5 are **identical across the
whole population** on any given frame. Only 3 and 4 differ, and only once a dino
has jumped. Generation 1 therefore behaves as a single mass that gradually fans
out — which is normal here, not a bug.

---

## 4. Output and physics

One sigmoid output; **jump when it exceeds 0.5**. Jumping is only permitted from
the ground (`onGround`), matching the original's `keyDown` gate, so holding the
output high does not produce a hover.

```
if (action > 0.5 && onGround) velY = -15
if (!onGround)               velY += 0.6
y += velY
if (y + DINO_H > PLAT_Y) { y = PLAT_Y - DINO_H; onGround = true }
```

Collision uses the original's forgiving hitbox: horizontal overlap with the
obstacle plus `y > obs.y - 75`.

---

## 5. Network and genome

`6 → 8 (ReLU) → 1 (sigmoid)`, 65 weights, layer-major and input-major with each
layer's bias row last:

| Segment | Indices | Count |
|---|---|---|
| Layer 1 weights | `[k * 8 + h]`, k ∈ 0..5, h ∈ 0..7 | 48 |
| Layer 1 biases | `[48 + h]` | 8 |
| Layer 2 weights | `[56 + h]` | 8 |
| Layer 2 bias | `[64]` | 1 |
| **Total** | | **65** |

---

## 6. Fitness and selection

```
fitness = score²
```

`score` counts frames survived. Squaring widens the gap between a dino that
clears three obstacles and one that clears four, which matters because roulette
selection is proportional to fitness — a linear score would give a mediocre dino
nearly the same breeding odds as a good one.

Selection is **roulette wheel** (`nextRouletteGeneration`): pick parents with
probability proportional to fitness, clone, then mutate each weight with
probability `mutateRate` by a gaussian of width `sigma`. No crossover.

---

## 7. Files

| File | Responsibility |
|---|---|
| `dino_buffers.ts` | Genome/state layout, world constants, GPU buffers |
| `dino.wgsl.ts` | Per-dino inputs, forward pass, physics, collision |
| `dino_evolution.ts` | Shared world (obstacle spawn, speed ramp), generation driver |
| `dino_renderer.ts` | Sprite rendering |
| `main.ts` | Settings panel, HUD, main loop |
| `../../utils/ga.ts` | **Shared** GA — roulette selection lives here |

World constants are duplicated as literals in `dino.wgsl.ts` because WGSL cannot
import; **the header comment in both files says to keep them in sync.** The
pacman demo solved this by interpolating TypeScript constants into the shader
template, which would be the better fix here too.

---

## 8. Notes and possible improvements

- **The shared world caps diversity.** Every dino faces the identical obstacle
  sequence, so a lucky spawn helps everyone equally and a hard one kills
  everyone. That is faithful to the original but gives the GA a narrow signal.
  Independent per-dino worlds (as pacman and snake use) would let selection see
  which dino handles *varied* obstacles.
- **Fitness is a single episode**, so it carries the same noise problem
  documented in the pacman README. Here it is milder, because the shared world
  removes most of the between-agent variance.
- **No crossover.** Roulette + mutation only. `nextCrossoverGeneration` and
  `nextTournamentGeneration` in the shared GA are drop-in alternatives worth
  measuring.
- **Only one obstacle is visible.** The network cannot see a second obstacle
  queued behind the first, so it cannot plan a landing.
