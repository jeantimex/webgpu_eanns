# Snake — Evolved Neural Networks on the GPU

Snake on a 38 × 38 board, with a population evolved by a genetic algorithm. Each
agent plays its **own complete game** — its own snake, its own apple sequence — in
a single GPU thread, so a population of hundreds runs in parallel with no shared
world state.

The vision model, network shape and fitness follow CodeBullet's SnakeAI.

---

## 1. At a glance

| Piece | Value |
|---|---|
| Inputs | 24 (8 directions × 3) |
| Network | `24 → 16 → 16 → 4`, ReLU on every layer |
| Genome | 740 weights |
| Output | argmax over 4 absolute directions |
| Selection | roulette + per-layer single-point crossover |
| Fitness | `lifetime² × 2^apples` |
| Board | 38 × 38 = 1444 cells |
| Turn model | one dispatch per move (turn-based, not real-time) |

---

## 2. The game

A snake starts length 3 in the centre. Eating an apple grows it by one and spawns
the next apple on a free cell drawn from a **per-agent xorshift RNG** — so every
agent gets a different apple sequence.

Collision uses two structures in parallel:

- a **bitmask** (`bodyMask`, one bit per cell) for O(1) "am I about to hit myself"
- a **ring buffer** (`ring`) of segment cell ids, so the tail can be popped in O(1)

That is what makes hundreds of snakes affordable: neither collision nor growth
requires walking the body.

### Termination — the starvation budget

| Rule | Value |
|---|---|
| Starting move budget | 200 |
| Gained per apple | +100 |
| Budget cap | 500 |
| Hard cap | 10,000 moves |

The budget decrements every move and the game ends at zero. This is a neat piece
of design worth noting: it needs **no penalty term at all**. A snake that loops
forever simply runs out of budget and dies, while one that keeps eating keeps
topping up. Compare the pacman demo, which uses a wall-clock stall timeout for the
same purpose.

---

## 3. Inputs — 8-direction vision

Twenty-four values: a raycast along each of the 8 compass directions (W, NW, N,
NE, E, SE, S, SW), each contributing three numbers:

| Per direction | Meaning |
|---|---|
| `foodFlag` | 1 if the apple lies anywhere along this ray |
| `bodyFlag` | 1 if the snake's own body lies anywhere along this ray |
| `1 / dist` | reciprocal distance to the wall in that direction |

The ray walks until it leaves the board, setting the flags on the way. `1/dist`
rather than `dist` means near walls produce large values and distant ones fade
toward zero, so the input is naturally scaled without an explicit normaliser.

This is a genuinely **raw** perception: no path-finding, no notion of reachability.
It is why the network needs two hidden layers — everything about the board's
structure has to be computed from these rays rather than handed over pre-digested.
(Contrast the pacman demo, whose inputs are BFS maze distances and which therefore
gets by with one small hidden layer.)

**A known limitation:** a ray reports *that* the apple is in some direction, not
whether it is reachable. A snake that has coiled around the apple sees exactly the
same `foodFlag` as one with a clear run at it.

---

## 4. Output

Four outputs, one per **absolute** direction (up / down / left / right), argmax
wins. A guard rejects a direct reversal:

```wgsl
if (dir > 3u || newDir != (dir ^ 1u)) { dir = newDir; }
```

`dir ^ 1` is the opposite direction under the 0=up, 1=down, 2=left, 3=right
encoding. `dir > 3` is the "not moving yet" start state, where any direction is
allowed.

Actions are deterministic — no sampling. This differs from the pacman demo, where
sampling turned out to be load-bearing; snake's turn-based movement and starvation
budget make loops self-limiting, so it does not need action noise to escape them.

---

## 5. Network and genome

`24 → 16 → 16 → 4` with **ReLU on every layer, including the output**. 740
weights, layer-major and input-major with each layer's bias row last:

| Layer | Shape | Weights | Biases | Running offset |
|---|---|---|---|---|
| 1 | 24 → 16 | 384 | 16 | 0 |
| 2 | 16 → 16 | 256 | 16 | 400 |
| 3 | 16 → 4 | 64 | 4 | 672 |
| **Total** | | | | **740** |

ReLU on the output layer is unusual — it clamps every negative preference to
exactly 0, so several directions can tie at zero and argmax then picks the lowest
index. It is faithful to the source, and worth knowing about if you ever see a
snake with a systematic bias toward "up".

---

## 6. Fitness

```js
score < 10 ? lifetime² * 2**score
           : lifetime² * 1024 * (score - 9)
```

This is the most instructive fitness function in the repo, and it is
**multiplicative on purpose**:

- **Neither factor scores alone.** Zero apples gives `2^0 = 1`, collapsing fitness
  to bare lifetime. Dying instantly gives `lifetime² ≈ 0`. You have to survive
  *and* eat.
- **Exponential early** (`2^score`) so the very first apple doubles fitness — that
  is what bootstraps a population out of random flailing.
- **Linear past 9 apples** so the term cannot overflow, and so late-game progress
  stays a smooth gradient rather than an unreachable cliff.

Note the shader contains **no reward shaping whatsoever** — it only plays the
game. All judgment happens on the CPU from two raw facts, `moves` and `score`.
That separation is worth preserving.

---

## 7. Selection

`nextCrossoverGeneration` in the shared GA: roulette selection of **two** parents,
then **per-layer single-point crossover** — a random row/column cut in each weight
matrix independently — then gaussian mutation with clamping, plus elitism.

Per-layer cuts (rather than one cut across the flat genome) mean a child can
inherit layer 1 from one parent and layer 2 from the other, which suits a network
whose layers do fairly distinct jobs.

---

## 8. Files

| File | Responsibility |
|---|---|
| `snake_buffers.ts` | Genome/state layout, board constants, GPU buffers |
| `snake.wgsl.ts` | Vision, forward pass, movement, growth, collision |
| `snake_evolution.ts` | Fitness, generation driver, best-genome tracking |
| `snake_renderer.ts` | Board and snake rendering |
| `model.ts` | Save/load genomes to JSON and localStorage |
| `main.ts` | Settings panel, HUD, network panel, main loop |
| `../../utils/ga.ts` | **Shared** GA — roulette, crossover, mutation, elitism |

Unlike the dino and flappy demos, this shader **interpolates its constants from
TypeScript** (`const GRID = ${GRID};`), so the CPU and GPU layouts cannot drift
apart. It is the pattern the other demos should follow.

---

## 9. Notes and possible improvements

- **Fitness is a single episode, and apples are random.** The same genome can
  score very differently depending on where apples happen to spawn, so selection
  is partly selecting luck. The pacman demo measured a 4× spread from exactly
  this and fixed it with `src/utils/evaluation.ts` (multi-episode scoring with
  common random numbers) — **that helper is shared and snake could adopt it
  directly.** This is the highest-value open item here.
- **Reachability is invisible.** The raycasts cannot distinguish an apple with a
  clear path from one the snake has sealed off. A flood-fill or BFS feature would
  address the late-game self-trapping that limits long snakes.
- **`bestReplaySeed` is set to `bestIndex`** — the *index* of the best agent, used
  as a replay seed. It works, but the name suggests a seed value rather than an
  index, which is easy to misread.
- **Deterministic actions.** Worth measuring whether a little sampling helps, as
  it did substantially in the pacman demo.
