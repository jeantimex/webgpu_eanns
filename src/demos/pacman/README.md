# Pac-Man — Evolved Neural Network Agents on the GPU

A genetic algorithm trains a small neural network to play the arcade Pac-Man maze.
Every agent plays its **own complete game** — its own pacman, its own four ghosts,
its own pellet field — in a single GPU thread, so a population of hundreds runs in
parallel with no shared world state.

The design follows a published writeup on training a Pac-Man AI with genetic
algorithms, with two deliberate departures that are documented and measured below.

---

## Contents

1. [Architecture at a glance](#1-architecture-at-a-glance)
2. [The environment](#2-the-environment)
3. [The maze distance table](#3-the-maze-distance-table)
4. [Inputs — the perception vector](#4-inputs--the-perception-vector)
5. [Output — softmax and sampling](#5-output--softmax-and-sampling)
6. [Network and genome layout](#6-network-and-genome-layout)
7. [Fitness](#7-fitness)
8. [The genetic algorithm](#8-the-genetic-algorithm)
9. [The sealed weight box — the bug that capped everything](#9-the-sealed-weight-box--the-bug-that-capped-everything)
10. [Things we got wrong along the way](#10-things-we-got-wrong-along-the-way)
11. [Measured results](#11-measured-results)
12. [Debug tooling](#12-debug-tooling)
13. [Files](#13-files)
14. [Verifying changes headlessly](#14-verifying-changes-headlessly)
15. [Known limits and next steps](#15-known-limits-and-next-steps)

---

## 1. Architecture at a glance

| Piece | Value |
|---|---|
| Perception vector | 11 inputs |
| Network | `11 → 6 (ReLU) → 4`, softmax over outputs |
| Genome | 100 float weights |
| Action | sampled from the softmax (temperature 1.0) |
| Selection | tournament, k = 3 |
| Crossover | single-point on the flat genome, rate 0.8 |
| Mutation | 50% reset / 50% gaussian drift, rate 0.05 adaptive |
| Elitism | top 2% copied unchanged |
| Agent state | 66 floats (264 bytes) per agent |
| Population | configurable, default 300 |

One compute dispatch advances every agent by one 60 Hz game tick. A generation
runs until every agent's game has ended, then the GA step happens on the CPU.

---

## 2. The environment

The maze, ghost AI, speeds and scoring are ported from `pacman-js` (MIT). Ghosts
use the authentic targeting rules — Blinky chases directly, Pinky aims four tiles
ahead, Inky mirrors through Blinky, Clyde scatters when close — with 7 s scatter /
20 s chase phases and timed house release.

An agent's game ends on any of:

| Termination | Rule | Why it exists |
|---|---|---|
| Caught | Any non-frightened ghost within 1 tile | `START_LIVES = 0`, one attempt per thread |
| Stall | `STALL_SECS = 8` with no pellet eaten | Kills campers and loopers; the writeup's §2.3 step cap |
| Board timeout | `LEVEL_SECS` (default 90 s, adjustable 30–180 s) | Bounds a generation's wall-clock cost |
| Hard cap | `MAX_GAME_TICKS = 21600` (6 min) | Backstop across multiple boards |

Clearing all 244 pellets advances to the next board with score and level carried
over.

---

## 3. The maze distance table

Everything spatial is measured in **maze steps**, never straight-line distance.
`mazeGraph()` in `maze.ts` runs a BFS from every reachable tile once at startup
and packs all-pairs shortest paths into a read-only storage buffer.

| Property | Value |
|---|---|
| Reachable tiles | 300 |
| Table size | 300 × 300 bytes = **90 KB**, `u8` packed 4 per `u32` |
| Maze diameter | 52 steps |
| Tunnel | Row 14 wraps column 0 ↔ 27, so paths route through it when shorter |
| Ghost house | 18 enclosed tiles map to index −1, so a housebound ghost reads as no threat |

**Why this matters more than anything else in the file.** A wall-blind metric has
local minima *inside wall blocks*. The direction to the "nearest" pellet can point
straight through a wall, and it flips sign as pacman slides past — producing a
stable two-tile oscillation where the agent paces back and forth forever instead
of eating. Path distance has no such minima: descending it always makes progress.

This is our **first deliberate departure from the writeup**, which uses Euclidean
distance. The writeup pays for that choice in its own §4.4, where it diagnoses its
plateau as a lack of global strategy: purely reactive, with no long-range planning.

Measured sanity check: greedy descent of this metric clears all 244 pellets in
**333 tiles ≈ 30 s** of the 90 s budget, with a longest gap between pellets of only
2.0 s. So the environment is comfortably solvable; anything short of that is a
learning limitation, not an environment one.

---

## 4. Inputs — the perception vector

11 dimensions. The writeup recommends 8–12 (too few and the AI turns ultra-passive;
too many and it overfits one map), so we sit inside that band.

| # | Input | Range | Meaning |
|---|---|---|---|
| 0 | `pelletDist` | 0–1 | Maze steps to nearest remaining pellet ÷ 52 |
| 1–2 | `pelletDir` | −1/0/1 each | **Unit vector** of the neighbour tile that starts the shortest path there |
| 3 | `ghostDist` | 0–1 | Maze steps to nearest on-board ghost ÷ 52 |
| 4–5 | `ghostDir` | −1/0/1 each | **Unit vector** of the neighbour starting the shortest path to it |
| 6 | `ghostState` | 0 or 1 | 0 = that ghost is hunting, 1 = frightened and edible |
| 7–8 | `powerDir` | −1/0/1 each | Unit vector toward nearest power pellet; `(0,0)` when none remain |
| 9 | `aheadClear` | 0 or 1 | Is the tile in front of the current heading open? |
| 10 | `progress` | 0–1 | Pellets eaten on this board ÷ 244 |

### Why each one

- **`pelletDist` + `pelletDir`** are the core drive. Distance says *how urgent*,
  direction says *which way*. Because direction is derived from the BFS table, it
  is the true first step of a shortest path, not a bearing.
- **`ghostDist` + `ghostDir`** are the survival signal, and the reason the writeup
  warns against dropping ghost inputs: an agent that can't see ghosts learns to
  hide rather than play.
- **`ghostState`** lets one pair of direction inputs serve two opposite behaviours.
  Without it the net cannot tell "flee" from "chase", and power pellets are useless.
- **`powerDir`** is the counterattack opportunity. `(0,0)` when all four are eaten
  encodes *absence* naturally, with no special-case value.
- **`aheadClear`** exists so wall-bumping is *learnable*. We deliberately do **not**
  mask illegal moves — the net may choose a wall, and pays for it through the
  fitness penalty. Masking would hide the mistake instead of teaching it.
- **`progress`** is a global signal that distinguishes early board from endgame.

### Second deliberate departure: directions are vectors, not indices

The writeup encodes direction as an **index** — up = 0, down = 0.33, left = 0.67,
right = 1.0 — on a single ordinal axis. That is a categorical variable flattened
onto a line. For the network to *follow* it, the hidden layer must construct four
separate bump functions on that axis, so that output *m* peaks only when the scalar
happens to land in slot *m*. Each bump costs roughly two ReLUs; we have six units
total, and evolution has to find that structure by random search.

As a unit vector the identical behaviour is a plain **linear** map:

```
out_up    = −dy      out_down  = +dy
out_left  = −dx      out_right = +dx
```

Cost: one extra dimension per direction feature. Measured benefit at generation 1,
same seed, before any evolution has happened:

| Direction encoding | Gen-1 avg pellets (random genomes) |
|---|---|
| Index `dirIndex/3` (writeup) | 8.5 |
| **Unit vector `(dx, dy)`** | **34.0** |

A 4× improvement from purely random weights, because a random linear map already
correlates with "go that way" while a random bump-detector does not.

Important caveat, because it took us a while to see it: **this change alone bought
almost nothing at generation 25** (43.8 → 44.9). It only paid off once weights could
grow — see §9. The encoding makes the right behaviour *cheap to express*; the weight
range determines whether it can be expressed *strongly*.

---

## 5. Output — softmax and sampling

Four outputs, one per direction, index-aligned with the movement encoding
(0 = up, 1 = down, 2 = left, 3 = right). Softmax turns them into a distribution and
the action is **sampled** from it, not taken as argmax:

```wgsl
probs[m] = exp((logits[m] - maxLogit) / temperature)
// then roulette-select over probs using a per-agent xorshift stream
```

Each agent carries its own RNG state (`A.rng`) so the streams are independent.

Sampling is not a detail — it is load-bearing. We swept the temperature at 25
generations, identical seed:

| Temperature | avg pellets | Interpretation |
|---|---|---|
| 0.0 (pure argmax) | **19.3** | Deterministic; gets trapped in loops with no noise to escape |
| 0.1 | 35.9 | Nearly deterministic |
| 0.3 | 61.8 | |
| **1.0 (default)** | **68.4** | Best |
| 1.5 | 49.1 | |
| 2.5 | 48.4 | Approaching a random walk |
| 4.0 | 50.1 | |

Argmax is the **worst** setting tested — worse than heavy randomness. The writeup's
claim that randomness aids exploration is correct, and temperature 1.0 is
genuinely near-optimal rather than an arbitrary default.

The replay/test agent always takes the **mode** so the pacman you watch on screen
behaves deterministically, while training agents sample.

---

## 6. Network and genome layout

`11 → 6 (ReLU) → 4`. Shallow on purpose: the writeup's §2.2 specifies one hidden
layer of 4–8 units, and a bigger network explodes the search space that a GA has to
cover by random variation alone.

Flat genome, layer-major and input-major with each layer's **bias row last**:

| Segment | Indices | Count |
|---|---|---|
| Layer 1 weights | `[k * 6 + h]`, k ∈ 0..10, h ∈ 0..5 | 66 |
| Layer 1 biases | `[66 + h]` | 6 |
| Layer 2 weights | `[72 + h * 4 + m]`, h ∈ 0..5, m ∈ 0..3 | 24 |
| Layer 2 biases | `[96 + m]` | 4 |
| **Total** | | **100** |

This is the same layout `NetworkPanel` expects, so the on-screen network diagram
renders the real genome with no translation.

---

## 7. Fitness

The writeup's first attempt was `fitness = final score`, and evolution immediately
found the degenerate optimum: **stand perfectly still** — score 0, but never caught
and never penalised. The composite fix (§3.1) uses four terms:

| Term | Weight | Source |
|---|---|---|
| Arcade score | 1.0 | `FIT_SCORE_W` |
| Survival time | **0.1** | `FIT_SURVIVAL_W` |
| Pellets eaten % | **2.0** | `FIT_PELLET_PCT_W` |
| Wasted-move penalty | −0.3 | `FIT_WASTED_W` |

The two extreme weights carry the whole lesson:

- **Survival is weighted lowest (0.1) on purpose.** Weight survival heavily and
  you breed hiding strategies, which is the writeup's explicit warning. This is also
  why we did **not** copy the snake demo's `lifetime²` term: snake has no clock, so
  living longer is pure upside, whereas pacman must clear a board against a timer.
- **Pellet percentage is weighted highest (2.0)** — it is the core driving force,
  and the actual objective.

### Two implementation details that are easy to get wrong

**All four terms are normalised to 0–100 first.** The writeup's raw weights assume
its own scoring scale. Applied to arcade scoring, raw score (2440 for one board,
plus up to 1600 per power pellet from ghost combos) would swamp the pellet term at
roughly 12:1 and breed ghost-hunters instead of board-clearers. `FIT_SCORE_NORM`
puts score on the same 0–100 footing as everything else.

**The wasted-move penalty counts the *fraction of life* spent bumping, not raw
bumps.** Generation-1 agents average **417** wall bumps. At a raw weight of 0.5 that
is a −208 deduction against terms worth ~10, so every fitness went negative, got
clamped by a `Math.max(1, …)` floor, and **the entire population tied at 1** —
tournament selection would have been choosing uniformly at random. Two fixes:
normalise the penalty by lifetime, and drop the positive-floor clamp entirely
(tournament only ever *compares* fitnesses, so negative values are perfectly fine —
it was roulette selection that needed positivity).

---

## 8. The genetic algorithm

Lives in `src/utils/ga.ts` as `nextTournamentGeneration()`, **shared, not
pacman-specific**. The snake demo's roulette GA (`nextCrossoverGeneration`) is
untouched and still available.

### Selection — tournament, k = 3

Draw 3 random individuals, the fittest becomes a parent. The writeup tested this
against roulette and preferred it for Pac-Man because it purges completely useless
random strategies fast. It is also indifferent to fitness *scale*
and *sign*, which is what lets us drop the positivity clamp above.

The cost is diversity, which is why elitism stays small and mutation adapts.

### Crossover — single-point, rate 0.8

Pick a cut point in the flat genome; the child keeps parent A before it and takes
parent B after. High rate (0.8) to encourage mixing of useful traits. The writeup
describes this as combining small effective behaviour modules like stacking
building blocks.

### Mutation — 50% reset, 50% drift, rate 0.05 (adaptive)

| Operator | Share | Effect |
|---|---|---|
| **Reset** | 50% | `w = uniform(−1, 1)` — a fresh small value. Exploration. |
| **Drift** | 50% | `w += gaussian × 0.5`, clamped to ±8. Lets magnitude grow. |

The reset operator is the writeup's. The drift operator is ours, and §9 explains
why it is not optional.

**Adaptive rate.** Per §3.3, when the best fitness goes 5 generations without
improving, nudge mutation up by 0.02 (capped at 0.12); when progress resumes, decay
it back toward the 0.05 base. Both directions matter — see §10.

### Elitism — top 2%

The best 2% are copied to the next generation untouched, so a discovered optimum is
never lost to unlucky crossover. The writeup calls elitism a double-edged sword:
it accelerates convergence *and* suppresses diversity, hence keeping it small.

### Environment randomisation

Per §5.1, ghost speed and house-release timing are re-rolled ±15% every generation
so strategies must generalise rather than overfit one fixed ghost schedule.

---

## 9. The sealed weight box — the bug that capped everything

This was the single largest win in the project, and it hid behind a faithful
implementation of the source material.

### Symptom

Agents plateaued around 40 pellets (16% of a board) and stayed there for 150
generations. About **70% of deaths were stall timeouts** — the agent wandered into
a corner and went 8 seconds without eating, on a board still covered in pellets.

### Diagnosis

We instrumented a trained genome and evaluated its policy on 4000 sampled states:

```
mean max softmax prob : 0.485    (0.25 = uniform, 1.0 = certain)
argmax == pelletDir   : 43.3%    (25% = chance)
largest |weight|      : 0.982    ← the tell
```

The network *was* using the pellet signal (43% ≫ 25% chance), but it could only ever
be about half-confident, so sampling overrode its intent constantly — a semi-random
walk that drifts into cleared pockets and stalls.

**No weight in the entire 100-gene genome ever exceeded 1.0.** The cause:

- Mutation was implemented as the writeup describes it — *reset* the weight to a
  fresh random value in a small range, `(rand × 2 − 1) × mutateRange` with
  `mutateRange = 1`.
- Crossover only **copies** values that already exist.
- Initialisation draws from `(−1, 1)`.

So no operator anywhere in the GA could produce a weight outside `[−1, 1]`. The
reachable weight space was a **sealed box**. Bounded weights → bounded logits →
a softmax that can never sharpen → a policy that can never commit.

### Fix

Split mutation: half the mutations still reset (the writeup's exploration
operator), half are gaussian drifts that can carry a weight outside its initial
range, clamped at ±8.

### Result — A/B, identical seed, 25 generations

| Mutation | avg pellets | best | stall deaths | max \|w\| |
|---|---|---|---|---|
| Reset-only (writeup, literal) | 40.3 | 112 | 216 / 300 | 0.99 |
| **50% reset + 50% drift** | **68.4** | **231** | 181 / 300 | 1.22 |

**+70% pellets.** Note how small the weight change is — 0.99 → 1.22. It is not that
the weights became large; it is that they became *free to become large*, and the
gradient of selection could finally act on magnitude as well as sign.

### Where the fix lives

In the **shared GA core**, `src/utils/ga.ts`, as `resetShare` / `driftSigma` /
`clamp` on `TournamentOptions`. `resetShare` **defaults to 0.5**, so the fix applies
to any future demo automatically rather than being an opt-in that must be
rediscovered. The option is documented in place with the measured cost of setting it
back to 1.

---

## 10. Things we got wrong along the way

Recording these because the wrong turns were more instructive than the right ones.

| Belief | Reality |
|---|---|
| "Low confidence causes the wandering, so sharpen the policy." | **Backwards.** Argmax scored 19.3 — the worst of every configuration tested. Exploration was doing real work. |
| "The ordinal direction encoding is the bottleneck." | Only half true. It 4×'d generation-1 performance but bought ~nothing by generation 25 until weights could grow. The two fixes are complementary, not alternatives. |
| "90 s per board is too short." | Measured: **zero** agents out of 300 hit the timeout, at either 90 s or 180 s. Doubling it changed nothing. The killer was the 8 s stall timer. |
| "Adaptive mutation should ratchet up when stuck." | It must also come **back down**. Comparing against an all-time-best that is essentially never beaten made the rate climb monotonically to its cap and stay there; at 30% × 100 genes that is random search, and fitness actively *regressed* from 44.5 to 37 over 150 generations. |
| "Multiplicative fitness like snake's `lifetime² × 2^food`." | Right instinct, wrong transfer. Snake has no clock so lifetime is pure upside; pacman races a timer, so a lifetime term rewards dawdling. |

The general lesson: **measure before concluding.** Every one of these was resolved by
a headless A/B in under two minutes, and every one of them contradicted a
confident-sounding hypothesis.

---

## 11. Measured results

120 generations, population 300, seed 12345, 90 s boards:

| Gen | avg pellets | best | avg alive | wasted % | mutation % |
|---|---|---|---|---|---|
| 1 | 34.0 | 98 | 17.0 s | 17% | 5% |
| 2 | 36.6 | 109 | 18.3 s | 15% | 5% |
| 10 | 44.7 | 285 | 23.3 s | 13% | 5% |
| 20 | 54.2 | 202 | 21.7 s | 12% | 7% |
| 50 | 54.7 | 442 | 20.7 s | 15% | 12% |
| 70 | **79.8** | 459 | 25.6 s | 10% | 6% |
| 120 | 64.8 | 422 | 22.6 s | 14% | 12% |

Bests above 244 mean the agent **cleared a board and started another**. The
generation-1 → generation-2 jump reproduces the writeup's §4.1 exactly: wall-bumping
collapses first. Evolution's first move is not becoming good; it is eliminating
the worst.

Average pellets swing between generations because fitness is a **single sampled
episode**, which is closer to a lottery than a measurement. See §15.

---

## 12. Debug tooling

Two overlays in the settings panel, both off by default, both drawing the true BFS
shortest path one quad per tile — so **the number of quads is the distance value the
network sees**, and the first quad shows which neighbour the direction input points at.

| Toggle | Colour | Visualises |
|---|---|---|
| Pellet BFS path | Green | `pelletDist` (inputs 0) and `pelletDir` (1–2) |
| Ghost BFS path | Yellow → red | `ghostDist` (input 3) and `ghostDir` (4–5) |

Both follow the same rules as the perception vector, so what you see is what the net
gets: eyes-mode and housebound ghosts are skipped, and paths route through the
tunnel when shorter. The pellet path reads live pellet bits, so it re-targets the
instant pacman eats the pellet it was heading for.

There is also a **Level time** slider (30–180 s, live, persisted as `?level=`).

---

## 13. Files

| File | Responsibility |
|---|---|
| `maze.ts` | Maze grid, pellet masks, and `mazeGraph()` — the all-pairs BFS table |
| `pacman_buffers.ts` | Genome/agent layouts, fitness weights, GPU buffer creation |
| `pacman.wgsl.ts` | The simulation: perception, forward pass, sampling, movement, ghosts |
| `pacman_evolution.ts` | Generation driver, fitness, adaptive mutation, env jitter |
| `pacman_renderer.ts` | Sprite rendering plus the two BFS debug overlays |
| `pacman_model.ts` | Save/load genomes to JSON and localStorage |
| `main.ts` | Wiring: settings panel, HUD, network panel, main loop |
| `../../utils/ga.ts` | **Shared** tournament GA — selection, crossover, mutation, elitism |

### What is shared vs. pacman-local

**Shared** (`src/utils/ga.ts`): tournament selection, single-point crossover, the
reset+drift mutation operator and its `clamp`, and elitism. Any demo calling
`nextTournamentGeneration` gets the §9 fix by default.

**Pacman-local** (`pacman_evolution.ts`): the adaptive mutation *schedule*, the
fitness formula and its weights, the softmax temperature, and environment
randomisation. These are policy choices about *this* problem rather than generic GA
machinery, so they stay with the demo.

---

## 14. Verifying changes headlessly

The whole simulation runs outside the browser under Deno's WebGPU, which makes it
practical to A/B a change in about two minutes instead of squinting at a canvas.
Every number in this document came from that path.

```bash
# 1. Compile-check the WGSL. TypeScript will happily accept a broken shader,
#    because it is just a template literal — this catches it before the browser does.
deno run --unstable-webgpu --allow-read validate.js pacman.wgsl

# 2. Run real generations and print a fitness curve.
esbuild train.ts --bundle --format=esm --outfile=train.mjs
deno run --unstable-webgpu --allow-read train.mjs
```

Two traps worth knowing, both of which bit us:

- **WGSL reserved words.** `from` and `target` are reserved. TypeScript compiles them
  without complaint inside the shader template; only `createShaderModule` rejects
  them. The validator catches both.
- **Backticks inside WGSL comments** terminate the enclosing template literal. This
  produces a confusing TypeScript parse error hundreds of lines away from the actual
  comment.

When measuring, read the agent buffer **before** calling `checkAndEvolve()` — it
resets agent state for the next generation, so reading after it yields all zeros.

---

## 15. Known limits and next steps

**Fitness is a single sampled episode.** With stochastic actions, the same genome
scores very differently run to run — `best` swings between 242 and 459 across
adjacent generations. Tournament selection on a noisy signal selects partly on luck.
Averaging 2–3 episodes per genome is the highest-value next change.

**One life.** `START_LIVES = 0` ends the run at first contact, around 21 s in, so
agents rarely see a late board and get little gradient toward *finishing* one.

**Stall is still the dominant death** (~181/300), but it now happens at ~68 pellets
rather than ~40 — increasingly an agent that has cleared its local pocket and cannot
commit to crossing the map, which is a substantially harder problem than the
near-random walk it was before.

**The writeup's own ceiling applies here too** (§4.4): a purely reactive policy over
instantaneous perception has no long-range planning. It will not cross a patrolled
corridor to reach a distant power pellet. Breaking that needs either memory (feeding
the previous action or perception back in, §5.2) or a different learning method
entirely.
