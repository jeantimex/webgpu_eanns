# Track — Evolved Self-Driving Cars on the GPU

A population of cars learns to drive a closed circuit, evolved by a genetic
algorithm. One GPU thread per car, all racing the same track from the same start
line. This is a port of the well-known Unity "self-driving car with a genetic
algorithm" project, and it follows that implementation closely — sensor layout,
network shape, activation, fitness and selection are all deliberate matches.

It is the most mechanically detailed demo in the repo, and the only one whose
network drives **continuous** outputs rather than picking a discrete action.

---

## 1. At a glance

| Piece | Value |
|---|---|
| Inputs | 5 raw sensor distances |
| Network | `5 → 4 → 3 → 2`, SoftSign on **every** layer |
| Genome | 47 weights |
| Outputs | 2 continuous: steering, throttle |
| Selection | remainder stochastic sampling + BLX/uniform crossover |
| Fitness | fraction of the track completed, 0 … 1 |
| Car state | 12 floats (48 bytes) |

---

## 2. Inputs — five raycasts

Five sensors cast rays against the track's wall segments and report **raw
distances** (not normalised), capped at 25 units:

| Sensor | Angle | Origin (car-local) |
|---|---|---|
| 0 | −45° | (−0.3, 0.54) |
| 1 | −20.6° | (−0.3, 0.84) |
| 2 | 0° (straight ahead) | (0.0, 0.84) |
| 3 | +20.6° | (0.3, 0.84) |
| 4 | +45° | (0.3, 0.54) |

Two details matter:

- **The origins are spread across the car's nose, not all at its centre.** The
  ±45° sensors sit further back on the flanks, so they see a wall the car is about
  to clip with its corner rather than one directly ahead. That is what lets a car
  learn to hug an apex without scraping.
- **Distances are fed in raw.** Most demos here normalise inputs to [0, 1]; this
  one does not, because SoftSign squashes them anyway (see below).

The raw distances are also written to a `sensors` buffer so the renderer can draw
the sensor-hit crosses — useful when debugging why a car turns where it does.

---

## 3. Network and genome

`5 → 4 → 3 → 2` with **SoftSign on every layer, including the output**:

```
softSign(x) = x / (1 + |x|)
```

SoftSign saturates toward ±1 but far more gently than tanh, so a large raw sensor
distance does not slam the neuron to its limit. It is what makes feeding unnormalised
distances workable.

Genome is 47 weights, laid out layer by layer, each layer an `(in+1) × out`
row-major matrix whose **last row is the bias** (constant input 1.0):

| Layer | Shape | Weights + bias | Count |
|---|---|---|---|
| 1 | 5 → 4 | (5+1) × 4 | 24 |
| 2 | 4 → 3 | (4+1) × 3 | 15 |
| 3 | 3 → 2 | (3+1) × 2 | 8 |
| **Total** | | | **47** |

`network.ts` contains a CPU implementation of the identical forward pass, which
`selftest.ts` uses to verify the GPU path agrees.

---

## 4. Outputs and vehicle physics

Two continuous outputs, both clamped to [−1, 1]:

| Output | Meaning |
|---|---|
| 0 | steering: −1 full left … +1 full right |
| 1 | throttle: −1 full brake/reverse … +1 full forward |

This is the only demo whose network produces a **continuous control signal**
rather than choosing among discrete actions — there is no argmax and no sampling.

| Constant | Value |
|---|---|
| Max velocity | 20 |
| Acceleration | 8 |
| Brake acceleration | 24 |
| Turn speed | 180 °/s |
| Friction (coasting) | 2 |
| Car half-width | 0.5 |

Throttle and brake are asymmetric: braking (24) is three times stronger than
accelerating (8), and the sign of the output relative to current velocity decides
whether it acts as brake or reverse. Releasing the throttle entirely applies
friction instead.

---

## 5. Fitness — fraction of track completed

The track carries a chain of checkpoints, each with a `reward` and an accumulated
`accReward`. A car's fitness is how far around the circuit it has actually got:

```wgsl
complete = max((cp.distToPrev - dist) / cp.distToPrev, 0)
fitness  = checkpoints[idx - 1].accReward + complete * cp.reward
```

So fitness is **continuous, not stepwise** — partial progress toward the next
checkpoint counts. That gives selection a usable gradient from the very first
generation, when no car has reached checkpoint 1 yet. Finishing the lap gives
exactly `1.0`.

Capturing a checkpoint uses a `while` loop rather than an `if`, so a fast car that
passes several within one frame is credited for all of them.

### Termination

| Rule | Value |
|---|---|
| Wall contact | instant death |
| No checkpoint for | 7 seconds → death |

The 7-second timeout is what stops a car scoring by driving in circles in an open
area, and it is why fitness needs no explicit anti-idling penalty.

**One deliberately harsh detail:** on crashing, partial credit for the in-progress
segment is stripped and fitness reverts to the last fully captured checkpoint's
`accReward`. Without that, a car could gain by flooring it at a wall — it would
bank the partial progress before dying. Removing it makes a suicidal wall hit
strictly worse than stopping short.

---

## 6. Selection — the most elaborate GA in the repo

This demo does **not** use `src/utils/ga.ts`; it has its own `ga.ts`, matching the
Unity original's configuration.

**1. Remainder stochastic sampling.** Relative fitness is `evaluation / average`.
Each genome gets `floor(relFitness)` guaranteed copies in an intermediate pool,
plus one more with probability equal to the fractional remainder. Compared with
plain roulette this is lower-variance — a genome at 2.4× average reliably gets 2
copies plus a 40% shot at a third, rather than being at the mercy of repeated
random draws.

**2. Crossover**, per weight:
- 60% **blend (BLX)**: `α ∈ [−0.1, 1.1]`, offspring = `α·p1 + (1−α)·p2`. The range
  extending slightly beyond [0, 1] lets children land *outside* the interval
  spanned by their parents, which is what keeps the population from collapsing
  inward.
- 40% **uniform**: swap the parents' weights outright.

**3. Dynamic elite scaling.** This is the unusual part:

| Situation | Elites | Mutation rate | Mutation amount |
|---|---|---|---|
| Still learning (maxFitness < 0.95) | 2 | 0.20 | ±1.2 |
| Track being completed (≥ 0.95) | **50% of population** | 0.05 | ±0.2 |

Once cars start finishing, the GA switches from exploration to refinement: it
preserves half the population unmutated and drops mutation to a gentle polish.
Visually this is why a large pack completes the lap together rather than a lone
survivor — and functionally it stops a solved policy being churned away by
mutation, which is exactly the failure the pacman demo hit when its adaptive
mutation ratcheted up and never came back down.

---

## 7. Files

| File | Responsibility |
|---|---|
| `track.ts` | Track definitions, wall segments, checkpoint table |
| `network.ts` | Topology, SoftSign, CPU forward pass |
| `ga.ts` | **Demo-specific** GA: sampling, BLX crossover, dynamic elites |
| `buffers.ts` | Car/genome layout, GPU buffers |
| `sim.wgsl.ts` | Raycasts, forward pass, physics, checkpoints, fitness |
| `car.ts` | Car geometry helpers |
| `renderer.ts` | Track, cars and sensor-hit rendering |
| `hud.ts` | On-screen stats |
| `model.ts` | Save/load genomes |
| `selftest.ts` | Verifies the GPU forward pass against the CPU one |
| `main.ts` | Settings panel, track selection, main loop |

---

## 8. Notes and possible improvements

- **`selftest.ts` is worth copying.** Checking the GPU forward pass against a CPU
  reference catches genome-layout drift immediately — the class of bug that is
  otherwise silent, because a mis-indexed weight still produces plausible driving.
- **The dynamic elite scaling is the good idea here.** Switching from exploration
  to refinement once the problem is solved is general, and none of the other demos
  do it.
- **This GA is not shared.** Remainder stochastic sampling and BLX crossover are
  both general techniques that the other demos could use; moving them into
  `src/utils/ga.ts` alongside the roulette, crossover and tournament variants
  would make them available everywhere.
- **Fitness is a single episode**, and the track is fixed, so there is no
  environment noise to average over — this demo does not need the multi-episode
  machinery in `src/utils/evaluation.ts`. It does mean cars can overfit one
  circuit; training across several tracks would produce a more general driver.
- **No memory.** The car cannot tell whether a wall is approaching or receding,
  only where it currently is.
