import type { BestCarSnapshot } from './evolution';

/** DOM overlay matching the Unity HUD: Turn/Engine/Eval top-left, Generation bottom-left. */
export class Hud {
  private readonly stats: HTMLDivElement;
  private readonly generation: HTMLDivElement;
  private lastStats = '';
  private lastGeneration = -1;

  constructor() {
    this.stats = document.createElement('div');
    this.stats.className = 'hud hud-stats';
    this.generation = document.createElement('div');
    this.generation.className = 'hud hud-generation';
    document.body.append(this.stats, this.generation);
  }

  update(best: BestCarSnapshot, generation: number): void {
    const text = `Population:   ${best.aliveCount}\nTurn:   ${best.turn.toFixed(5)}\nEngine: ${best.engine.toFixed(5)}\nEval:   ${best.fitness.toFixed(5)}`;
    if (text !== this.lastStats) {
      this.stats.textContent = text;
      this.lastStats = text;
    }
    if (generation !== this.lastGeneration) {
      this.generation.textContent = `Generation: ${generation}`;
      this.lastGeneration = generation;
    }
  }
}
