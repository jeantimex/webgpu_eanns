import '../../style.css';
import { currentSettings, setupControls } from '../../gui/controls_gui';
import { NetworkPanel } from '../../ui/networkPanel';
import { requiredElement } from '../../utils/dom';
import { initializeWebGPU } from '../../webgpu/utils';
import { SnakeEvolution, type BestSnakeSnapshot } from './snake_evolution';
import { SnakeRenderer } from './snake_renderer';

const canvas = requiredElement<HTMLCanvasElement>('#webgpu-canvas');
const message = requiredElement<HTMLDivElement>('#message');

function showMessage(text: string): void {
  console.log(text);
  message.textContent = text;
  message.classList.add('visible');
}

function showError(error: unknown): void {
  console.error(error);
  showMessage(error instanceof Error ? error.message : 'Unable to start WebGPU.');
}

/** DOM overlay: Apples/Length/Moves top-left, Generation bottom-left. */
function createHud(): { update(best: BestSnakeSnapshot, generation: number): void } {
  // Dark ink on an LCD-colored chip: readable over both the light board and
  // the dark page margin.
  const chip = (el: HTMLDivElement): void => {
    el.style.background = 'rgba(170, 176, 155, 0.92)';
    el.style.color = '#1e1e1e';
    el.style.padding = '0.35rem 0.6rem';
    el.style.borderRadius = '0.4rem';
  };
  const stats = document.createElement('div');
  stats.className = 'hud hud-stats';
  chip(stats);
  const gen = document.createElement('div');
  gen.className = 'hud hud-generation';
  chip(gen);
  document.body.append(stats, gen);
  let lastStats = '';
  let lastGen = -1;
  return {
    update(best, generation) {
      const text = `Apples:  ${best.apples}\nLength:  ${best.length}\nMoves:   ${best.moves}\nScore:   ${best.score.toFixed(1)}\nAlive:   ${best.aliveCount}`;
      if (text !== lastStats) {
        stats.textContent = text;
        lastStats = text;
      }
      if (generation !== lastGen) {
        gen.textContent = `Generation: ${generation}`;
        lastGen = generation;
      }
    },
  };
}

async function main(): Promise<void> {
  const gpu = await initializeWebGPU(canvas);
  const settings = currentSettings();

  const evolution = SnakeEvolution.init(gpu.device, settings.population);
  const renderer = new SnakeRenderer(canvas, gpu, evolution.buffers);
  const hud = createHud();
  const networkPanel = new NetworkPanel([16, 12, 3]);

  const notWired = (): void => {
    showMessage('Model save/load is not wired up for Snake yet.');
  };
  const controls = setupControls({
    onSaveModel: notWired,
    onLoadSavedBest: notWired,
    onLoadModelFile: notWired,
  });

  // Turn-based: 10 moves/sec base rate x sim speed, independent of display Hz.
  let last = performance.now();
  let acc = 0;
  const loop = (now: number): void => {
    acc += (Math.min(now - last, 100) / 1000) * 10 * controls.speed;
    last = now;
    const steps = Math.min(Math.floor(acc), 600);
    acc -= steps;
    if (steps > 0) evolution.substeps(steps);
    void evolution.checkAndEvolve();
    void evolution.readBestAgentState().then((best) => {
      renderer.setBestIndex(best.index);
      hud.update(best, evolution.generation);
      networkPanel.draw(evolution.genomeAt(best.index));
    });
    renderer.render();
    requestAnimationFrame(loop);
  };
  loop(performance.now());
}

main().catch(showError);
