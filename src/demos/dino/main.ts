import '../../style.css';
import { currentSettings, setupControls } from '../../gui/controls_gui';
import { NetworkPanel } from '../../ui/networkPanel';
import { requiredElement } from '../../utils/dom';
import { initializeWebGPU } from '../../webgpu/utils';
import { DinoEvolution, type BestDinoSnapshot } from './dino_evolution';
import { DinoRenderer } from './dino_renderer';

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

/** DOM overlay like the other demos: Score/Cleared/Alive top-left, Generation bottom-left. */
function createHud(): { update(best: BestDinoSnapshot, cleared: number, generation: number): void } {
  const stats = document.createElement('div');
  stats.className = 'hud hud-stats';
  const gen = document.createElement('div');
  gen.className = 'hud hud-generation';
  document.body.append(stats, gen);
  let lastStats = '';
  let lastGen = -1;
  return {
    update(best, cleared, generation) {
      // Score shown in the original's units (one point per ~7 frames).
      const text = `Score:   ${Math.floor(best.score / 7)}\nCleared: ${cleared}\nAlive:   ${best.aliveCount}\nFitness: ${best.fitness.toFixed(1)}`;
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

  const evolution = DinoEvolution.init(gpu.device, settings.population);
  const renderer = await DinoRenderer.create(canvas, gpu, evolution.buffers);
  const hud = createHud();
  const networkPanel = new NetworkPanel([6, 8, 1]);

  const notWired = (): void => {
    showMessage('Model save/load is not wired up for Chrome Dino yet.');
  };
  const controls = setupControls({
    onSaveModel: notWired,
    onLoadSavedBest: notWired,
    onLoadModelFile: notWired,
  });

  // Fixed 60 Hz sim ticks (the original's frame rate) x sim speed, independent of display Hz.
  let last = performance.now();
  let acc = 0;
  const loop = (now: number): void => {
    acc += (Math.min(now - last, 100) / 1000) * 60 * controls.speed;
    last = now;
    const steps = Math.min(Math.floor(acc), 240);
    acc -= steps;
    if (steps > 0) evolution.substeps(steps);
    void evolution.checkAndEvolve();
    void evolution.readBestDinoState().then((best) => {
      renderer.setBestIndex(best.index);
      hud.update(best, evolution.cleared, evolution.generation);
      networkPanel.draw(evolution.genomeAt(best.index));
    });
    renderer.render(evolution.obstacle, evolution.groundScroll, evolution.runFrame);
    requestAnimationFrame(loop);
  };
  loop(performance.now());
}

main().catch(showError);
