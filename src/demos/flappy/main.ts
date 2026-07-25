import '../../style.css';
import { currentSettings } from '../../gui/controls_gui';
import { createDemoSettingsPanel } from '../../ui/demoSettingsPanel';
import { NetworkPanel } from '../../ui/networkPanel';
import { requiredElement } from '../../utils/dom';
import { initializeWebGPU } from '../../webgpu/utils';
import { FlappyEvolution, type BestBirdSnapshot } from './flappy_evolution';
import { FlappyRenderer } from './flappy_renderer';

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

/** DOM overlay like the track HUD: Score/Alive top-left, Generation bottom-left. */
function createHud(): { update(best: BestBirdSnapshot, generation: number): void } {
  const stats = document.createElement('div');
  stats.className = 'hud hud-stats';
  const gen = document.createElement('div');
  gen.className = 'hud hud-generation';
  document.body.append(stats, gen);
  let lastStats = '';
  let lastGen = -1;
  return {
    update(best, generation) {
      const text = `Pipes:   ${best.pipes}\nAlive:   ${best.aliveCount}\nVelY:    ${best.velY.toFixed(3)}\nFitness: ${best.fitness.toFixed(1)}`;
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
  document.body.classList.add('snake-layout');
  const gpu = await initializeWebGPU(canvas);
  const settings = currentSettings();

  const evolution = FlappyEvolution.init(gpu.device, settings.population);
  const renderer = await FlappyRenderer.create(canvas, gpu, evolution.buffers);
  const hud = createHud();
  const networkPanel = new NetworkPanel([5, 8, 1], {
    variant: 'snake',
    outputLabels: ['FLAP'],
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  });

  const notWired = (): void => {
    showMessage('Model save/load is not wired up for Flappy Bird yet.');
  };
  const controls = createDemoSettingsPanel(settings, {
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
    void evolution.readBestBirdState().then((best) => {
      renderer.setBestIndex(best.index);
      hud.update(best, evolution.generation);
      networkPanel.draw(evolution.genomeAt(best.index), {
        stats: [
          ['GEN', evolution.generation],
          ['PIPES', best.pipes],
          ['POP LEFT', best.aliveCount],
          ['FITNESS', best.fitness.toFixed(1)],
          ['SCORE', best.score],
          ['VEL Y', best.velY.toFixed(2)],
        ],
      });
    });
    renderer.render(evolution.pipesList.length);
    requestAnimationFrame(loop);
  };
  loop(performance.now());
}

main().catch(showError);
