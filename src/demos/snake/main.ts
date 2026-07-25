import '../../style.css';
import { currentSettings, persistMode, setupControls, updateSetting } from '../../gui/controls_gui';
import { NetworkPanel } from '../../ui/networkPanel';
import { requiredElement } from '../../utils/dom';
import { initializeWebGPU } from '../../webgpu/utils';
import { loadSavedSnakeBest, loadSnakeModelFile, loadSnakeTestModel, saveSnakeModel, saveSnakeTestModel } from './model';
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
      const text = `Apples:  ${best.apples}\nLength:  ${best.length}\nMoves:   ${best.moves}\nLife:    ${best.life}\nAlive:   ${best.aliveCount}`;
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
  let isTest = settings.mode === 'Test';
  let noModelWarning = false;
  const testModel = loadSnakeTestModel();
  if (isTest && !testModel) {
    isTest = false;
    noModelWarning = true;
    persistMode('Train');
  }

  const evolution = SnakeEvolution.init(gpu.device, isTest ? 1 : settings.population);
  if (isTest) evolution.injectBest(testModel!);
  const renderer = new SnakeRenderer(canvas, gpu, evolution.buffers);
  const hud = createHud();
  const networkPanel = new NetworkPanel([24, 16, 16, 4]);
  if (noModelWarning) {
    showMessage('Test mode needs a model - starting in Train mode. Use "Load model file" to test one.');
    setTimeout(() => message.classList.remove('visible'), 6000);
  }

  const controls = setupControls({
    onSaveModel: () => {
      saveSnakeModel(evolution.bestGenome(), evolution.bestMeta());
    },
    onLoadSavedBest: () => {
      const saved = loadSavedSnakeBest();
      if (!saved) {
        showMessage('No saved best Snake model yet.');
        return;
      }
      saveSnakeTestModel(saved.weights);
      updateSetting('mode', 'Test');
    },
    onLoadModelFile: (file) => {
      loadSnakeModelFile(file)
        .then((weights) => {
          saveSnakeTestModel(weights);
          updateSetting('mode', 'Test');
        })
        .catch((error: unknown) => alert(error instanceof Error ? error.message : String(error)));
    },
  });

  // Turn-based: SnakeAI runs at 100 moves/sec, multiplied by sim speed.
  let last = performance.now();
  let acc = 0;
  const loop = (now: number): void => {
    acc += (Math.min(now - last, 100) / 1000) * 100 * controls.speed;
    last = now;
    const steps = Math.min(Math.floor(acc), 600);
    acc -= steps;
    if (steps > 0) evolution.substeps(steps);
    if (isTest) void evolution.restartDisplayIfDead();
    else void evolution.checkAndEvolve();
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
