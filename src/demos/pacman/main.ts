import '../../style.css';
import { currentSettings, persistMode, updateSetting } from '../../gui/controls_gui';
import { createDemoSettingsPanel } from '../../ui/demoSettingsPanel';
import { NetworkPanel } from '../../ui/networkPanel';
import { requiredElement } from '../../utils/dom';
import { initializeWebGPU } from '../../webgpu/utils';
import { PacmanEvolution, type BestAgentSnapshot } from './pacman_evolution';
import { loadPacmanModelFile, loadPacmanTestModel, loadSavedPacmanBest, savePacmanModel, savePacmanTestModel } from './pacman_model';
import { PacmanRenderer } from './pacman_renderer';

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

/** DOM overlay: Score/Lives/Dots/Level top-left, Generation bottom-left. */
function createHud(): { update(best: BestAgentSnapshot, generation: number): void } {
  const stats = document.createElement('div');
  stats.className = 'hud hud-stats';
  stats.style.color = '#f1f5f9'; // arcade-black background
  const gen = document.createElement('div');
  gen.className = 'hud hud-generation';
  gen.style.color = '#f1f5f9';
  document.body.append(stats, gen);
  let lastStats = '';
  let lastGen = -1;
  return {
    update(best, generation) {
      const text = `Score:   ${best.score}\nDots:    ${best.dotsLeft}\nLevel:   ${best.level}\nPlaying: ${best.aliveCount}`;
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
  let isTest = settings.mode === 'Test';
  let noModelWarning = false;
  const testModel = loadPacmanTestModel();
  if (isTest && !testModel) {
    isTest = false;
    noModelWarning = true;
    persistMode('Train');
  }

  const evolution = PacmanEvolution.init(gpu.device, isTest ? 1 : settings.population);
  if (isTest) evolution.injectBest(testModel!);
  const renderer = await PacmanRenderer.create(canvas, gpu, evolution.buffers);
  const hud = createHud();
  const networkPanel = new NetworkPanel(PacmanEvolution.topology, {
    variant: 'snake',
    outputLabels: ['UP', 'RIGHT', 'DOWN', 'LEFT'],
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  });

  const isPlayMode = settings.mode === 'Play';
  evolution.setPlayMode(isPlayMode);
  // Play mode starts frozen at the initial position; the first arrow key starts the game.
  let waiting = isPlayMode;

  if (isPlayMode) {
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        waiting = true;
        evolution.resetDisplayAgent();
        return;
      }
      let dir = -1;
      if (e.key === 'ArrowUp' || e.code === 'KeyW') dir = 0;
      else if (e.key === 'ArrowRight' || e.code === 'KeyD') dir = 3;
      else if (e.key === 'ArrowDown' || e.code === 'KeyS') dir = 1;
      else if (e.key === 'ArrowLeft' || e.code === 'KeyA') dir = 2;

      if (dir >= 0) {
        e.preventDefault();
        waiting = false;
        evolution.setPlayerDesiredDir(dir);
        void evolution.readBestAgentState().then((best) => {
          if (best.gameOver) evolution.resetDisplayAgent();
        });
      }
    });
  }
  if (noModelWarning) {
    showMessage('Test mode needs a model - starting in Train mode. Use "Load model file" to test one.');
    setTimeout(() => message.classList.remove('visible'), 6000);
  }

  const controls = createDemoSettingsPanel(settings, {
    onSaveModel: () => {
      savePacmanModel(evolution.displayGenome(), evolution.bestMeta());
    },
    onLoadSavedBest: () => {
      const saved = loadSavedPacmanBest();
      if (!saved) {
        showMessage('No saved best model found for Pac-Man.');
        return;
      }
      savePacmanTestModel(saved.weights);
      updateSetting('mode', 'Test');
    },
    onLoadModelFile: async (file) => {
      try {
        const weights = await loadPacmanModelFile(file);
        savePacmanTestModel(weights);
        updateSetting('mode', 'Test');
      } catch (err) {
        showError(err);
      }
    },
  });

  // Fixed 60 Hz sim ticks (the source engine's logic rate) x sim speed.
  let last = performance.now();
  let acc = 0;
  const startTime = last;
  const loop = (now: number): void => {
    acc += (Math.min(now - last, 100) / 1000) * 60 * controls.speed;
    last = now;
    const steps = Math.min(Math.floor(acc), 240);
    acc -= steps;
    if (steps > 0 && !waiting) evolution.substeps(steps);
    if (isTest) {
      void evolution.restartTestAgentIfDead();
    } else if (!isPlayMode) {
      void evolution.checkAndEvolve();
    }
    void evolution.readBestAgentState().then((best) => {
      renderer.setBestIndex(best.index);
      hud.update(best, isPlayMode || isTest ? 0 : evolution.generation);
      networkPanel.draw(evolution.genomeAt(best.index), {
        stats: isPlayMode
          ? [
              ['MODE', 'PLAY'],
              ['SCORE', best.score],
              ['DOTS', best.dotsLeft],
              ['LEVEL', best.level],
              ['STATUS', waiting ? 'READY (Press Arrow Key)' : best.gameOver ? 'GAME OVER (Press Arrow Key)' : 'PLAYING'],
            ]
          : isTest
            ? [
                ['MODE', 'TEST'],
                ['SCORE', best.score],
                ['DOTS', best.dotsLeft],
                ['LEVEL', best.level],
                ['STATUS', best.gameOver ? 'RESTARTING' : 'RUNNING'],
              ]
          : [
              ['GEN', evolution.generation],
              ['SCORE', best.score],
              ['BEST SCORE', best.bestScore],
              ['BEST LEVEL', best.bestLevel],
              ['BEST GEN', best.bestGeneration],
              ['DOTS', best.dotsLeft],
              ['LEVEL', best.level],
              ['POP LEFT', best.aliveCount],
            ],
      });
    });
    renderer.render((now - startTime) / 1000);
    requestAnimationFrame(loop);
  };
  loop(performance.now());
}

main().catch(showError);
