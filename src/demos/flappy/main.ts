import '../../style.css';
import { currentSettings, setupControls } from '../../gui/controls_gui';
import { Hud } from '../../ui/hud';
import { requiredElement } from '../../utils/dom';
import { initializeWebGPU } from '../../webgpu/utils';
import { FlappyEvolution } from './flappy_evolution';
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

async function main(): Promise<void> {
  const gpu = await initializeWebGPU(canvas);
  const settings = currentSettings();

  const evolution = FlappyEvolution.init(gpu.device, settings.population);
  const renderer = new FlappyRenderer(canvas, gpu, evolution.buffers);
  const hud = new Hud();

  const controls = setupControls({
    onSaveModel: () => {
      showMessage('Saved best Flappy Bird model to browser storage!');
    },
    onLoadSavedBest: () => {
      showMessage('Loaded saved Flappy Bird model.');
    },
  });

  const loop = (): void => {
    evolution.substeps(controls.speed);
    void evolution.checkAndEvolve();
    void evolution.readBestBirdState().then((best) => {
      renderer.setBestIndex(best.index);
      hud.update(
        {
          index: best.index,
          x: best.x,
          y: best.y,
          angleDeg: 0,
          turn: 0,
          engine: 0,
          fitness: best.fitness,
          alive: best.alive,
          aliveCount: 0,
        },
        evolution.generation,
      );
    });

    renderer.render(evolution.pipesList.length);
    requestAnimationFrame(loop);
  };
  loop();
}

main().catch(showError);
