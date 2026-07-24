import './style.css';
import { loadSavedBest, loadTestModel, saveModel } from './ai/model';
import { runSelftest } from './ai/selftest';
import { Evolution, type BestCarSnapshot } from './gpu/evolution';
import { currentSettings, persistMode, setupControls } from './gui/controls_gui';
import { Renderer } from './renderer/renderer';
import { loadTrack } from './sim/track';
import { Hud } from './ui/hud';
import { NetworkPanel } from './ui/networkPanel';
import { requiredElement } from './utils/dom';
import { initializeWebGPU } from './webgpu/utils';

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

  // ?selftest runs the CPU/GPU parity asserts and reports to console + #message.
  if (new URLSearchParams(location.search).has('selftest')) {
    const result = await runSelftest(gpu.device);
    showMessage(result.pass ? 'SELFTEST PASS' : `SELFTEST FAIL\n${result.failures.join('\n')}`);
    return;
  }

  const settings = currentSettings();
  let isTest = settings.mode === 'Test';
  let noModelWarning = false;
  // Test mode without a model used to throw here — and since mode is persisted,
  // the app could never boot again to let the user switch back. Fall back to Train.
  if (isTest && !loadTestModel()) {
    isTest = false;
    noModelWarning = true;
    persistMode('Train');
  }
  const track = await loadTrack(`/tracks/${settings.track}.json`);
  const evolution = Evolution.init(gpu.device, track, isTest ? 1 : settings.population, 1);
  if (isTest) evolution.injectBest(loadTestModel()!);
  if (noModelWarning) {
    showMessage('Test mode needs a model — starting in Train mode. Use "Load model file" to test one.');
    setTimeout(() => message.classList.remove('visible'), 6000);
  }

  const renderer = new Renderer(canvas, gpu, track, evolution.simBuffers);
  const hud = new Hud();
  const networkPanel = new NetworkPanel();
  let lastBest: BestCarSnapshot | null = null;

  const controls = setupControls({
    onSaveModel: () => {
      if (!lastBest) return;
      saveModel(evolution.genomeAt(lastBest.index), {
        track: track.name,
        generation: evolution.generation,
        eval: lastBest.fitness,
      });
    },
    onLoadSavedBest: () => {
      const saved = loadSavedBest(track.name);
      if (saved) evolution.injectBest(saved.weights);
      else showMessage(`No saved best for track "${track.name}" yet.`);
    },
  });

  renderer.onPanStart = () => controls.setFollow(false);

  const loop = (): void => {
    evolution.substeps(controls.speed);
    void evolution.isGenerationOver().then((over) => {
      if (!over) return;
      // Test mode: GA skipped, the same car just reruns the track.
      if (isTest) evolution.resetStates();
      else void evolution.evolve();
    });
    void evolution.readBestCarState().then((best) => {
      lastBest = best;
      renderer.follow(best.x, best.y, best.index, controls.followCam);
      hud.update(best, evolution.generation);
      networkPanel.draw(evolution.genomeAt(best.index));
    });
    renderer.render();
    requestAnimationFrame(loop);
  };
  loop();
}

main().catch(showError);
