import '../../style.css';
import { currentSettings, persistMode, updateSetting } from '../../gui/controls_gui';
import { createDemoSettingsPanel } from '../../ui/demoSettingsPanel';
import { NetworkPanel } from '../../ui/networkPanel';
import { requiredElement } from '../../utils/dom';
import { initializeWebGPU } from '../../webgpu/utils';
import { Evolution, type BestCarSnapshot } from './evolution';
import { Hud } from './hud';
import { loadModelFile, loadSavedBest, loadTestModel, saveModel, saveTestModel } from './model';
import { TOPOLOGY } from './network';
import { Renderer } from './renderer';
import { runSelftest } from './selftest';
import { loadTrack } from './track';

/** Available tracks (public/tracks/<name>.json); adding one is a line here + the JSON file. */
export const TRACKS = ['track1', 'track2', 'track3', 'track4', 'practice'];

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
  document.body.classList.add('snake-layout');
  const gpu = await initializeWebGPU(canvas);

  // ?selftest runs the CPU/GPU parity asserts and reports to console + #message.
  if (new URLSearchParams(location.search).has('selftest')) {
    const result = await runSelftest(gpu.device);
    showMessage(result.pass ? 'SELFTEST PASS' : `SELFTEST FAIL\n${result.failures.join('\n')}`);
    return;
  }

  const settings = currentSettings(TRACKS);
  let isTest = settings.mode === 'Test';
  let noModelWarning = false;
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
  const networkPanel = new NetworkPanel(TOPOLOGY, {
    variant: 'snake',
    outputLabels: ['TURN', 'ENGINE'],
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  });
  let lastBest: BestCarSnapshot | null = null;

  const controls = createDemoSettingsPanel(
    settings,
    {
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
      onLoadModelFile: (file) => {
        loadModelFile(file)
          .then((weights) => {
            saveTestModel(weights);
            updateSetting('mode', 'Test');
          })
          .catch((error: unknown) => alert(error instanceof Error ? error.message : String(error)));
      },
    },
    { tracks: TRACKS },
  );

  const loop = (): void => {
    evolution.substeps(controls.speed);
    void evolution.checkAndEvolve(isTest);
    void evolution.readBestCarState().then((best) => {
      lastBest = best;
      renderer.follow(best.x, best.y, best.index, false);
      hud.update(best, evolution.generation);
      networkPanel.draw(evolution.genomeAt(best.index), {
        stats: [
          ['GEN', evolution.generation],
          ['FITNESS', best.fitness.toFixed(3)],
          ['POP LEFT', best.aliveCount],
          ['TURN', best.turn.toFixed(2)],
          ['ENGINE', best.engine.toFixed(2)],
          ['TRACK', track.name],
        ],
      });
    });
    renderer.render();
    requestAnimationFrame(loop);
  };
  loop();
}

main().catch(showError);
