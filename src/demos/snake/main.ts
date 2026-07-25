import '../../style.css';
import { currentSettings, persistMode, updateSetting, updateUrlParamLive, type Settings } from '../../gui/controls_gui';
import { NetworkPanel } from '../../ui/networkPanel';
import { requiredElement } from '../../utils/dom';
import { initializeWebGPU } from '../../webgpu/utils';
import { loadSavedSnakeBest, loadSnakeModelFile, loadSnakeTestModel, saveSnakeModel, saveSnakeTestModel } from './model';
import { SnakeEvolution, type BestSnakeSnapshot } from './snake_evolution';
import { SnakeRenderer } from './snake_renderer';

const canvas = requiredElement<HTMLCanvasElement>('#webgpu-canvas');
const message = requiredElement<HTMLDivElement>('#message');

interface SnakeSettingsActions {
  onSaveModel(): void;
  onLoadSavedBest(): void;
  onLoadModelFile(file: File): void;
}

function showMessage(text: string): void {
  console.log(text);
  message.textContent = text;
  message.classList.add('visible');
}

function showError(error: unknown): void {
  console.error(error);
  showMessage(error instanceof Error ? error.message : 'Unable to start WebGPU.');
}

function createFpsDisplay(): { update(now: number): void; setVisible(visible: boolean): void } {
  const el = document.createElement('div');
  el.className = 'snake-fps';
  document.body.append(el);
  let last = performance.now();
  let frames = 0;
  return {
    update(now) {
      frames++;
      if (now - last < 500) return;
      el.textContent = `${Math.round((frames * 1000) / (now - last))} FPS`;
      frames = 0;
      last = now;
    },
    setVisible(visible) {
      el.classList.toggle('visible', visible);
    },
  };
}

function createSnakeSettingsPanel(
  settings: Settings,
  actions: SnakeSettingsActions,
): { speed: number; showFps: boolean } {
  const controls = { speed: settings.speed, showFps: false };
  const panel = document.createElement('aside');
  panel.className = 'snake-settings-panel';

  const toggle = document.createElement('button');
  toggle.className = 'snake-settings-toggle';
  toggle.type = 'button';
  toggle.title = 'Collapse settings';
  const toggleIcon = document.createElement('span');
  toggleIcon.className = 'material-chevron chevron-right';
  toggle.append(toggleIcon);
  toggle.addEventListener('click', () => {
    panel.classList.toggle('collapsed');
    const collapsed = panel.classList.contains('collapsed');
    document.body.classList.toggle('snake-settings-collapsed', collapsed);
    toggleIcon.className = `material-chevron ${collapsed ? 'chevron-left' : 'chevron-right'}`;
    toggle.title = collapsed ? 'Expand settings' : 'Collapse settings';
  });

  const resizeHandle = document.createElement('div');
  resizeHandle.className = 'snake-settings-resize';
  resizeHandle.title = 'Resize settings';
  resizeHandle.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    resizeHandle.setPointerCapture(event.pointerId);
    document.body.classList.add('snake-resizing');
  });
  resizeHandle.addEventListener('pointermove', (event) => {
    if (!resizeHandle.hasPointerCapture(event.pointerId)) return;
    const width = Math.max(220, Math.min(window.innerWidth * 0.42, window.innerWidth - event.clientX));
    document.body.style.setProperty('--snake-settings-open-width', `${Math.round(width)}px`);
  });
  const endResize = (event: PointerEvent): void => {
    if (resizeHandle.hasPointerCapture(event.pointerId)) resizeHandle.releasePointerCapture(event.pointerId);
    document.body.classList.remove('snake-resizing');
  };
  resizeHandle.addEventListener('pointerup', endResize);
  resizeHandle.addEventListener('pointercancel', endResize);

  const content = document.createElement('div');
  content.className = 'snake-settings-content';
  const header = document.createElement('div');
  header.className = 'snake-settings-header';
  const title = document.createElement('h2');
  title.textContent = 'Settings';
  header.append(toggle, title);

  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.json,.txt,.csv';
  fileInput.style.display = 'none';
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    fileInput.value = '';
    if (file) actions.onLoadModelFile(file);
  });

  const field = (labelText: string, control: HTMLElement): HTMLLabelElement => {
    const label = document.createElement('label');
    label.className = 'snake-settings-field';
    const span = document.createElement('span');
    span.textContent = labelText;
    label.append(span, control);
    return label;
  };

  const mode = document.createElement('select');
  for (const value of ['Train', 'Test'] as const) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    option.selected = settings.mode === value;
    mode.append(option);
  }
  mode.addEventListener('change', () => updateSetting('mode', mode.value));

  const population = document.createElement('input');
  population.type = 'number';
  population.min = '3';
  population.max = '10000';
  population.step = '1';
  population.value = String(settings.population);
  population.addEventListener('change', () => updateSetting('pop', Math.max(3, Math.min(10000, Number(population.value) || settings.population))));

  const speed = document.createElement('input');
  speed.type = 'range';
  speed.min = '1';
  speed.max = '100';
  speed.step = '1';
  speed.value = String(settings.speed);
  const speedValue = document.createElement('output');
  speedValue.textContent = String(settings.speed);
  speed.addEventListener('input', () => {
    controls.speed = Number(speed.value);
    speedValue.textContent = speed.value;
    updateUrlParamLive('speed', speed.value);
  });
  const speedWrap = document.createElement('div');
  speedWrap.className = 'snake-settings-range';
  speedWrap.append(speed, speedValue);

  const showFps = document.createElement('input');
  showFps.type = 'checkbox';
  showFps.checked = controls.showFps;
  showFps.addEventListener('change', () => {
    controls.showFps = showFps.checked;
  });
  const fpsField = field('Show FPS', showFps);
  fpsField.classList.add('snake-settings-check');

  const buttons = document.createElement('div');
  buttons.className = 'snake-settings-buttons';
  const button = (text: string, onClick: () => void): HTMLButtonElement => {
    const el = document.createElement('button');
    el.type = 'button';
    el.textContent = text;
    el.addEventListener('click', onClick);
    return el;
  };
  buttons.append(
    button('Save best model', actions.onSaveModel),
    button('Load model file', () => fileInput.click()),
    button('Load saved best', actions.onLoadSavedBest),
  );

  content.append(header, field('Mode', mode), field('Population', population), field('Sim speed', speedWrap), fpsField, buttons, fileInput);
  panel.append(resizeHandle, content);
  document.body.append(panel);
  return controls;
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
  document.body.classList.add('snake-layout');
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
  const fpsDisplay = createFpsDisplay();
  const networkPanel = new NetworkPanel([24, 16, 16, 4], {
    variant: 'snake',
    outputLabels: ['UP', 'DOWN', 'LEFT', 'RIGHT'],
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  });
  if (noModelWarning) {
    showMessage('Test mode needs a model - starting in Train mode. Use "Load model file" to test one.');
    setTimeout(() => message.classList.remove('visible'), 6000);
  }

  const controls = createSnakeSettingsPanel(settings, {
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
    fpsDisplay.setVisible(controls.showFps);
    if (controls.showFps) fpsDisplay.update(now);
    if (isTest) void evolution.restartDisplayIfDead();
    else void evolution.checkAndEvolve();
    void evolution.readBestAgentState().then((best) => {
      renderer.setBestIndex(best.index);
      hud.update(best, evolution.generation);
      const meta = evolution.bestMeta();
      networkPanel.draw(evolution.genomeAt(best.index), {
        inputs: best.vision,
        outputs: best.decision,
        stats: [
          ['GEN', evolution.generation],
          ['BEST FITNESS', Math.floor(meta.eval)],
          ['POP LEFT', best.aliveCount],
          ['MOVES LEFT', Math.max(0, Math.floor(best.life))],
          ['MUTATION RATE', '0.05'],
          ['SCORE', Math.floor(best.score)],
          ['BEST SCORE', Math.floor(meta.score)],
          ['BEST GEN', meta.generation],
        ],
      });
    });
    renderer.render();
    requestAnimationFrame(loop);
  };
  loop(performance.now());
}

main().catch(showError);
