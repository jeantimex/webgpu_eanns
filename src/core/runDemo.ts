import { currentSettings, persistMode, updateSetting } from '../gui/controls_gui';
import {
  createDemoSettingsPanel,
  createFpsDisplay,
  type DemoSettingsSlider,
  type DemoSettingsToggle,
} from '../ui/demoSettingsPanel';
import { NetworkPanel } from '../ui/networkPanel';
import { requiredElement } from '../utils/dom';
import { initializeWebGPU, type WebGPUState } from '../webgpu/utils';
import { Evolution, type EvolutionCallbacks, type Simulation } from './evolution';
import type { GAConfig } from './ga';
import { autosaveBestModel, downloadModel, loadBestModel, loadTestModel, parseModelText, saveTestModel } from './modelStore';
import { forwardCPU, type Network } from './network';
import type { EpisodeAggregate } from '../utils/evaluation';

export interface DemoRenderer {
  render(): void;
  setBestIndex?(index: number): void;
}

export type DemoMode = 'Train' | 'Test' | 'Play';

/** Human-playable mode (pacman): the display agent is steered by the keyboard. */
export interface PlayMode {
  /** Frozen until the first input (pacman waits for an arrow key). */
  waiting: boolean;
  /** Called once when the page boots in Play mode. */
  setup?(evo: Evolution): void;
  onKeydown?(event: KeyboardEvent, evo: Evolution): void;
}

export interface DemoDescriptor<R extends DemoRenderer = DemoRenderer> {
  /** Settings/model key prefix: eanns:best:<namespace>, eanns:testModel:<namespace>. */
  namespace: string;
  network: Network;
  simulation: Simulation;
  ga: GAConfig;
  episodes?: number;
  episodeAggregate?: EpisodeAggregate;
  /** Trailing replay agent (default true). */
  displayAgent?: boolean;
  /** Sim steps per real second at speed 1, accumulator mode only (snake: 100, 60 Hz games: 60). */
  stepsPerSecond?: number;
  /** Cap on steps per rAF frame (default 600). */
  maxStepsPerFrame?: number;
  /** GA seed (default 1). */
  seed?: number;
  /** Population in Test mode (default 1: only the loaded model runs). */
  testPopulation?: number;
  /** Track names for the settings panel's track selector. */
  tracks?: readonly string[];
  /** Class added to document.body (page layout hooks in style.css). */
  bodyClass?: string;
  /** HUD chip text color (pacman: light text over the arcade-black board). */
  hudColor?: string;
  /** Non-JSON model file formats (SnakeAI CSV, Unity genotypes). */
  legacyModelParser?: (text: string) => Float64Array;
  /** Enables the settings panel's Play mode. */
  play?: PlayMode;
  /**
   * Sim stepping model. Default: fixed-Hz accumulator (stepsPerSecond x speed).
   * 'per-frame': exactly `speed` substeps every rAF (the track demo's model).
   */
  stepMode?: 'accumulator' | 'per-frame';
  /** localStorage namespaces for models; default both = namespace. The track demo
   *  keys best models per track name and uses the legacy global test slot (''). */
  modelNamespaces?: (track: string) => { best: string; test: string };
  /** Autosave the best genome to localStorage on every new best (default true). */
  autosave?: boolean;
  /** Overrides for the settings panel's Save/Load buttons. */
  actions?: {
    onSaveModel?(evo: Evolution): { genome: Float64Array; meta?: Record<string, number | string> } | null;
    onLoadSavedBest?(evo: Evolution): void;
  };
  /** Runs before settings are read; return true to take over the page (track's ?selftest). */
  beforeStart?(ctx: { gpu: WebGPUState; showMessage: (text: string) => void }): Promise<boolean>;
  /** Per-frame hook after the shown agent is picked (track's camera follow). */
  afterFrame?(renderer: R, evo: Evolution, states: Float32Array, shown: number): void;
  /** LCD chip background on the HUD (default false; snake's styled chips pass true). */
  hudChipStyle?: boolean;

  createRenderer(canvas: HTMLCanvasElement, gpu: WebGPUState, evo: Evolution): R | Promise<R>;

  /** Which agent the camera/panel follows; default the display agent (or 0). */
  pickShownAgent?(states: Float32Array, evo: Evolution): number;
  /** Stats chip text (top-left HUD). No HUD chip when omitted. */
  hud?(evo: Evolution, states: Float32Array, shown: number, mode: DemoMode): string;
  /** NetworkPanel config; no panel when omitted. */
  networkPanel?: {
    variant?: 'compact' | 'snake';
    outputLabels?: readonly string[];
    onToggle?: (collapsed: boolean) => void;
  };
  /** Rows for the NetworkPanel stats table. */
  panelStats?(evo: Evolution, shown: number, inputs: Float32Array | undefined, states: Float32Array, mode: DemoMode): (string | readonly [string, string | number])[];
  /** Demo-specific settings controls; built after the evolution/renderer exist. */
  toggles?: (evo: Evolution, renderer: R) => readonly DemoSettingsToggle[];
  sliders?: (evo: Evolution, renderer: R) => readonly DemoSettingsSlider[];
  /** Test-mode per-frame hook, replacing the default display-agent restart. */
  testTick?: (evo: Evolution) => void;
  /** Extra evolution callbacks; autosave-on-new-best is always wired. */
  callbacks?: EvolutionCallbacks;
}

function createHudChips(color?: string, chipStyle = false): { stats: HTMLDivElement; generation: HTMLDivElement } {
  const chip = (el: HTMLDivElement): void => {
    if (color) el.style.color = color;
    if (!chipStyle) return;
    el.style.background = 'rgba(170, 176, 155, 0.92)';
    if (!color) el.style.color = '#1e1e1e';
    el.style.padding = '0.35rem 0.6rem';
    el.style.borderRadius = '0.4rem';
  };
  const stats = document.createElement('div');
  stats.className = 'hud hud-stats';
  chip(stats);
  const generation = document.createElement('div');
  generation.className = 'hud hud-generation';
  chip(generation);
  document.body.append(stats, generation);
  return { stats, generation };
}

/**
 * The shared demo bootstrap (what every demo's main.ts used to hand-roll):
 * settings → WebGPU → Evolution → renderer → HUD → NetworkPanel → settings
 * panel → rAF loop with fixed-Hz accumulator, speed control, and Train/Test
 * modes. Model save/load buttons are wired to core/modelStore.
 */
export async function runDemo<R extends DemoRenderer>(descriptor: DemoDescriptor<R>): Promise<void> {
  const { namespace, network, simulation } = descriptor;
  const canvas = requiredElement<HTMLCanvasElement>('#webgpu-canvas');
  const message = requiredElement<HTMLDivElement>('#message');

  const showMessage = (text: string): void => {
    console.log(text);
    message.textContent = text;
    message.classList.add('visible');
  };

  if (descriptor.bodyClass) document.body.classList.add(descriptor.bodyClass);
  const gpu = await initializeWebGPU(canvas);
  if (await descriptor.beforeStart?.({ gpu, showMessage })) return;
  const settings = currentSettings(descriptor.tracks);
  const modelNs = descriptor.modelNamespaces?.(settings.track) ?? { best: namespace, test: namespace };
  const isPlay = settings.mode === 'Play' && descriptor.play !== undefined;
  let isTest = settings.mode === 'Test';
  let noModelWarning = false;
  const testModel = loadTestModel(modelNs.test, network);
  if (isTest && !testModel) {
    isTest = false;
    noModelWarning = true;
    persistMode('Train');
  }
  const mode: DemoMode = isPlay ? 'Play' : isTest ? 'Test' : 'Train';

  const callbacks: EvolutionCallbacks = {
    onNewBest: (evo, genome, fitness, meta) => {
      if (descriptor.autosave ?? true) autosaveBestModel(modelNs.best, network, genome, fitness, meta);
      descriptor.callbacks?.onNewBest?.(evo, genome, fitness, meta);
    },
    onFitness: (evo, fitnesses) => descriptor.callbacks?.onFitness?.(evo, fitnesses),
    onGeneration: (evo) => descriptor.callbacks?.onGeneration?.(evo),
  };

  const evolution = Evolution.init(gpu.device, network, simulation, {
    populationSize: isTest ? (descriptor.testPopulation ?? 1) : settings.population,
    seed: descriptor.seed,
    episodes: descriptor.episodes,
    episodeAggregate: descriptor.episodeAggregate,
    displayAgent: descriptor.displayAgent,
    ga: descriptor.ga,
    callbacks,
  });
  if (isTest) evolution.injectBest(testModel!);

  const renderer = await descriptor.createRenderer(canvas, gpu, evolution);
  const hudChips = createHudChips(descriptor.hudColor, descriptor.hudChipStyle ?? false);
  if (!descriptor.hud) hudChips.stats.style.display = 'none';
  const fpsDisplay = createFpsDisplay();
  const panelOptions = descriptor.networkPanel;
  const networkPanel = panelOptions
    ? new NetworkPanel(network.topology, {
        variant: panelOptions.variant,
        outputLabels: panelOptions.outputLabels,
        onToggle: panelOptions.onToggle,
      })
    : null;
  if (noModelWarning) {
    showMessage('Test mode needs a model - starting in Train mode. Use "Load model file" to test one.');
    setTimeout(() => message.classList.remove('visible'), 6000);
  }

  if (isPlay && descriptor.play) {
    descriptor.play.setup?.(evolution);
    if (descriptor.play.onKeydown) {
      const onKeydown = descriptor.play.onKeydown;
      window.addEventListener('keydown', (e) => onKeydown(e, evolution));
    }
  }

  const controls = createDemoSettingsPanel(settings, {
    onSaveModel: () => {
      const custom = descriptor.actions?.onSaveModel?.(evolution);
      if (descriptor.actions?.onSaveModel && !custom) return;
      const genome = custom?.genome ?? evolution.bestGenome ?? evolution.genomeAt(0);
      downloadModel(modelNs.best, network, genome, custom?.meta ?? {
        generation: evolution.bestGeneration,
        eval: Number.isFinite(evolution.bestFitness) ? evolution.bestFitness : 0,
        ...evolution.bestMetaValue,
      });
    },
    onLoadSavedBest: () => {
      if (descriptor.actions?.onLoadSavedBest) {
        descriptor.actions.onLoadSavedBest(evolution);
        return;
      }
      const saved = loadBestModel(modelNs.best, network);
      if (!saved) {
        showMessage(`No saved best ${namespace} model yet.`);
        return;
      }
      saveTestModel(modelNs.test, network, Float64Array.from(saved.weights));
      updateSetting('mode', 'Test');
    },
    onLoadModelFile: (file) => {
      file
        .text()
        .then((text) => {
          saveTestModel(modelNs.test, network, parseModelText(text, network, descriptor.legacyModelParser));
          updateSetting('mode', 'Test');
        })
        .catch((error: unknown) => alert(error instanceof Error ? error.message : String(error)));
    },
  }, {
    showFps: true,
    tracks: descriptor.tracks,
    toggles: descriptor.toggles?.(evolution, renderer),
    sliders: descriptor.sliders?.(evolution, renderer),
  });

  const stepsPerSecond = descriptor.stepsPerSecond ?? 60;
  const maxSteps = descriptor.maxStepsPerFrame ?? 600;
  const perFrame = descriptor.stepMode === 'per-frame';
  let last = performance.now();
  let acc = 0;
  let lastStats = '';
  let lastGen = -1;
  const loop = (now: number): void => {
    let steps: number;
    if (perFrame) {
      steps = controls.speed;
    } else {
      acc += (Math.min(now - last, 100) / 1000) * stepsPerSecond * controls.speed;
      steps = Math.min(Math.floor(acc), maxSteps);
      acc -= steps;
    }
    last = now;
    if (steps > 0 && !(isPlay && descriptor.play?.waiting)) evolution.substeps(steps);
    fpsDisplay.setVisible(controls.showFps);
    if (controls.showFps) fpsDisplay.update(now);
    if (isTest) {
      if (descriptor.testTick) descriptor.testTick(evolution);
      else if (descriptor.displayAgent ?? true) void evolution.restartDisplayIfDead();
      else void evolution.resetIfOver();
    } else if (!isPlay) {
      void evolution.checkAndEvolve();
    }
    void evolution.readStates().then((states) => {
      const shown =
        descriptor.pickShownAgent?.(states, evolution) ??
        ((descriptor.displayAgent ?? true) ? evolution.displayAgentIndex : 0);
      renderer.setBestIndex?.(shown);
      descriptor.afterFrame?.(renderer, evolution, states, shown);
      if (descriptor.hud) {
        const text = descriptor.hud(evolution, states, shown, mode);
        if (text !== lastStats) {
          hudChips.stats.textContent = text;
          lastStats = text;
        }
      }
      if (evolution.generation !== lastGen) {
        hudChips.generation.textContent = `Generation: ${evolution.generation}`;
        lastGen = evolution.generation;
      }
      if (networkPanel) {
        const probe = simulation.probe?.(states, shown);
        const genome = evolution.genomeAt(shown);
        networkPanel.draw(genome, {
          inputs: probe?.inputs,
          outputs: probe?.inputs ? forwardCPU(network, genome, probe.inputs) : undefined,
          stats: descriptor.panelStats?.(evolution, shown, probe?.inputs, states, mode) ?? probe?.stats,
        });
      }
      renderer.render();
    });
    requestAnimationFrame(loop);
  };
  loop(performance.now());
}

/** Entry-point wrapper matching the old mains' error handling. */
export function startDemo<R extends DemoRenderer>(descriptor: DemoDescriptor<R>): void {
  runDemo(descriptor).catch((error: unknown) => {
    console.error(error);
    const message = document.querySelector<HTMLDivElement>('#message');
    const text = error instanceof Error ? error.message : 'Unable to start WebGPU.';
    if (message) {
      message.textContent = text;
      message.classList.add('visible');
    }
  });
}
