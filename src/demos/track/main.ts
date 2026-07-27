import '../../style.css';
import { autosaveBestModel, loadBestModel, startDemo } from '../../core';
import { CAR_FLOATS } from './buffers';
import { parseUnityGenotype } from './model';
import { trackNetwork } from './network';
import { Renderer } from './renderer';
import { runSelftest } from './selftest';
import { loadTrack } from './track';
import { createTrackSim } from './track_sim';

/** Available tracks (public/tracks/<name>.json); adding one is a line here + the JSON file. */
export const TRACKS = ['track1', 'track2', 'track3', 'track4', 'practice'];

async function boot(): Promise<void> {
// The track is chosen by URL param (?track=track2) before the sim can be built,
// so resolve it the same way currentSettings will, then load it up front.
const trackName = (() => {
  const raw = new URLSearchParams(window.location.search).get('track') ?? localStorage.getItem('eanns:track');
  return raw && TRACKS.includes(raw) ? raw : TRACKS[0];
})();
const track = await loadTrack(`/tracks/${trackName}.json`);

/** Highest-fitness car (what the camera follows). */
function pickShown(states: Float32Array): number {
  let best = 0;
  for (let i = 1; i < states.length / CAR_FLOATS; i++) {
    if (states[i * CAR_FLOATS + 7] > states[best * CAR_FLOATS + 7]) best = i;
  }
  return best;
}

// Save best model downloads the genome of the car currently on screen.
let lastShown = -1;
let lastShownFitness = 0;

startDemo({
  namespace: 'track',
  network: trackNetwork,
  simulation: createTrackSim(track),
  // Unity's EvolutionManager defaults: remainder stochastic sampling + blend
  // crossover + dynamic elite scaling (ported as utils/ga.ts's remainder-blend).
  ga: { selection: 'remainder-blend' },
  displayAgent: false,
  stepMode: 'per-frame', // the original steps `speed` substeps every rAF
  seed: 1,
  tracks: TRACKS,
  bodyClass: 'snake-layout',
  legacyModelParser: parseUnityGenotype,
  // Best models are keyed per track ('eanns:best:track1'); the test slot is the
  // legacy global 'eanns:testModel'. Autosave is per-generation (below), not
  // only on new records.
  modelNamespaces: () => ({ best: track.name, test: '' }),
  autosave: false,
  callbacks: {
    onFitness: (evo, fitnesses) => {
      let best = 0;
      for (let i = 1; i < fitnesses.length; i++) if (fitnesses[i] > fitnesses[best]) best = i;
      autosaveBestModel(track.name, trackNetwork, evo.genomeAt(best), fitnesses[best], {
        track: track.name,
        generation: evo.generation,
      });
    },
  },
  actions: {
    onSaveModel: (evo) =>
      lastShown >= 0
        ? {
            genome: evo.genomeAt(lastShown),
            meta: { track: track.name, generation: evo.generation, eval: lastShownFitness },
          }
        : null,
    onLoadSavedBest: (evo) => {
      const saved = loadBestModel(track.name, trackNetwork);
      if (saved) evo.injectBest(Float64Array.from(saved.weights));
      else console.log(`No saved best for track "${track.name}" yet.`);
    },
  },

  beforeStart: async ({ gpu, showMessage }) => {
    // ?selftest runs the CPU/GPU parity asserts and reports to console + #message.
    if (!new URLSearchParams(location.search).has('selftest')) return false;
    const result = await runSelftest(gpu.device);
    showMessage(result.pass ? 'SELFTEST PASS' : `SELFTEST FAIL\n${result.failures.join('\n')}`);
    return true;
  },

  createRenderer: (canvas, gpu, evo) => new Renderer(canvas, gpu, track, evo.buffers),

  pickShownAgent: (states) => {
    lastShown = pickShown(states);
    lastShownFitness = states[lastShown * CAR_FLOATS + 7];
    return lastShown;
  },

  afterFrame: (renderer, _evo, states, shown) => {
    renderer.follow(states[shown * CAR_FLOATS], states[shown * CAR_FLOATS + 1], shown, false);
  },

  hud: (evo, states, shown) => {
    const o = shown * CAR_FLOATS;
    return (
      `Population:   ${evo.countAlive(states)}\n` +
      `Turn:   ${states[o + 8].toFixed(5)}\n` +
      `Engine: ${states[o + 9].toFixed(5)}\n` +
      `Eval:   ${states[o + 7].toFixed(5)}`
    );
  },

  networkPanel: {
    variant: 'snake',
    outputLabels: ['TURN', 'ENGINE'],
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  },

  panelStats: (evo, shown, _inputs, states) => {
    const o = shown * CAR_FLOATS;
    return [
      ['GEN', evo.generation],
      ['FITNESS', states[o + 7].toFixed(3)],
      ['POP LEFT', evo.countAlive(states)],
      ['TURN', states[o + 8].toFixed(2)],
      ['ENGINE', states[o + 9].toFixed(2)],
      ['TRACK', track.name],
    ];
  },
});
}

void boot();
