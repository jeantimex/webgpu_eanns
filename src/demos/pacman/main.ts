import '../../style.css';
import { startDemo, type Evolution } from '../../core';
import { updateUrlParamLive } from '../../gui/controls_gui';
import {
  A,
  AGENT_FLOATS,
  EPISODES_PER_GENOME,
  LEVEL_SECS,
  LEVEL_SECS_MAX,
  LEVEL_SECS_MIN,
  PACMAN_OUTPUT_LABELS,
} from './pacman_buffers';
import { pacmanNetwork } from './pacman_net';
import { pacmanControls, pacmanSim, pacmanState, pickShownAgent } from './pacman_sim';
import { PacmanRenderer } from './pacman_renderer';

const rawLevelSecs = Number(new URLSearchParams(window.location.search).get('level'));
const initialLevelSecs = Number.isFinite(rawLevelSecs) && rawLevelSecs > 0
  ? Math.min(LEVEL_SECS_MAX, Math.max(LEVEL_SECS_MIN, Math.round(rawLevelSecs)))
  : LEVEL_SECS;

function levelTimeLeft(states: Float32Array, shown: number): string {
  const left = Math.max(0, pacmanState.levelSeconds - states[shown * AGENT_FLOATS + A.levelTicks] / 60);
  return `${Math.ceil(left)}s`;
}

/** Play mode starts frozen at the initial position; the first arrow key starts the game. */
const play = {
  waiting: true,
  setup(evo: Evolution): void {
    pacmanState.playMode = true;
    evo.writeParams();
  },
  onKeydown(e: KeyboardEvent, evo: Evolution): void {
    if (e.code === 'Space') {
      e.preventDefault();
      play.waiting = true;
      evo.resetDisplayAgent();
      return;
    }
    let dir = -1;
    if (e.key === 'ArrowUp' || e.code === 'KeyW') dir = 0;
    else if (e.key === 'ArrowRight' || e.code === 'KeyD') dir = 3;
    else if (e.key === 'ArrowDown' || e.code === 'KeyS') dir = 1;
    else if (e.key === 'ArrowLeft' || e.code === 'KeyA') dir = 2;

    if (dir >= 0) {
      e.preventDefault();
      play.waiting = false;
      pacmanControls.setPlayerDesiredDir(evo, dir);
      void evo.readStates().then((states) => {
        if (pacmanSim.isAgentDone(states, evo.displayAgentIndex)) evo.resetDisplayAgent();
      });
    }
  },
};

startDemo({
  namespace: 'pacman',
  network: pacmanNetwork,
  simulation: pacmanSim,
  episodes: EPISODES_PER_GENOME,
  // Tournament selection (§3.2): 对于《吃豆人》这个问题，锦标赛选择的效果更好,
  // because it purges the useless random strategies fast.
  ga: {
    selection: 'tournament',
    tournamentSize: 3,
    eliteFraction: 0.02,
    crossoverRate: 0.8,
    mutateRate: 0.05, // adaptive: pacmanSim.beforeGaStep drives it from here
    mutateRange: 1,
    resetShare: 0.5, // half resets (explore), half drift (let confident weights grow)
    driftSigma: 0.5,
    clamp: 8,
  },
  stepsPerSecond: 60, // the source engine's logic rate
  maxStepsPerFrame: 240,
  bodyClass: 'snake-layout',
  hudColor: '#f1f5f9', // arcade-black background
  play,

  // Test mode replays agent 0 (a training slot running the injected genome),
  // re-seeded on every restart so the sampled policy shows its full range.
  testTick: (evo) => {
    void evo.readStates().then((states) => {
      if (pacmanSim.isAgentDone(states, 0)) {
        pacmanState.replaySeed = (pacmanState.replaySeed + 1) >>> 0;
        evo.resetAgent(0, pacmanState.replaySeed);
      }
    });
  },

  createRenderer: async (canvas, gpu, evo) => {
    const renderer = await PacmanRenderer.create(canvas, gpu, evo.buffers);
    const startTime = performance.now();
    return {
      render: () => renderer.render((performance.now() - startTime) / 1000),
      setBestIndex: (i) => renderer.setBestIndex(i),
      setShowPelletPath: (on: boolean) => renderer.setShowPelletPath(on),
      setShowGhostPath: (on: boolean) => renderer.setShowGhostPath(on),
    };
  },

  pickShownAgent,

  hud: (evo, states, shown) => {
    const o = shown * AGENT_FLOATS;
    return (
      `Score:   ${states[o + A.score]}\n` +
      `Dots:    ${states[o + A.dotsLeft]}\n` +
      `Level:   ${states[o + A.level]}\n` +
      `Playing: ${evo.countAlive(states)}`
    );
  },

  networkPanel: {
    variant: 'snake',
    outputLabels: PACMAN_OUTPUT_LABELS,
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  },

  panelStats: (evo, shown, _inputs, states, mode) => {
    const o = shown * AGENT_FLOATS;
    const score = states[o + A.score];
    const dots = states[o + A.dotsLeft];
    const level = states[o + A.level];
    const time = levelTimeLeft(states, shown);
    const gameOver = states[o + A.gameOver] > 0.5;
    if (mode === 'Play') {
      return [
        ['MODE', 'PLAY'],
        ['SCORE', score],
        ['DOTS', dots],
        ['LEVEL', level],
        ['TIME', time],
        ['STATUS', play.waiting ? 'READY (Press Arrow Key)' : gameOver ? 'GAME OVER (Press Arrow Key)' : 'PLAYING'],
      ];
    }
    if (mode === 'Test') {
      return [
        ['MODE', 'TEST'],
        ['SCORE', score],
        ['DOTS', dots],
        ['LEVEL', level],
        ['TIME', time],
        ['STATUS', gameOver ? 'RESTARTING' : 'RUNNING'],
      ];
    }
    return [
      ['GEN', evo.generation],
      ['MUT', `${(pacmanState.mutateRate * 100).toFixed(0)}%`],
      ['SCORE', score],
      ['BEST SCORE', pacmanState.bestScore],
      ['BEST LEVEL', pacmanState.highestLevel],
      ['BEST GEN', evo.bestGeneration],
      ['DOTS', dots],
      ['LEVEL', level],
      ['TIME', time],
      ['POP LEFT', evo.countAlive(states)],
    ];
  },

  sliders: (evo) => [
    {
      label: 'Level time',
      min: LEVEL_SECS_MIN,
      max: LEVEL_SECS_MAX,
      step: 5,
      initial: initialLevelSecs,
      unit: 's',
      onChange: (value) => {
        pacmanControls.setLevelSeconds(evo, value);
        updateUrlParamLive('level', value);
      },
    },
  ],

  // Debug overlays for the perception vector the network actually sees.
  toggles: (_evo, renderer) => [
    { label: 'Pellet BFS path', onChange: (on) => renderer.setShowPelletPath(on) },
    { label: 'Ghost BFS path', onChange: (on) => renderer.setShowGhostPath(on) },
  ],
});
