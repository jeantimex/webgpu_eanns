import '../../style.css';
import { currentSettings } from '../../gui/controls_gui';
import { createDemoSettingsPanel } from '../../ui/demoSettingsPanel';
import { NetworkPanel } from '../../ui/networkPanel';
import { requiredElement } from '../../utils/dom';
import { initializeWebGPU } from '../../webgpu/utils';
import { TetrisEvolution, type BestTetrisSnapshot } from './tetris_evolution';
import { PIECES, PIECE_COLORS } from './tetris_pieces';
import { TetrisRenderer } from './tetris_renderer';

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

/** NES-style right panel: LINES / SCORE / LEVEL + NEXT piece preview. */
function createPanel(): { update(best: BestTetrisSnapshot, generation: number): void } {
  const panel = document.createElement('div');
  panel.className = 'nes-panel';
  const makeRow = (label: string): HTMLDivElement => {
    const box = document.createElement('div');
    box.className = 'nes-box';
    const l = document.createElement('div');
    l.className = 'nes-label';
    l.textContent = label;
    const v = document.createElement('div');
    v.className = 'nes-value';
    box.append(l, v);
    panel.append(box);
    return v;
  };
  const lines = makeRow('LINES');
  const score = makeRow('SCORE');
  const level = makeRow('LEVEL');
  const alive = makeRow('ALIVE');
  const genBox = document.createElement('div');
  genBox.className = 'nes-box';
  const genLabel = document.createElement('div');
  genLabel.className = 'nes-label';
  genLabel.textContent = 'NEXT';
  const nextCanvas = document.createElement('canvas');
  nextCanvas.width = 96;
  nextCanvas.height = 64;
  genBox.append(genLabel, nextCanvas);
  panel.append(genBox);
  const gen = document.createElement('div');
  gen.className = 'nes-gen';
  panel.append(gen);
  document.body.append(panel);

  const ctx = nextCanvas.getContext('2d')!;
  let lastNext = -1;
  let lastText = '';

  return {
    update(best, generation) {
      const text = `${best.lines}|${best.score}|${best.level}|${best.aliveCount}|${generation}`;
      if (text !== lastText) {
        lines.textContent = String(best.lines).padStart(3, '0');
        score.textContent = String(best.score).padStart(6, '0');
        level.textContent = String(best.level).padStart(2, '0');
        alive.textContent = String(best.aliveCount).padStart(3, '0');
        gen.textContent = `GENERATION: ${generation}`;
        lastText = text;
      }
      if (best.nextType !== lastNext) {
        lastNext = best.nextType;
        ctx.clearRect(0, 0, 96, 64);
        const rot = PIECES[best.nextType][0];
        const [r, g, b] = PIECE_COLORS[best.nextType];
        ctx.fillStyle = `rgb(${r * 255},${g * 255},${b * 255})`;
        const xs = rot.map((c) => c[0]);
        const ys = rot.map((c) => c[1]);
        const offX = 44 - (Math.min(...xs) + Math.max(...xs)) * 8;
        const offY = 28 - (Math.min(...ys) + Math.max(...ys)) * 8;
        for (const [x, y] of rot) ctx.fillRect(offX + x * 16 + 1, offY + y * 16 + 1, 14, 14);
      }
    },
  };
}

async function main(): Promise<void> {
  document.body.classList.add('snake-layout');
  const gpu = await initializeWebGPU(canvas);
  const settings = currentSettings();

  const evolution = TetrisEvolution.init(gpu.device, settings.population);
  const renderer = new TetrisRenderer(canvas, gpu, evolution.buffers);
  const panel = createPanel();
  const networkPanel = new NetworkPanel([15, 8, 1], {
    variant: 'snake',
    outputLabels: ['PLACE'],
    onToggle: (collapsed) => document.body.classList.toggle('snake-panel-collapsed', collapsed),
  });

  const notWired = (): void => {
    showMessage('Model save/load is not wired up for Tetris yet.');
  };
  const controls = createDemoSettingsPanel(settings, {
    onSaveModel: notWired,
    onLoadSavedBest: notWired,
    onLoadModelFile: notWired,
  });

  // One placement per tick: 2 pieces/sec base rate x sim speed.
  let last = performance.now();
  let acc = 0;
  const loop = (now: number): void => {
    acc += (Math.min(now - last, 100) / 1000) * 2 * controls.speed;
    last = now;
    const steps = Math.min(Math.floor(acc), 120);
    acc -= steps;
    if (steps > 0) evolution.substeps(steps);
    void evolution.checkAndEvolve();
    void evolution.readBestAgentState().then((best) => {
      renderer.setBestIndex(best.index);
      panel.update(best, evolution.generation);
      networkPanel.draw(evolution.genomeAt(best.index), {
        stats: [
          ['GEN', evolution.generation],
          ['SCORE', best.score],
          ['LINES', best.lines],
          ['LEVEL', best.level],
          ['POP LEFT', best.aliveCount],
        ],
      });
    });
    renderer.render(evolution.tickCount);
    requestAnimationFrame(loop);
  };
  loop(performance.now());
}

main().catch(showError);
