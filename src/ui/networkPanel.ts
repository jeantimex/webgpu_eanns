/**
 * Live diagram of the best genome's network, like UINeuralNetworkPanel:
 * nodes as circles, edges green for weight > 0 / red for < 0, width ∝ |weight| (min 1px).
 * Bias rows are not drawn (the Unity panel shows neurons only).
 * Genome layout: layer-major row-major, bias rows last.
 */
export interface NetworkPanelDrawOptions {
  inputs?: ArrayLike<number>;
  outputs?: ArrayLike<number>;
  stats?: readonly (string | readonly [string, string | number])[];
}

export interface NetworkPanelOptions {
  variant?: 'compact' | 'snake';
  outputLabels?: readonly string[];
  onToggle?: (collapsed: boolean) => void;
}

export class NetworkPanel {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly topology: readonly number[];
  private readonly options: NetworkPanelOptions;
  private readonly canvas: HTMLCanvasElement;
  private readonly container?: HTMLDivElement;
  private readonly statsEl?: HTMLDivElement;
  private lastDraw = 0;

  constructor(topology: readonly number[], options: NetworkPanelOptions = {}) {
    this.topology = topology;
    this.options = options;
    const canvas = document.createElement('canvas');
    this.canvas = canvas;
    if (options.variant === 'snake') {
      const container = document.createElement('div');
      container.className = 'snake-network-panel';
      const statsEl = document.createElement('div');
      statsEl.className = 'snake-network-stats';
      const toggle = document.createElement('button');
      toggle.className = 'snake-network-toggle';
      toggle.type = 'button';
      toggle.title = 'Collapse neural map';
      const toggleIcon = document.createElement('span');
      toggleIcon.className = 'material-chevron chevron-left';
      toggle.append(toggleIcon);
      toggle.addEventListener('click', () => {
        container.classList.toggle('collapsed');
        const isCollapsed = container.classList.contains('collapsed');
        toggleIcon.className = `material-chevron ${isCollapsed ? 'chevron-right' : 'chevron-left'}`;
        toggle.title = isCollapsed ? 'Expand neural map' : 'Collapse neural map';
        options.onToggle?.(isCollapsed);
      });
      const resizeHandle = document.createElement('div');
      resizeHandle.className = 'snake-network-resize';
      resizeHandle.title = 'Resize neural map';
      resizeHandle.addEventListener('pointerdown', (event) => {
        event.preventDefault();
        resizeHandle.setPointerCapture(event.pointerId);
        document.body.classList.add('snake-resizing');
      });
      resizeHandle.addEventListener('pointermove', (event) => {
        if (!resizeHandle.hasPointerCapture(event.pointerId)) return;
        const width = Math.max(420, Math.min(window.innerWidth * 0.62, event.clientX));
        document.body.style.setProperty('--snake-panel-open-width', `${Math.round(width)}px`);
        this.resizeSnakeCanvas();
      });
      const endResize = (event: PointerEvent): void => {
        if (resizeHandle.hasPointerCapture(event.pointerId)) resizeHandle.releasePointerCapture(event.pointerId);
        document.body.classList.remove('snake-resizing');
      };
      resizeHandle.addEventListener('pointerup', endResize);
      resizeHandle.addEventListener('pointercancel', endResize);
      canvas.className = 'network-panel snake-network-canvas';
      container.append(statsEl, canvas, resizeHandle, toggle);
      document.body.append(container);
      this.container = container;
      this.statsEl = statsEl;
      this.resizeSnakeCanvas();
      window.addEventListener('resize', () => this.resizeSnakeCanvas());
    } else {
      canvas.className = 'network-panel';
      canvas.width = 240;
      canvas.height = 170;
      document.body.append(canvas);
    }
    this.ctx = canvas.getContext('2d')!;
  }

  private resizeSnakeCanvas(): void {
    if (!this.container || this.options.variant !== 'snake') return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(260, Math.floor(rect.width * dpr));
    const height = Math.max(360, Math.floor(rect.height * dpr));
    if (this.canvas.width === width && this.canvas.height === height) return;
    this.canvas.width = width;
    this.canvas.height = height;
  }

  /** Redraws at most 5 times per second. */
  draw(genome: Float64Array, drawOptions: NetworkPanelDrawOptions = {}): void {
    const now = performance.now();
    if (now - this.lastDraw < 200) return;
    this.lastDraw = now;
    if (this.options.variant === 'snake') {
      this.drawSnake(genome, drawOptions);
      return;
    }

    const { ctx, topology } = this;
    const { width, height } = ctx.canvas;
    ctx.clearRect(0, 0, width, height);

    // Node positions: layers along x, neurons spread along y.
    const xs = topology.map((_, l) => 24 + (l * (width - 48)) / (topology.length - 1));
    const ys = topology.map((count) => Array.from({ length: count }, (_, i) => ((i + 1) * height) / (count + 1)));

    // Edges.
    let offset = 0;
    for (let l = 0; l < topology.length - 1; l++) {
      const inCount = topology[l];
      const outCount = topology[l + 1];
      for (let i = 0; i < inCount; i++) {
        for (let j = 0; j < outCount; j++) {
          const w = genome[offset + i * outCount + j];
          ctx.strokeStyle = w >= 0 ? '#2e8b2e' : '#c0392b';
          ctx.lineWidth = Math.max(1, Math.min(6, Math.abs(w)));
          ctx.beginPath();
          ctx.moveTo(xs[l], ys[l][i]);
          ctx.lineTo(xs[l + 1], ys[l + 1][j]);
          ctx.stroke();
        }
      }
      offset += (inCount + 1) * outCount;
    }

    // Nodes.
    ctx.fillStyle = '#3a3a6e';
    for (let l = 0; l < topology.length; l++) {
      for (const y of ys[l]) {
        ctx.beginPath();
        ctx.arc(xs[l], y, 7, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  private drawSnake(genome: Float64Array, drawOptions: NetworkPanelDrawOptions): void {
    this.resizeSnakeCanvas();
    const { ctx, topology } = this;
    const { width, height } = ctx.canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, width, height);

    const dpr = window.devicePixelRatio || 1;
    const px = (x: number) => x * width;
    const py = (y: number) => y * height;
    const textScale = Math.max(0.75, Math.min(1.25, width / 560));

    const stats = drawOptions.stats ?? [];
    if (this.statsEl) {
      this.statsEl.replaceChildren();
      for (const stat of stats) {
        const item = document.createElement('div');
        item.className = 'snake-network-stat';
        const label = document.createElement('span');
        label.className = 'snake-network-stat-label';
        const value = document.createElement('span');
        value.className = 'snake-network-stat-value';
        if (typeof stat === 'string') {
          const [rawLabel, ...rest] = stat.split(':');
          label.textContent = rawLabel.trim();
          value.textContent = rest.join(':').trim();
        } else {
          label.textContent = stat[0];
          value.textContent = String(stat[1]);
        }
        item.append(label, value);
        this.statsEl.append(item);
      }
    }

    ctx.save();
    const xs = [0.13, 0.43, 0.66, 0.84].map(px);
    const layerTop = Math.max(py(0.12), 96 * dpr);
    const layerBottom = height - 88 * dpr;
    const ys = topology.map((count) =>
      Array.from({ length: count }, (_, i) => layerTop + ((i + 0.5) * (layerBottom - layerTop)) / count),
    );
    const radius = Math.max(5, Math.min(10, width * 0.015));

    let offset = 0;
    ctx.lineCap = 'round';
    for (let l = 0; l < topology.length - 1; l++) {
      const inCount = topology[l];
      const outCount = topology[l + 1];
      for (let i = 0; i < inCount; i++) {
        for (let j = 0; j < outCount; j++) {
          const w = genome[offset + i * outCount + j];
          const mag = Math.min(1, Math.abs(w));
          ctx.strokeStyle = w >= 0 ? `rgba(0, 50, 255, ${0.18 + mag * 0.55})` : `rgba(255, 20, 20, ${0.18 + mag * 0.55})`;
          ctx.lineWidth = (0.6 + mag * 1.4) * dpr;
          ctx.beginPath();
          ctx.moveTo(xs[l], ys[l][i]);
          ctx.lineTo(xs[l + 1], ys[l + 1][j]);
          ctx.stroke();
        }
      }
      offset += (inCount + 1) * outCount;
    }

    const inputs = drawOptions.inputs;
    const outputs = drawOptions.outputs;
    let selectedOutput = -1;
    if (outputs && outputs.length > 0) {
      selectedOutput = 0;
      for (let i = 1; i < outputs.length; i++) if (outputs[i] > outputs[selectedOutput]) selectedOutput = i;
    }

    for (let l = 0; l < topology.length; l++) {
      for (let i = 0; i < topology[l]; i++) {
        const activeInput = l === 0 && inputs && Math.abs(inputs[i]) > 0;
        const activeOutput = l === topology.length - 1 && i === selectedOutput;
        ctx.fillStyle = activeInput || activeOutput ? '#00d12d' : '#f5f5f5';
        ctx.strokeStyle = '#202020';
        ctx.lineWidth = 1.5 * dpr;
        ctx.beginPath();
        ctx.arc(xs[l], ys[l][i], radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    const labels = this.options.outputLabels ?? [];
    ctx.font = `${15 * textScale * dpr}px "JetBrains Mono", monospace`;
    ctx.fillStyle = '#f5f5f5';
    ctx.textBaseline = 'middle';
    const outLayer = topology.length - 1;
    for (let i = 0; i < Math.min(labels.length, topology[outLayer]); i++) {
      ctx.fillText(labels[i], xs[outLayer] + 22 * dpr, ys[outLayer][i]);
    }

    // Weight-sign key, centred either side of the midline. The two labels sit far
    // enough apart to read as separate items rather than one run-on string.
    ctx.font = `${11 * textScale * dpr}px "JetBrains Mono", monospace`;
    ctx.textAlign = 'center';
    const keyY = height - 58 * dpr;
    ctx.fillStyle = '#ff3030';
    ctx.fillText('RED < 0', px(0.38), keyY);
    ctx.fillStyle = '#102cff';
    ctx.fillText('BLUE > 0', px(0.62), keyY);
    ctx.restore();
  }
}
