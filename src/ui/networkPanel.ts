/**
 * Live diagram of the best genome's network, like UINeuralNetworkPanel:
 * nodes as circles, edges green for weight > 0 / red for < 0, width ∝ |weight| (min 1px).
 * Bias rows are not drawn (the Unity panel shows neurons only).
 * Genome layout: layer-major row-major, bias rows last.
 */
export class NetworkPanel {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly topology: readonly number[];
  private lastDraw = 0;

  constructor(topology: readonly number[]) {
    this.topology = topology;
    const canvas = document.createElement('canvas');
    canvas.className = 'network-panel';
    canvas.width = 240;
    canvas.height = 170;
    document.body.append(canvas);
    this.ctx = canvas.getContext('2d')!;
  }

  /** Redraws at most 5 times per second. */
  draw(genome: Float64Array): void {
    const now = performance.now();
    if (now - this.lastDraw < 200) return;
    this.lastDraw = now;

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
}
