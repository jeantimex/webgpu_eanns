import { snakeNetwork } from './snake_net';

/** Parser for CodeBullet's SnakeAI CSV exports (L0/L1/L2 columns), kept as the
 *  legacy format accepted by "Load model file" alongside the JSON format. */

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        quoted = !quoted;
      }
    } else if (ch === ',' && !quoted) {
      out.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function snakeAiLayerToGenome(values: number[], rows: number, cols: number): Float64Array {
  if (values.length !== rows * cols) throw new Error(`SnakeAI layer has ${values.length} weights, expected ${rows * cols}.`);
  const flat = new Float64Array(values.length);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      flat[c * rows + r] = values[r * cols + c];
    }
  }
  return flat;
}

export function parseSnakeAiCsv(text: string): Float64Array {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const header = parseCsvLine(lines[0]);
  const l0 = header.indexOf('L0');
  const l1 = header.indexOf('L1');
  const l2 = header.indexOf('L2');
  if (l0 < 0 || l1 < 0 || l2 < 0) throw new Error('SnakeAI CSV must contain L0, L1, and L2 columns.');

  const columns = [l0, l1, l2].map((col) => {
    const values: number[] = [];
    for (let r = 1; r < lines.length; r++) {
      const raw = parseCsvLine(lines[r])[col]?.trim();
      if (raw) values.push(Number(raw));
    }
    return values;
  });

  const layers = [
    snakeAiLayerToGenome(columns[0], 16, 25),
    snakeAiLayerToGenome(columns[1], 16, 17),
    snakeAiLayerToGenome(columns[2], 4, 17),
  ];
  const genome = new Float64Array(snakeNetwork.genomeSize);
  let offset = 0;
  for (const layer of layers) {
    genome.set(layer, offset);
    offset += layer.length;
  }
  return genome;
}
