import { SNAKE_GENOME_SIZE, SNAKE_TOPOLOGY } from './snake_buffers';

export interface SnakeModel {
  topology: number[];
  weights: number[];
  meta?: { generation?: number; eval?: number; score?: number };
}

const STORAGE_KEY = 'eanns:best:snake';
const TEST_KEY = 'eanns:testModel:snake';

function validateWeights(weights: unknown): Float64Array {
  if (!Array.isArray(weights) || weights.length !== SNAKE_GENOME_SIZE || !weights.every((w) => Number.isFinite(w))) {
    throw new Error(`Snake model must contain exactly ${SNAKE_GENOME_SIZE} finite weights.`);
  }
  return Float64Array.from(weights as number[]);
}

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

function parseSnakeAiCsv(text: string): Float64Array {
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
  const genome = new Float64Array(SNAKE_GENOME_SIZE);
  let offset = 0;
  for (const layer of layers) {
    genome.set(layer, offset);
    offset += layer.length;
  }
  return genome;
}

export function parseSnakeModelText(text: string): Float64Array {
  if (!text.trimStart().startsWith('{')) return parseSnakeAiCsv(text);
  const model = JSON.parse(text) as SnakeModel;
  if (JSON.stringify(model.topology) !== JSON.stringify([...SNAKE_TOPOLOGY])) {
    throw new Error(`Unsupported Snake topology [${model.topology}], expected [${[...SNAKE_TOPOLOGY]}].`);
  }
  return validateWeights(model.weights);
}

export function saveSnakeModel(weights: Float64Array, meta: SnakeModel['meta']): void {
  const model: SnakeModel = { topology: [...SNAKE_TOPOLOGY], weights: [...weights], meta };
  const url = URL.createObjectURL(new Blob([JSON.stringify(model, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `eanns-snake-gen${meta?.generation ?? 0}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function loadSnakeModelFile(file: File): Promise<Float64Array> {
  return parseSnakeModelText(await file.text());
}

export function autosaveSnakeBest(weights: Float64Array, generation: number, eval_: number, score: number): void {
  const saved = loadSavedSnakeBest();
  if (saved && saved.eval > eval_) return;
  const model: SnakeModel = { topology: [...SNAKE_TOPOLOGY], weights: [...weights], meta: { generation, eval: eval_, score } };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(model));
}

export function loadSavedSnakeBest(): { weights: Float64Array; generation: number; eval: number; score: number } | null {
  const text = localStorage.getItem(STORAGE_KEY);
  if (!text) return null;
  try {
    const model = JSON.parse(text) as SnakeModel;
    return {
      weights: validateWeights(model.weights),
      generation: model.meta?.generation ?? 0,
      eval: model.meta?.eval ?? 0,
      score: model.meta?.score ?? 0,
    };
  } catch {
    return null;
  }
}

export function saveSnakeTestModel(weights: Float64Array): void {
  const model: SnakeModel = { topology: [...SNAKE_TOPOLOGY], weights: [...weights] };
  localStorage.setItem(TEST_KEY, JSON.stringify(model));
}

export function loadSnakeTestModel(): Float64Array | null {
  const text = localStorage.getItem(TEST_KEY);
  if (!text) return null;
  try {
    return validateWeights((JSON.parse(text) as SnakeModel).weights);
  } catch {
    return null;
  }
}
