import { PACMAN_GENOME_SIZE, PACMAN_TOPOLOGY } from './pacman_buffers';

export interface PacmanModel {
  topology: number[];
  weights: number[];
  meta?: { generation?: number; eval?: number; score?: number; level?: number };
}

const STORAGE_KEY = 'eanns:best:pacman';
const TEST_KEY = 'eanns:testModel:pacman';

function validateWeights(weights: unknown): Float64Array {
  if (!Array.isArray(weights) || weights.length !== PACMAN_GENOME_SIZE || !weights.every((w) => Number.isFinite(w))) {
    throw new Error(`Pacman model must contain exactly ${PACMAN_GENOME_SIZE} finite weights.`);
  }
  return Float64Array.from(weights as number[]);
}

export function parsePacmanModelText(text: string): Float64Array {
  const model = JSON.parse(text) as PacmanModel;
  if (JSON.stringify(model.topology) !== JSON.stringify([...PACMAN_TOPOLOGY])) {
    throw new Error(`Unsupported Pacman topology [${model.topology}], expected [${[...PACMAN_TOPOLOGY]}].`);
  }
  return validateWeights(model.weights);
}

export function savePacmanModel(weights: Float64Array, meta: PacmanModel['meta']): void {
  const model: PacmanModel = { topology: [...PACMAN_TOPOLOGY], weights: [...weights], meta };
  const url = URL.createObjectURL(new Blob([JSON.stringify(model, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `eanns-pacman-gen${meta?.generation ?? 0}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function loadPacmanModelFile(file: File): Promise<Float64Array> {
  return parsePacmanModelText(await file.text());
}

export function savePacmanTestModel(weights: Float64Array): void {
  localStorage.setItem(TEST_KEY, JSON.stringify([...weights]));
}

export function loadPacmanTestModel(): Float64Array | null {
  const text = localStorage.getItem(TEST_KEY);
  if (!text) return null;
  try {
    return validateWeights(JSON.parse(text));
  } catch {
    return null;
  }
}

export function autosavePacmanBest(weights: Float64Array, generation: number, eval_: number, score: number, level: number): void {
  const saved = loadSavedPacmanBest();
  if (saved && saved.eval > eval_) return;
  const model: PacmanModel = { topology: [...PACMAN_TOPOLOGY], weights: [...weights], meta: { generation, eval: eval_, score, level } };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(model));
}

export function loadSavedPacmanBest(): { weights: Float64Array; generation: number; eval: number; score: number; level: number } | null {
  const text = localStorage.getItem(STORAGE_KEY);
  if (!text) return null;
  try {
    const model = JSON.parse(text) as PacmanModel;
    return {
      weights: validateWeights(model.weights),
      generation: model.meta?.generation ?? 0,
      eval: model.meta?.eval ?? 0,
      score: model.meta?.score ?? 0,
      level: model.meta?.level ?? 1,
    };
  } catch {
    return null;
  }
}
