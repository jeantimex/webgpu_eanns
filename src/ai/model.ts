import { GENOME_SIZE, TOPOLOGY } from './network';

/** On-disk model format (see README). */
export interface Model {
  topology: number[];
  weights: number[];
  meta?: { track?: string; generation?: number; eval?: number };
}

function validateWeights(weights: unknown): Float64Array {
  if (!Array.isArray(weights) || weights.length !== GENOME_SIZE || !weights.every((w) => Number.isFinite(w)))
    throw new Error(`Model must contain exactly ${GENOME_SIZE} finite weights, got ${Array.isArray(weights) ? weights.length : 'none'}.`);
  return Float64Array.from(weights as number[]);
}

/**
 * Parses both our JSON format and Unity's native genotype format
 * ("p0;p1;…;p46", Genotype.SaveToFile). Returns the 47 genome floats.
 */
export function parseModelText(text: string): Float64Array {
  if (!text.trimStart().startsWith('{')) {
    // Unity genotype: ';'-separated floats, possibly with locale decimal commas.
    return validateWeights(text.trim().split(';').map((p) => Number(p.trim().replace(',', '.'))));
  }
  const model = JSON.parse(text) as Model;
  if (JSON.stringify(model.topology) !== JSON.stringify([...TOPOLOGY]))
    throw new Error(`Unsupported topology [${model.topology}], expected [${[...TOPOLOGY]}].`);
  return validateWeights(model.weights);
}

export function saveModel(weights: Float64Array, meta: Model['meta']): void {
  const model: Model = { topology: [...TOPOLOGY], weights: [...weights], meta };
  const url = URL.createObjectURL(new Blob([JSON.stringify(model, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `eanns-${meta?.track ?? 'model'}-gen${meta?.generation ?? 0}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function loadModelFile(file: File): Promise<Float64Array> {
  return parseModelText(await file.text());
}

const storageKey = (track: string) => `eanns:best:${track}`;

/** Persists the generation's best genotype per track; never downgrades a stored better eval. */
export function autosaveBest(track: string, weights: Float64Array, generation: number, eval_: number): void {
  const saved = loadSavedBest(track);
  if (saved && saved.eval > eval_) return;
  const model: Model = { topology: [...TOPOLOGY], weights: [...weights], meta: { track, generation, eval: eval_ } };
  localStorage.setItem(storageKey(track), JSON.stringify(model));
}

export function loadSavedBest(track: string): { weights: Float64Array; generation: number; eval: number } | null {
  const text = localStorage.getItem(storageKey(track));
  if (!text) return null;
  try {
    const model = JSON.parse(text) as Model;
    return {
      weights: validateWeights(model.weights),
      generation: model.meta?.generation ?? 0,
      eval: model.meta?.eval ?? 0,
    };
  } catch {
    return null; // corrupt entry: treat as absent
  }
}

// Genotype loaded from a file, consumed by Test mode (survives the mode-switch reload).
const TEST_KEY = 'eanns:testModel';

export function saveTestModel(weights: Float64Array): void {
  const model: Model = { topology: [...TOPOLOGY], weights: [...weights] };
  localStorage.setItem(TEST_KEY, JSON.stringify(model));
}

export function loadTestModel(): Float64Array | null {
  const text = localStorage.getItem(TEST_KEY);
  if (!text) return null;
  try {
    return validateWeights((JSON.parse(text) as Model).weights);
  } catch {
    return null;
  }
}
