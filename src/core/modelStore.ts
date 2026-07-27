import type { Network } from './network';

/**
 * Model persistence, consolidated from the per-demo `model.ts` copies.
 * Format: JSON `{ topology, weights, meta? }`; localStorage keys
 * `eanns:best:<namespace>` / `eanns:testModel:<namespace>`.
 * Legacy text formats (SnakeAI CSV, Unity genotypes) stay demo-owned and are
 * passed to `parseModelText` via `legacyParser`.
 */

export interface SavedModel {
  topology: number[];
  weights: number[];
  meta?: Record<string, number | string>;
}

const bestKey = (namespace: string) => `eanns:best:${namespace}`;
// Empty namespace is the track demo's legacy global test slot ('eanns:testModel').
const testKey = (namespace: string) => (namespace ? `eanns:testModel:${namespace}` : 'eanns:testModel');

function validate(net: Network, model: SavedModel): Float64Array {
  if (JSON.stringify(model.topology) !== JSON.stringify([...net.topology])) {
    throw new Error(`Unsupported topology [${model.topology}], expected [${[...net.topology]}].`);
  }
  const { weights } = model;
  if (!Array.isArray(weights) || weights.length !== net.genomeSize || !weights.every((w) => Number.isFinite(w))) {
    throw new Error(`Model must contain exactly ${net.genomeSize} finite weights.`);
  }
  return Float64Array.from(weights);
}

/** Parse a model file; throws on invalid input. Non-JSON text goes to `legacyParser`. */
export function parseModelText(
  text: string,
  net: Network,
  legacyParser?: (text: string) => Float64Array,
): Float64Array {
  if (!text.trimStart().startsWith('{')) {
    if (!legacyParser) throw new Error('Not a JSON model file.');
    const weights = legacyParser(text);
    if (weights.length !== net.genomeSize) {
      throw new Error(`Legacy model has ${weights.length} weights, expected ${net.genomeSize}.`);
    }
    return weights;
  }
  return validate(net, JSON.parse(text) as SavedModel);
}

/** Download the genome as a JSON model file. */
export function downloadModel(
  namespace: string,
  net: Network,
  genome: Float64Array,
  meta?: Record<string, number | string>,
): void {
  const model: SavedModel = { topology: [...net.topology], weights: [...genome], meta };
  const url = URL.createObjectURL(new Blob([JSON.stringify(model, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `eanns-${namespace}-gen${meta?.generation ?? 0}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Autosave a new best genome, unless localStorage already holds a better one
 * (a previous session's champion must survive a fresh page's early generations).
 */
export function autosaveBestModel(
  namespace: string,
  net: Network,
  genome: Float64Array,
  fitness: number,
  meta?: Record<string, number | string>,
): void {
  const saved = loadBestModel(namespace, net);
  const savedEval = saved?.meta?.eval;
  if (typeof savedEval === 'number' && savedEval > fitness) return;
  const model: SavedModel = {
    topology: [...net.topology],
    weights: [...genome],
    meta: { ...meta, eval: fitness },
  };
  localStorage.setItem(bestKey(namespace), JSON.stringify(model));
}

export function loadBestModel(namespace: string, net: Network): SavedModel | null {
  const text = localStorage.getItem(bestKey(namespace));
  if (!text) return null;
  try {
    const model = JSON.parse(text) as SavedModel;
    validate(net, model);
    return model;
  } catch {
    return null;
  }
}

export function saveTestModel(namespace: string, net: Network, genome: Float64Array): void {
  const model: SavedModel = { topology: [...net.topology], weights: [...genome] };
  localStorage.setItem(testKey(namespace), JSON.stringify(model));
}

export function loadTestModel(namespace: string, net: Network): Float64Array | null {
  const text = localStorage.getItem(testKey(namespace));
  if (!text) return null;
  try {
    const parsed = JSON.parse(text) as SavedModel | number[];
    // Legacy pacman test models were stored as a bare weight array.
    const model: SavedModel = Array.isArray(parsed) ? { topology: [...net.topology], weights: parsed } : parsed;
    return validate(net, model);
  } catch {
    return null;
  }
}
