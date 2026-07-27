import { GENOME_SIZE } from './network';

/** Parser for Unity's native genotype format ("p0;p1;…;p46", Genotype.SaveToFile),
 *  kept as the legacy format accepted by "Load model file" alongside JSON. */
export function parseUnityGenotype(text: string): Float64Array {
  // ';'-separated floats, possibly with locale decimal commas.
  const weights = text.trim().split(';').map((p) => Number(p.trim().replace(',', '.')));
  if (weights.length !== GENOME_SIZE || !weights.every((w) => Number.isFinite(w))) {
    throw new Error(`Model must contain exactly ${GENOME_SIZE} finite weights, got ${weights.length}.`);
  }
  return Float64Array.from(weights);
}
