/** Unity FNNTopology. */
export const TOPOLOGY = [5, 4, 3, 2] as const;

/** (5+1)*4 + (4+1)*3 + (3+1)*2, bias row included per layer. */
export const GENOME_SIZE = 47;

/** MathHelper.SoftSignFunction, applied on ALL layers. */
export function softSign(x: number): number {
  return x / (1 + Math.abs(x));
}

/**
 * Forward pass. Genome layout matches Agent.cs:77-88: layer by layer, each layer's
 * (in+1) x out weight matrix row-major (row i = input neuron i; last row = bias,
 * constant input 1.0). Inputs are the 5 raw sensor distances; returns [turn, engine].
 */
export function forward(genome: ArrayLike<number>, inputs: ArrayLike<number>): [number, number] {
  let cur: ArrayLike<number> = inputs;
  let offset = 0;
  for (let l = 0; l < TOPOLOGY.length - 1; l++) {
    const inCount = TOPOLOGY[l];
    const outCount = TOPOLOGY[l + 1];
    const next = new Array<number>(outCount);
    for (let j = 0; j < outCount; j++) {
      let sum = 0;
      for (let i = 0; i < inCount; i++) sum += cur[i] * genome[offset + i * outCount + j];
      sum += genome[offset + inCount * outCount + j]; // bias row, input 1.0
      next[j] = softSign(sum);
    }
    cur = next;
    offset += (inCount + 1) * outCount;
  }
  return [cur[0], cur[1]];
}
