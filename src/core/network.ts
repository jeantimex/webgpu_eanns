/**
 * Network definition: the single source of truth for topology, genome size, and
 * the flat genome layout. Replaces the per-demo hand-counted `*_GENOME_SIZE`
 * constants and hand-written CPU forward passes.
 *
 * Genome layout (shared convention of every demo and the WGSL shaders):
 * layer-major; within a layer, the weight matrix is input-major (all outputs for
 * input 0, then input 1, …) with the bias row last:
 *
 *     genome[offset + k * out + j]   weight from input k to output j
 *     genome[offset + in * out + j]  bias of output j
 */

export type Activation = 'identity' | 'relu' | 'sigmoid' | 'tanh' | 'softsign';

export interface LayerLayout {
  /** Offset of this layer's block inside the flat genome. */
  offset: number;
  in: number;
  out: number;
  activation: Activation;
}

export interface Network {
  topology: readonly number[];
  /** Computed: sum over layers of (in + 1) * out. */
  genomeSize: number;
  layers: readonly LayerLayout[];
}

export function defineNetwork(
  topology: readonly number[],
  opts: { hidden?: Activation; output?: Activation } = {},
): Network {
  if (topology.length < 2) throw new Error('Network needs at least input and output layers.');
  const { hidden = 'relu', output = 'identity' } = opts;
  const layers: LayerLayout[] = [];
  let offset = 0;
  for (let l = 0; l < topology.length - 1; l++) {
    const input = topology[l];
    const out = topology[l + 1];
    if (input < 1 || out < 1) throw new Error(`Invalid topology [${topology}].`);
    layers.push({ offset, in: input, out, activation: l === topology.length - 2 ? output : hidden });
    offset += (input + 1) * out;
  }
  return { topology, genomeSize: offset, layers };
}

const ACTIVATIONS: Record<Activation, (x: number) => number> = {
  identity: (x) => x,
  relu: (x) => Math.max(0, x),
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  tanh: (x) => Math.tanh(x),
  softsign: (x) => x / (1 + Math.abs(x)),
};

/** CPU forward pass — replaces the per-demo hand-written copies. */
export function forwardCPU(net: Network, genome: ArrayLike<number>, inputs: ArrayLike<number>): Float64Array {
  if (inputs.length !== net.topology[0]) {
    throw new Error(`Expected ${net.topology[0]} inputs, got ${inputs.length}.`);
  }
  let current = Float64Array.from(inputs);
  for (const layer of net.layers) {
    const next = new Float64Array(layer.out);
    const act = ACTIVATIONS[layer.activation];
    for (let j = 0; j < layer.out; j++) {
      let sum = genome[layer.offset + layer.in * layer.out + j];
      for (let k = 0; k < layer.in; k++) sum += current[k] * genome[layer.offset + k * layer.out + j];
      next[j] = act(sum);
    }
    current = next;
  }
  return current;
}

export interface LayerWeights {
  /** in x out, input-major (same order as the genome block, minus the bias row). */
  weights: Float64Array;
  biases: Float64Array;
}

/** TorchGA's model_weights_as_dict: flat genome → per-layer tensors. */
export function unflattenWeights(net: Network, genome: ArrayLike<number>): LayerWeights[] {
  if (genome.length !== net.genomeSize) {
    throw new Error(`Expected genome of ${net.genomeSize} weights, got ${genome.length}.`);
  }
  return net.layers.map((layer) => ({
    weights: Float64Array.from({ length: layer.in * layer.out }, (_, i) => genome[layer.offset + i]),
    biases: Float64Array.from({ length: layer.out }, (_, j) => genome[layer.offset + layer.in * layer.out + j]),
  }));
}

/** TorchGA's model_weights_as_vector: per-layer tensors → flat genome. */
export function flattenWeights(net: Network, layers: readonly LayerWeights[]): Float64Array {
  const genome = new Float64Array(net.genomeSize);
  net.layers.forEach((layer, l) => {
    const src = layers[l];
    if (src.weights.length !== layer.in * layer.out || src.biases.length !== layer.out) {
      throw new Error(`Layer ${l} shape mismatch for topology [${net.topology}].`);
    }
    genome.set(src.weights, layer.offset);
    genome.set(src.biases, layer.offset + layer.in * layer.out);
  });
  return genome;
}
