import { defineNetwork } from '../../core';

/**
 * The snake network [24 -> 16 -> 16 -> 4] (SnakeAI). genomeSize is 740
 * (25x16 + 17x16 + 17x4), computed by defineNetwork — the flat genome is
 * layer-major, input-major, bias row last, shared by the GA and NetworkPanel.
 * Output activation is relu only for panel display parity with the old
 * hand-written CPU forward pass; the shader argmaxes the raw outputs.
 */
export const snakeNetwork = defineNetwork([24, 16, 16, 4], { hidden: 'relu', output: 'relu' });
