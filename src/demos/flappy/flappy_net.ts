import { defineNetwork } from '../../core';

/** Flappy network [5 -> 8 -> 1]: relu hidden, sigmoid output (TF.js original). genomeSize = 57. */
export const flappyNetwork = defineNetwork([5, 8, 1], { hidden: 'relu', output: 'sigmoid' });
