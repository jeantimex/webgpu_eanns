import { defineNetwork } from '../../core';

/** Dino network [6 -> 8 -> 1]: relu hidden, sigmoid output. genomeSize = 65. */
export const dinoNetwork = defineNetwork([6, 8, 1], { hidden: 'relu', output: 'sigmoid' });
