import { defineNetwork } from '../../core';
import { PACMAN_TOPOLOGY } from './pacman_buffers';

/**
 * Pac-Man network [14 -> 6 -> 4] (EANN-Pacman writeup §2.2): one hidden ReLU
 * layer; the 4 outputs become a softmax distribution the shader *samples*.
 * genomeSize = 82.
 */
export const pacmanNetwork = defineNetwork(PACMAN_TOPOLOGY, { hidden: 'relu' });
