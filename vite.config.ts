import { resolve } from 'path';
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        track: resolve(__dirname, 'track.html'),
        flappy: resolve(__dirname, 'flappy-bird.html'),
        dino: resolve(__dirname, 'dino.html'),
        pacman: resolve(__dirname, 'pacman.html'),
        snake: resolve(__dirname, 'snake.html'),
        tetris: resolve(__dirname, 'tetris.html'),
      },
    },
  },
});
