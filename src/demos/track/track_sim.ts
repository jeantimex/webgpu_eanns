import type { Simulation } from '../../core';
import { createBufferWithData } from '../../webgpu/utils';
import { CAR_FLOATS, initialCarStates } from './buffers';
import { simShader } from './sim.wgsl';
import { buildCheckpointTable, checkpointsFlat, wallsFlat, type Track } from './track';

/** Car sim for one track: physics + sensor raycasts + the NN forward pass run in
 *  the compute shader, one thread per car. */
export function createTrackSim(track: Track): Simulation {
  return {
    agentFloats: CAR_FLOATS,
    shader: simShader,
    initialStates: (count) => initialCarStates(track, count),

    // SimParams uniform: carCount, wallCount, cpCount (u32) + dt (f32) = 16 bytes.
    writeParams(data, evo) {
      const u32 = new Uint32Array(data);
      u32[0] = evo.trainingAgents;
      u32[1] = track.walls.length;
      u32[2] = track.checkpoints.length;
      new Float32Array(data)[3] = 1 / 50;
    },

    extraBuffers: (device, evo) => ({
      walls: createBufferWithData(device, 'walls', wallsFlat(track), GPUBufferUsage.STORAGE),
      checkpoints: createBufferWithData(device, 'checkpoints', checkpointsFlat(track, buildCheckpointTable(track)), GPUBufferUsage.STORAGE),
      /** 5 raw sensor distances per car, written by the sim, read by the renderer. */
      sensors: device.createBuffer({ label: 'sensor distances', size: evo.trainingAgents * 5 * 4, usage: GPUBufferUsage.STORAGE }),
    }),

    isAgentDone: (states, i) => new Uint32Array(states.buffer)[i * CAR_FLOATS + 4] !== 1,

    /** Generation ends when every car is dead (wall or checkpoint timeout). */
    isGenerationOver(states, evo) {
      for (let i = 0; i < evo.trainingAgents; i++) {
        if (!this.isAgentDone(states, i)) return false;
      }
      return true;
    },

    /** Checkpoint progress, written by the shader. */
    fitness(states, evo) {
      const out = new Float64Array(evo.trainingAgents);
      for (let i = 0; i < evo.trainingAgents; i++) out[i] = states[i * CAR_FLOATS + 7];
      return out;
    },
  };
}
