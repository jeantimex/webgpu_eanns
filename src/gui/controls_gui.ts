import GUI from 'lil-gui';
import { loadModelFile, saveTestModel } from '../ai/model';

/** Available tracks (public/tracks/<name>.json); adding one is a line here + the JSON file. */
export const TRACKS = ['track1', 'track2', 'track3', 'track4', 'practice'];

const MODE_KEY = 'eanns:mode';
const TRACK_KEY = 'eanns:track';
const POPULATION_KEY = 'eanns:population';

/** Persist mode without a reload (main uses it for the no-model fallback). */
export function persistMode(mode: 'Train' | 'Test'): void {
  localStorage.setItem(MODE_KEY, mode);
}

export interface Settings {
  mode: 'Train' | 'Test';
  track: string;
  population: number;
}

/** Persisted boot settings (mode/track/population changes reload the page). */
export function currentSettings(): Settings {
  const trackName = localStorage.getItem(TRACK_KEY);
  return {
    mode: localStorage.getItem(MODE_KEY) === 'Test' ? 'Test' : 'Train',
    track: trackName && TRACKS.includes(trackName) ? trackName : TRACKS[0],
    // GA needs at least 3 (best 3 form the intermediate population).
    population: Math.min(500, Math.max(3, Number(localStorage.getItem(POPULATION_KEY)) || 30)),
  };
}

export interface GuiActions {
  onSaveModel(): void;
  onLoadSavedBest(): void;
}

/** lil-gui control panel; returns live, non-persisted view/sim options. */
export function setupControls(actions: GuiActions): {
  speed: number;
  followCam: boolean;
  setFollow(v: boolean): void;
} {
  const controls = { ...currentSettings(), speed: 1, followCam: true };

  // Hidden file input behind the "Load model file" button.
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.json,.txt';
  fileInput.style.display = 'none';
  document.body.append(fileInput);
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    fileInput.value = '';
    if (!file) return;
    loadModelFile(file)
      .then((weights) => {
        saveTestModel(weights);
        localStorage.setItem(MODE_KEY, 'Test');
        location.reload();
      })
      .catch((error: unknown) => alert(error instanceof Error ? error.message : String(error)));
  });

  const gui = new GUI({ title: 'EANNs' });
  const reloadWith = (key: string) => (value: string | number) => {
    localStorage.setItem(key, String(value));
    location.reload();
  };
  gui.add(controls, 'mode', ['Train', 'Test']).onChange(reloadWith(MODE_KEY));
  gui.add(controls, 'track', TRACKS).onChange(reloadWith(TRACK_KEY));
  gui.add(controls, 'population', 3, 500, 1).onFinishChange(reloadWith(POPULATION_KEY));
  gui.add(controls, 'speed', 1, 64, 1).name('Sim speed');
  const followController = gui.add(controls, 'followCam').name('Follow best car');
  const buttons = {
    saveBestModel: actions.onSaveModel,
    loadModelFile: () => fileInput.click(),
    loadSavedBest: actions.onLoadSavedBest,
  };
  gui.add(buttons, 'saveBestModel').name('Save best model');
  gui.add(buttons, 'loadModelFile').name('Load model file (Test mode)');
  gui.add(buttons, 'loadSavedBest').name('Load saved best');
  return Object.assign(controls, {
    setFollow: (v: boolean) => {
      controls.followCam = v;
      followController.updateDisplay();
    },
  });
}
