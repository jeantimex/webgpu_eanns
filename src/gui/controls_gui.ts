import GUI from 'lil-gui';
import { loadModelFile, saveTestModel } from '../ai/model';

/** Available tracks (public/tracks/<name>.json); adding one is a line here + the JSON file. */
export const TRACKS = ['track1', 'track2', 'track3', 'track4', 'practice'];

const MODE_KEY = 'eanns:mode';
const TRACK_KEY = 'eanns:track';
const POP_KEY = 'eanns:pop';
const SPEED_KEY = 'eanns:speed';

export interface Settings {
  mode: 'Train' | 'Test';
  track: string;
  population: number;
  speed: number;
}

/** Boot settings read from URL parameters (e.g. ?pop=1000&speed=16&track=track2&mode=Train), syncing with localStorage. */
export function currentSettings(): Settings {
  const params = new URLSearchParams(window.location.search);

  // 1. Mode: ?mode=Train|Test
  const rawMode = params.get('mode') ?? localStorage.getItem(MODE_KEY);
  const mode: 'Train' | 'Test' = rawMode === 'Test' ? 'Test' : 'Train';

  // 2. Track: ?track=track1..4|practice
  const rawTrack = params.get('track') ?? localStorage.getItem(TRACK_KEY);
  const track = rawTrack && TRACKS.includes(rawTrack) ? rawTrack : TRACKS[0];

  // 3. Population: ?pop=10000 (or ?population=10000), clamped 3..10000
  const rawPop = params.get('pop') ?? params.get('population') ?? localStorage.getItem(POP_KEY) ?? localStorage.getItem('eanns:population');
  const parsedPop = Number(rawPop);
  const population = Math.min(10000, Math.max(3, Number.isFinite(parsedPop) && parsedPop > 0 ? parsedPop : 30));

  // 4. Speed: ?speed=1..64
  const rawSpeed = params.get('speed') ?? localStorage.getItem(SPEED_KEY);
  const parsedSpeed = Number(rawSpeed);
  const speed = Math.min(64, Math.max(1, Number.isFinite(parsedSpeed) && parsedSpeed > 0 ? Math.floor(parsedSpeed) : 1));

  // Sync URL parameters cleanly
  params.set('mode', mode);
  params.set('track', track);
  params.set('pop', String(population));
  params.set('speed', String(speed));
  params.delete('population');

  const newSearch = `?${params.toString()}`;
  if (window.location.search !== newSearch) {
    window.history.replaceState(null, '', newSearch);
  }

  // Backup to localStorage
  localStorage.setItem(MODE_KEY, mode);
  localStorage.setItem(TRACK_KEY, track);
  localStorage.setItem(POP_KEY, String(population));
  localStorage.setItem(SPEED_KEY, String(speed));

  return { mode, track, population, speed };
}

export function updateSetting(key: 'mode' | 'track' | 'pop' | 'speed', value: string | number): void {
  const params = new URLSearchParams(window.location.search);
  params.set(key, String(value));
  if (key === 'pop') params.delete('population');
  localStorage.setItem(`eanns:${key === 'pop' ? 'population' : key}`, String(value));
  window.location.search = params.toString();
}

/** Update a URL parameter in-place without triggering a page reload (e.g. for sim speed slider). */
export function updateUrlParamLive(key: string, value: string | number): void {
  const params = new URLSearchParams(window.location.search);
  params.set(key, String(value));
  window.history.replaceState(null, '', `?${params.toString()}`);
  localStorage.setItem(`eanns:${key === 'pop' ? 'population' : key}`, String(value));
}

/** Persist mode without a reload (main uses it for the no-model fallback). */
export function persistMode(mode: 'Train' | 'Test'): void {
  updateUrlParamLive('mode', mode);
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
  const current = currentSettings();
  const controls = { ...current, followCam: false };

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
        updateSetting('mode', 'Test');
      })
      .catch((error: unknown) => alert(error instanceof Error ? error.message : String(error)));
  });

  const gui = new GUI({ title: 'EANNs' });
  gui.add(controls, 'mode', ['Train', 'Test']).onChange((val: 'Train' | 'Test') => updateSetting('mode', val));
  gui.add(controls, 'track', TRACKS).onChange((val: string) => updateSetting('track', val));
  gui.add(controls, 'population', 3, 10000, 1).name('Population (URL)').onFinishChange((val: number) => updateSetting('pop', val));
  gui.add(controls, 'speed', 1, 64, 1).name('Sim speed').onChange((val: number) => updateUrlParamLive('speed', val));
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
