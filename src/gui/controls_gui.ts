import GUI from 'lil-gui';

const MODE_KEY = 'eanns:mode';
const TRACK_KEY = 'eanns:track';
const POP_KEY = 'eanns:pop';
const SPEED_KEY = 'eanns:speed';

export interface Settings {
  mode: 'Train' | 'Test' | 'Play';
  /** Selected track name, or '' when the demo has no tracks. */
  track: string;
  population: number;
  speed: number;
}

/**
 * Boot settings read from URL parameters (e.g. ?pop=1000&speed=16&track=track2&mode=Train),
 * syncing with localStorage. `tracks` is the demo's track list; demos without tracks
 * (pass nothing) skip the track parameter entirely.
 */
export function currentSettings(tracks?: readonly string[]): Settings {
  const params = new URLSearchParams(window.location.search);

  // 1. Mode: ?mode=Train|Test|Play
  const rawMode = params.get('mode') ?? localStorage.getItem(MODE_KEY);
  const mode: 'Train' | 'Test' | 'Play' = rawMode === 'Play' ? 'Play' : rawMode === 'Test' ? 'Test' : 'Train';
  params.set('mode', mode);
  localStorage.setItem(MODE_KEY, mode);

  // 2. Track: ?track=... (only for demos with tracks)
  let track = '';
  if (tracks && tracks.length > 0) {
    const rawTrack = params.get('track') ?? localStorage.getItem(TRACK_KEY);
    track = rawTrack && tracks.includes(rawTrack) ? rawTrack : tracks[0];
    params.set('track', track);
    localStorage.setItem(TRACK_KEY, track);
  }

  // 3. Population: ?pop=10000 (or ?population=10000), clamped 3..10000
  const rawPop = params.get('pop') ?? params.get('population') ?? localStorage.getItem(POP_KEY) ?? localStorage.getItem('eanns:population');
  const parsedPop = Number(rawPop);
  const population = Math.min(10000, Math.max(3, Number.isFinite(parsedPop) && parsedPop > 0 ? parsedPop : 30));
  params.set('pop', String(population));
  params.delete('population');
  localStorage.setItem(POP_KEY, String(population));

  // 4. Speed: ?speed=1..100
  const rawSpeed = params.get('speed') ?? localStorage.getItem(SPEED_KEY);
  const parsedSpeed = Number(rawSpeed);
  const speed = Math.min(100, Math.max(1, Number.isFinite(parsedSpeed) && parsedSpeed > 0 ? Math.floor(parsedSpeed) : 1));
  params.set('speed', String(speed));
  localStorage.setItem(SPEED_KEY, String(speed));

  // Sync URL parameters cleanly
  const newSearch = `?${params.toString()}`;
  if (window.location.search !== newSearch) {
    window.history.replaceState(null, '', newSearch);
  }

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
  /** Called with the file picked via "Load model file"; the demo decides what to do with it. */
  onLoadModelFile(file: File): void;
}

export interface GuiOptions {
  /** Demo's track list; when set, a track selector is shown and switching reloads with ?track=. */
  tracks?: readonly string[];
  /** Track uses this; demos without camera follow can hide it. */
  showFollowCam?: boolean;
}

/** lil-gui control panel; returns live, non-persisted view/sim options. */
export function setupControls(
  actions: GuiActions,
  options?: GuiOptions,
): {
  speed: number;
  followCam: boolean;
  setFollow(v: boolean): void;
} {
  const current = currentSettings(options?.tracks);
  const controls = { ...current, followCam: false };

  // Hidden file input behind the "Load model file" button.
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.json,.txt,.csv';
  fileInput.style.display = 'none';
  document.body.append(fileInput);
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    fileInput.value = '';
    if (file) actions.onLoadModelFile(file);
  });

  const gui = new GUI({ title: 'EANNs' });
  gui.add(controls, 'mode', ['Train', 'Test', 'Play']).onChange((val: 'Train' | 'Test' | 'Play') => updateSetting('mode', val));
  if (options?.tracks) {
    gui.add(controls, 'track', options.tracks).onChange((val: string) => updateSetting('track', val));
  }
  gui.add(controls, 'population', 3, 10000, 1).name('Population (URL)').onFinishChange((val: number) => updateSetting('pop', val));
  gui.add(controls, 'speed', 1, 100, 1).name('Sim speed').onChange((val: number) => updateUrlParamLive('speed', val));
  const followController = options?.showFollowCam === false ? null : gui.add(controls, 'followCam').name('Follow best car');
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
      followController?.updateDisplay();
    },
  });
}
