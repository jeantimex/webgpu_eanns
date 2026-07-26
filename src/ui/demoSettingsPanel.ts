import { updateSetting, updateUrlParamLive, type Settings } from '../gui/controls_gui';

export interface DemoSettingsActions {
  onSaveModel(): void;
  onLoadSavedBest(): void;
  onLoadModelFile(file: File): void;
}

/** A checkbox a demo adds for itself, e.g. a debug overlay switch. */
export interface DemoSettingsToggle {
  label: string;
  initial?: boolean;
  onChange: (enabled: boolean) => void;
}

export interface DemoSettingsOptions {
  tracks?: readonly string[];
  showFps?: boolean;
  initialShowFps?: boolean;
  onFpsChange?: (visible: boolean) => void;
  /** Demo-specific switches, rendered under the shared fields. */
  toggles?: readonly DemoSettingsToggle[];
}

export interface DemoSettingsControls {
  speed: number;
  showFps: boolean;
}

export function createDemoSettingsPanel(
  settings: Settings,
  actions: DemoSettingsActions,
  options: DemoSettingsOptions = {},
): DemoSettingsControls {
  const controls = { speed: settings.speed, showFps: options.initialShowFps ?? false };
  const panel = document.createElement('aside');
  panel.className = 'snake-settings-panel';

  const toggle = document.createElement('button');
  toggle.className = 'snake-settings-toggle';
  toggle.type = 'button';
  toggle.title = 'Collapse settings';
  const toggleIcon = document.createElement('span');
  toggleIcon.className = 'material-chevron chevron-right';
  toggle.append(toggleIcon);
  toggle.addEventListener('click', () => {
    panel.classList.toggle('collapsed');
    const collapsed = panel.classList.contains('collapsed');
    document.body.classList.toggle('snake-settings-collapsed', collapsed);
    toggleIcon.className = `material-chevron ${collapsed ? 'chevron-left' : 'chevron-right'}`;
    toggle.title = collapsed ? 'Expand settings' : 'Collapse settings';
  });

  const resizeHandle = document.createElement('div');
  resizeHandle.className = 'snake-settings-resize';
  resizeHandle.title = 'Resize settings';
  resizeHandle.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    resizeHandle.setPointerCapture(event.pointerId);
    document.body.classList.add('snake-resizing');
  });
  resizeHandle.addEventListener('pointermove', (event) => {
    if (!resizeHandle.hasPointerCapture(event.pointerId)) return;
    const width = Math.max(220, Math.min(window.innerWidth * 0.42, window.innerWidth - event.clientX));
    document.body.style.setProperty('--snake-settings-open-width', `${Math.round(width)}px`);
  });
  const endResize = (event: PointerEvent): void => {
    if (resizeHandle.hasPointerCapture(event.pointerId)) resizeHandle.releasePointerCapture(event.pointerId);
    document.body.classList.remove('snake-resizing');
  };
  resizeHandle.addEventListener('pointerup', endResize);
  resizeHandle.addEventListener('pointercancel', endResize);

  const content = document.createElement('div');
  content.className = 'snake-settings-content';
  const header = document.createElement('div');
  header.className = 'snake-settings-header';
  const title = document.createElement('h2');
  title.textContent = 'Settings';
  header.append(toggle, title);

  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.json,.txt,.csv';
  fileInput.style.display = 'none';
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    fileInput.value = '';
    if (file) actions.onLoadModelFile(file);
  });

  const field = (labelText: string, control: HTMLElement): HTMLLabelElement => {
    const label = document.createElement('label');
    label.className = 'snake-settings-field';
    const span = document.createElement('span');
    span.textContent = labelText;
    label.append(span, control);
    return label;
  };

  const mode = document.createElement('select');
  for (const value of ['Train', 'Test', 'Play'] as const) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    option.selected = settings.mode === value;
    mode.append(option);
  }
  mode.addEventListener('change', () => updateSetting('mode', mode.value));

  const fields: HTMLElement[] = [header, field('Mode', mode)];

  if (options.tracks && options.tracks.length > 0) {
    const track = document.createElement('select');
    for (const value of options.tracks) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      option.selected = settings.track === value;
      track.append(option);
    }
    track.addEventListener('change', () => updateSetting('track', track.value));
    fields.push(field('Track', track));
  }

  const population = document.createElement('input');
  population.type = 'number';
  population.min = '3';
  population.max = '10000';
  population.step = '1';
  population.value = String(settings.population);
  population.addEventListener('change', () => updateSetting('pop', Math.max(3, Math.min(10000, Number(population.value) || settings.population))));
  fields.push(field('Population', population));

  const speed = document.createElement('input');
  speed.type = 'range';
  speed.min = '1';
  speed.max = '100';
  speed.step = '1';
  speed.value = String(settings.speed);
  const speedValue = document.createElement('output');
  speedValue.textContent = String(settings.speed);
  speed.addEventListener('input', () => {
    controls.speed = Number(speed.value);
    speedValue.textContent = speed.value;
    updateUrlParamLive('speed', speed.value);
  });
  const speedWrap = document.createElement('div');
  speedWrap.className = 'snake-settings-range';
  speedWrap.append(speed, speedValue);
  fields.push(field('Sim speed', speedWrap));

  if (options.showFps) {
    const showFps = document.createElement('input');
    showFps.type = 'checkbox';
    showFps.checked = controls.showFps;
    showFps.addEventListener('change', () => {
      controls.showFps = showFps.checked;
      options.onFpsChange?.(controls.showFps);
    });
    const fpsField = field('Show FPS', showFps);
    fpsField.classList.add('snake-settings-check');
    fields.push(fpsField);
  }

  for (const toggle of options.toggles ?? []) {
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = toggle.initial ?? false;
    box.addEventListener('change', () => toggle.onChange(box.checked));
    const toggleField = field(toggle.label, box);
    toggleField.classList.add('snake-settings-check');
    fields.push(toggleField);
    toggle.onChange(box.checked);
  }

  const buttons = document.createElement('div');
  buttons.className = 'snake-settings-buttons';
  const button = (text: string, onClick: () => void): HTMLButtonElement => {
    const el = document.createElement('button');
    el.type = 'button';
    el.textContent = text;
    el.addEventListener('click', onClick);
    return el;
  };
  buttons.append(
    button('Save best model', actions.onSaveModel),
    button('Load model file', () => fileInput.click()),
    button('Load saved best', actions.onLoadSavedBest),
  );

  content.append(...fields, buttons, fileInput);
  panel.append(resizeHandle, content);
  document.body.append(panel);
  return controls;
}

export function createFpsDisplay(): { update(now: number): void; setVisible(visible: boolean): void } {
  const el = document.createElement('div');
  el.className = 'snake-fps';
  document.body.append(el);
  let last = performance.now();
  let frames = 0;
  return {
    update(now) {
      frames++;
      if (now - last < 500) return;
      el.textContent = `${Math.round((frames * 1000) / (now - last))} FPS`;
      frames = 0;
      last = now;
    },
    setVisible(visible) {
      el.classList.toggle('visible', visible);
    },
  };
}
