import { hydrateIcons } from './icons';

// The floating interface around the stage: which panels are open (remembered between
// visits), hiding everything to see only the stage, and the glass tone following the stage.

const STORE_KEY = 'facial-ui';

interface ChromeState {
  scene: boolean;
  tracks: boolean;
  stats: boolean;
}

function loadState(): ChromeState {
  const state: ChromeState = { scene: true, tracks: true, stats: false };
  try {
    Object.assign(state, JSON.parse(localStorage.getItem(STORE_KEY) ?? '{}'));
  } catch {
    // Unreadable or blocked storage: start from the defaults.
  }
  return state;
}

/** Glass is light over the white stage and dark over the dark one. */
export function setStageTone(dark: boolean): void {
  document.documentElement.dataset.stage = dark ? 'dark' : 'light';
  window.dispatchEvent(new Event('stagetone'));
}

type LayoutListener = (coveredTop: number, coveredBottom: number) => void;

/** `onLayout` reports how much of the window's top and bottom the interface covers (CSS px). */
export function setupChrome(): { onLayout(cb: LayoutListener): void } {
  hydrateIcons();
  const body = document.body;
  const $ = (id: string) => document.getElementById(id) as HTMLElement;
  const state = loadState();

  const apply = () => {
    body.classList.toggle('no-scene', !state.scene);
    body.classList.toggle('no-tracks', !state.tracks);
    $('info').hidden = !state.stats;
    $('toggle-scene').setAttribute('aria-pressed', String(state.scene));
    $('toggle-timeline').setAttribute('aria-pressed', String(state.tracks));
    $('toggle-stats').setAttribute('aria-pressed', String(state.stats));
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(state));
    } catch {
      // Not remembered; nothing else depends on it.
    }
  };
  const toggle = (key: keyof ChromeState) => () => {
    state[key] = !state[key];
    apply();
  };
  $('toggle-scene').addEventListener('click', toggle('scene'));
  $('toggle-timeline').addEventListener('click', toggle('tracks'));
  $('toggle-stats').addEventListener('click', toggle('stats'));
  apply();

  let layout: LayoutListener | null = null;
  const dock = $('timeline');
  const topbar = $('topbar');
  const report = () => {
    if (body.classList.contains('immersive')) layout?.(0, 0);
    else layout?.(topbar.getBoundingClientRect().bottom, window.innerHeight - dock.getBoundingClientRect().top);
  };

  const setImmersive = (on: boolean) => {
    body.classList.toggle('immersive', on);
    report();
    if (on) ($('view-settings') as HTMLElement & { hidePopover?: () => void }).hidePopover?.();
  };
  $('hide-ui').addEventListener('click', () => setImmersive(true));
  $('show-ui').addEventListener('click', () => setImmersive(false));
  window.addEventListener('keydown', (e) => {
    const t = e.target;
    if (t instanceof HTMLInputElement || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) return;
    if (t instanceof Element && t.closest('dialog')) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.code === 'KeyH') setImmersive(!body.classList.contains('immersive'));
    else if (e.code === 'Escape' && body.classList.contains('immersive')) setImmersive(false);
  });

  // Things anchored above the timeline dock follow its height.
  new ResizeObserver(() => {
    body.style.setProperty('--dock-h', `${dock.offsetHeight}px`);
    report();
  }).observe(dock);
  window.addEventListener('resize', report);

  return {
    onLayout(cb) {
      layout = cb;
      report();
    },
  };
}
