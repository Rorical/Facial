// Inline SVG icons (24 × 24, stroked with currentColor).

// Drawn after the system symbol set: round caps, solid transport glyphs.
const SOLID = 'fill="currentColor" stroke-width="1.6"';
const ICONS = {
  play: `<path d="M8.2 5.6v12.8L18.6 12z" ${SOLID}/>`,
  pause: '<rect x="6.2" y="5" width="4.2" height="14" rx="1.4" fill="currentColor" stroke="none"/><rect x="13.6" y="5" width="4.2" height="14" rx="1.4" fill="currentColor" stroke="none"/>',
  toStart: `<rect x="4.6" y="5.4" width="2.8" height="13.2" rx="1.4" fill="currentColor" stroke="none"/><path d="M18.6 6.6v10.8L10 12z" ${SOLID}/>`,
  toEnd: `<rect x="16.6" y="5.4" width="2.8" height="13.2" rx="1.4" fill="currentColor" stroke="none"/><path d="M5.4 6.6v10.8L14 12z" ${SOLID}/>`,
  stepBack: '<path d="M14.5 6l-6 6 6 6" stroke-width="2.3"/>',
  stepForward: '<path d="M9.5 6l6 6-6 6" stroke-width="2.3"/>',
  prevKey: `<path d="M8.5 7l-5 5 5 5" stroke-width="2.3"/><path d="M16.2 8.2L20 12l-3.8 3.8L12.4 12z" ${SOLID}/>`,
  nextKey: `<path d="M15.5 7l5 5-5 5" stroke-width="2.3"/><path d="M7.8 8.2L11.6 12l-3.8 3.8L4 12z" ${SOLID}/>`,
  loop: '<path d="M16.5 2.5l4 4-4 4"/><path d="M3.5 11v-.5a4 4 0 0 1 4-4h13"/><path d="M7.5 21.5l-4-4 4-4"/><path d="M20.5 13v.5a4 4 0 0 1-4 4h-13"/>',
  rangeIn: '<path d="M10 4.5H6.5v15H10"/><path d="M12 12h7"/>',
  rangeOut: '<path d="M14 4.5h3.5v15H14"/><path d="M5 12h7"/>',
  close: '<path d="M7 7l10 10M17 7L7 17" stroke-width="2.3"/>',
  plus: '<path d="M12 5.5v13M5.5 12h13" stroke-width="2.3"/>',
  minus: '<path d="M5.5 12h13" stroke-width="2.3"/>',
  fit: '<path d="M7.5 8l-4 4 4 4M16.5 8l4 4-4 4M4 12h16"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3.2" fill="currentColor" stroke="none"/>',
  eyeOff: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z" opacity=".45"/><circle cx="12" cy="12" r="3.2" fill="currentColor" stroke="none" opacity=".45"/><path d="M4.5 4l15 16" stroke-width="2.2"/>',
  refresh: '<path d="M20 12a8 8 0 1 1-2.6-5.9"/><path d="M20.2 3.5v5h-5"/>',
  sidebar: '<rect x="2.8" y="4.5" width="18.4" height="15" rx="4"/><path d="M9.2 4.5v15"/><path d="M5.4 8.5h1.4M5.4 11.5h1.4" stroke-width="1.6"/>',
  tracks: '<rect x="2.8" y="4.5" width="18.4" height="15" rx="4"/><rect x="5.6" y="13" width="12.8" height="3.8" rx="1.6" fill="currentColor" stroke="none"/>',
  stats: '<rect x="4" y="11" width="4" height="9" rx="1.5" fill="currentColor" stroke="none"/><rect x="10" y="4" width="4" height="16" rx="1.5" fill="currentColor" stroke="none"/><rect x="16" y="14" width="4" height="6" rx="1.5" fill="currentColor" stroke="none"/>',
  sliders: '<path d="M3.5 6.5h8M17 6.5h3.5M3.5 12h2M11 12h9.5M3.5 17.5h10M19 17.5h1.5"/><circle cx="14.2" cy="6.5" r="2.3"/><circle cx="8.2" cy="12" r="2.3"/><circle cx="16.2" cy="17.5" r="2.3"/>',
  expand: '<path d="M14 4.5h5.5V10M19.5 4.5L13 11M10 19.5H4.5V14M4.5 19.5L11 13"/>',
  collapse: '<path d="M19.5 4.5L13 11M13 5.5V11h5.5M4.5 19.5L11 13M11 18.5V13H5.5"/>',
} as const;

export type IconName = keyof typeof ICONS;

export function icon(name: IconName): SVGSVGElement {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  svg.classList.add('icon');
  svg.innerHTML = ICONS[name];
  return svg;
}

/** Icon-only button; the title doubles as the accessible name. */
export function iconButton(name: IconName, title: string, onClick: (e: MouseEvent) => void, className = 'icon-btn'): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = className;
  setIcon(b, name, title);
  b.addEventListener('click', onClick);
  return b;
}

export function setIcon(button: HTMLElement, name: IconName, title?: string): void {
  button.replaceChildren(icon(name));
  if (title !== undefined) {
    button.title = title;
    button.setAttribute('aria-label', title);
  }
}

/** Fill every `[data-icon]` element in the static markup. */
export function hydrateIcons(root: ParentNode = document): void {
  for (const el of root.querySelectorAll<HTMLElement>('[data-icon]')) {
    const name = el.dataset.icon as IconName;
    if (name in ICONS) setIcon(el, name, el.title || undefined);
  }
}
