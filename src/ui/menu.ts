// Top menu bar with dropdowns (文件 / 导入 / 导出). Items are re-evaluated for
// enabled-state each time a menu opens.

export type MenuItem =
  | { label: string; shortcut?: string; action: () => void; enabled?: () => boolean }
  | 'separator';

export interface Menu {
  label: string;
  items: MenuItem[];
}

export class MenuBar {
  private open: { button: HTMLButtonElement; panel: HTMLDivElement } | null = null;

  constructor(root: HTMLElement, menus: Menu[]) {
    for (const menu of menus) {
      const wrap = document.createElement('div');
      wrap.className = 'menu';
      const button = document.createElement('button');
      button.className = 'menu-button';
      button.textContent = menu.label;
      const panel = document.createElement('div');
      panel.className = 'menu-panel';
      panel.hidden = true;
      wrap.append(button, panel);
      root.appendChild(wrap);

      const show = () => {
        this.close();
        this.fill(panel, menu.items);
        panel.hidden = false;
        button.classList.add('active');
        this.open = { button, panel };
      };
      button.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.open?.panel === panel) this.close();
        else show();
      });
      // Moving across the bar while a menu is open switches menus, like a native menu bar.
      button.addEventListener('pointerenter', () => {
        if (this.open && this.open.panel !== panel) show();
      });
    }
    document.addEventListener('click', () => this.close());
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.close();
    });
  }

  private fill(panel: HTMLDivElement, items: MenuItem[]): void {
    panel.replaceChildren();
    for (const item of items) {
      if (item === 'separator') {
        panel.appendChild(Object.assign(document.createElement('div'), { className: 'menu-sep' }));
        continue;
      }
      const b = document.createElement('button');
      b.className = 'menu-item';
      b.disabled = item.enabled ? !item.enabled() : false;
      const label = document.createElement('span');
      label.textContent = item.label;
      b.appendChild(label);
      if (item.shortcut) {
        const k = document.createElement('kbd');
        k.textContent = item.shortcut;
        b.appendChild(k);
      }
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        this.close();
        item.action();
      });
      panel.appendChild(b);
    }
  }

  close(): void {
    if (!this.open) return;
    this.open.panel.hidden = true;
    this.open.button.classList.remove('active');
    this.open = null;
  }
}

/** Open a native file picker; resolves with the chosen files (empty if cancelled). */
export function pickFiles(options: { accept?: string; multiple?: boolean; directory?: boolean }): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    if (options.accept) input.accept = options.accept;
    input.multiple = options.multiple ?? false;
    if (options.directory) input.webkitdirectory = true;
    input.addEventListener('change', () => resolve(Array.from(input.files ?? [])));
    input.addEventListener('cancel', () => resolve([]));
    input.click();
  });
}
