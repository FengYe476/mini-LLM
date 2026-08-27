import argparse
import re

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINE = re.compile(r'step\s+(\d+).*?train loss = ([\d.]+).*?val loss = ([\d.]+).*?bpb = ([\d.]+)')
FINAL = re.compile(r'\[epoch end\].*?step\s+(\d+).*?val loss = ([\d.]+).*?bpb = ([\d.]+)')

THEMES = {
    'light': {
        'surface': '#fcfcfb', 'primary': '#0b0b0b', 'secondary': '#52514e',
        'grid': '#e3e2df', 'train': '#eb6834', 'val': '#2a78d6',
    },
    'dark': {
        'surface': '#1a1a19', 'primary': '#ffffff', 'secondary': '#c3c2b7',
        'grid': '#333330', 'train': '#d95926', 'val': '#3987e5',
    },
}


def parse_log(path: Path) -> tuple[list, list, list, tuple | None]:
    steps, train, val = [], [], []
    final = None
    for line in path.read_text(errors = 'replace', encoding = 'utf-8').splitlines():
        m = LINE.search(line)
        if m and 'epoch end' not in line:
            steps.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
            continue
        f = FINAL.search(line)
        if f:
            final = (int(f.group(1)), float(f.group(2)), float(f.group(3)))
    if not steps:
        raise ValueError(f'[plot]: no "step N | ... train loss = ... val loss = ..." lines found in {path}')
    order = sorted(range(len(steps)), key = lambda i: steps[i])
    return [steps[i] for i in order], [train[i] for i in order], [val[i] for i in order], final


def render(steps, train, val, final, theme: str, out: Path, title: str) -> None:
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize = (9, 4.6), dpi = 200)
    fig.patch.set_facecolor(c['surface'])
    ax.set_facecolor(c['surface'])

    ax.grid(True, color = c['grid'], linewidth = 1, linestyle = '-', zorder = 0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(c['grid'])
        ax.spines[side].set_linewidth(1)

    ax.plot(steps, train, color = c['train'], linewidth = 1.4, alpha = 0.75,
            solid_joinstyle = 'round', solid_capstyle = 'round', zorder = 2,
            label = 'train loss  (one micro-batch, noisy)')
    ax.plot(steps, val, color = c['val'], linewidth = 2, solid_joinstyle = 'round',
            solid_capstyle = 'round', zorder = 3, label = 'val loss  (full split, every 200 steps)')
    ax.plot(steps, val, 'o', color = c['val'], markersize = 4.5,
            markeredgecolor = c['surface'], markeredgewidth = 1.6, zorder = 5)

    last_x, last_v = (final[0], final[1]) if final else (steps[-1], val[-1])
    if final:
        ax.plot([steps[-1], final[0]], [val[-1], final[1]], color = c['val'], linewidth = 2, zorder = 3)
        ax.plot([final[0]], [final[1]], 'o', color = c['val'], markersize = 6,
                markeredgecolor = c['surface'], markeredgewidth = 1.8, zorder = 6)

    ax.annotate(f'{last_v:.4f}', xy = (last_x, last_v), xytext = (12, 0),
                textcoords = 'offset points', color = c['primary'],
                fontsize = 11, fontweight = 'bold', va = 'center')

    ax.set_xlabel('optimizer step  (524,288 tokens each)', color = c['secondary'], fontsize = 10)
    ax.set_ylabel('cross-entropy loss', color = c['secondary'], fontsize = 10)
    ax.set_title(title, color = c['primary'], fontsize = 13, fontweight = 'bold', loc = 'left', pad = 14)
    ax.tick_params(colors = c['secondary'], labelsize = 9, length = 0)
    ax.set_xlim(-150, last_x * 1.075)

    legend = ax.legend(frameon = False, loc = 'upper right', fontsize = 9.5)
    for text in legend.get_texts():
        text.set_color(c['secondary'])

    fig.tight_layout()
    fig.savefig(out, facecolor = c['surface'])
    plt.close(fig)
    return


def main() -> None:
    parser = argparse.ArgumentParser(description = 'plot the pretraining loss curve from a training log')
    parser.add_argument('--log', default = 'train.log')
    parser.add_argument('--out-dir', default = 'docs')
    parser.add_argument('--name', default = 'loss-curve')
    parser.add_argument('--title', default = 'mini-LLM 132M — one epoch over 5.75B tokens')
    args = parser.parse_args()

    steps, train, val, final = parse_log(Path(args.log))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    for theme in THEMES:
        out = out_dir / f'{args.name}-{theme}.png'
        render(steps, train, val, final, theme, out, args.title)
        print(f'[plot]: {out} ({out.stat().st_size // 1024} KB)')
    print(f'[plot]: {len(steps)} evaluation points, steps {steps[0]}..{final[0] if final else steps[-1]}')
    return


if __name__ == '__main__':
    main()
