"""
Figura 2: Resultados del experimento piloto (tres paneles).
Panel 1 — Pérdida de test.
Panel 2 — Perplejidad.
Panel 3 — Brecha de generalización (gap = test loss − train loss).
Estilo: serif/CM, paleta teal–magenta, spines limpios.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family']      = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth']   = 0.8

MEDIUM = '#1FB6B6'   # teal
HIGH   = '#C9189E'   # magenta
WIDTH  = 0.32
ALPHA  = 0.88

ARCHS  = ['Transformer', 'LLaMA-style']
x      = np.arange(len(ARCHS))

# ── Datos ─────────────────────────────────────────────────────────────────────
test_med  = [2.0744, 1.6651]
test_high = [2.4824, 2.1373]

ppl_med   = [7.96,  5.29]
ppl_high  = [11.97, 8.48]

gap_med   = [-0.0322, 0.0410]
gap_high  = [ 0.2471, 0.3652]

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
fig.subplots_adjust(wspace=0.42)

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9, length=3)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(ARCHS, fontsize=9.5)

def add_labels(ax, bars, fmt, offset_pos, offset_neg=None):
    for bar in bars:
        h = bar.get_height()
        if h >= 0:
            ypos = h + offset_pos
            va = 'bottom'
        else:
            ypos = h + (offset_neg if offset_neg is not None else -offset_pos * 2.5)
            va = 'top'
        ax.text(bar.get_x() + bar.get_width() / 2,
                ypos, fmt.format(h),
                ha='center', va=va, fontsize=7.5, color='#333333')

# ── Panel 1: Pérdida de test ──────────────────────────────────────────────────
ax = axes[0]
b1 = ax.bar(x - WIDTH/2, test_med,  WIDTH, label='Medium',
            color=MEDIUM, alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)
b2 = ax.bar(x + WIDTH/2, test_high, WIDTH, label='High',
            color=HIGH,   alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)
ax.set_ylim(0, 3.05)
ax.set_ylabel('Pérdida de test (entropía cruzada)', fontsize=9.5)
ax.set_title('Pérdida de test', fontsize=11, pad=8)
ax.legend(fontsize=9, framealpha=0.0)
style_ax(ax)
add_labels(ax, b1, '{:.4f}', 0.03)
add_labels(ax, b2, '{:.4f}', 0.03)

# ── Panel 2: Perplejidad ──────────────────────────────────────────────────────
ax = axes[1]
b3 = ax.bar(x - WIDTH/2, ppl_med,  WIDTH, label='Medium',
            color=MEDIUM, alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)
b4 = ax.bar(x + WIDTH/2, ppl_high, WIDTH, label='High',
            color=HIGH,   alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)
ax.set_ylim(0, 14.5)
ax.set_ylabel('Perplejidad', fontsize=9.5)
ax.set_title('Perplejidad', fontsize=11, pad=8)
ax.legend(fontsize=9, framealpha=0.0)
style_ax(ax)
add_labels(ax, b3, '{:.2f}', 0.15)
add_labels(ax, b4, '{:.2f}', 0.15)

# ── Panel 3: Brecha de generalización ────────────────────────────────────────
ax = axes[2]
b5 = ax.bar(x - WIDTH/2, gap_med,  WIDTH, label='Medium',
            color=MEDIUM, alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)
b6 = ax.bar(x + WIDTH/2, gap_high, WIDTH, label='High',
            color=HIGH,   alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)
ax.axhline(0, color='#444444', linewidth=0.7, linestyle='-', zorder=2)
ax.set_ylim(-0.16, 0.50)
ax.set_ylabel('Gap = test loss $-$ train loss', fontsize=9.5)
ax.set_title('Brecha de generalización', fontsize=11, pad=8)
ax.legend(fontsize=9, framealpha=0.0)
style_ax(ax)
add_labels(ax, b5, '{:+.4f}', 0.010, offset_neg=-0.018)
add_labels(ax, b6, '{:+.4f}', 0.010)

plt.savefig('fig_results.pdf', bbox_inches='tight', dpi=200)
plt.savefig('fig_results.png', bbox_inches='tight', dpi=180)
print("fig_results OK")
