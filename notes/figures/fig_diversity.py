"""
Figura 1: Comparación de métricas de diversidad (Medium vs High).
Panel izquierdo  — entropías H1, H2, H3.
Panel derecho    — Distinct-4 y Gzip ratio.
Estilo: serif/CM, paleta teal–magenta, spines limpios.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family']      = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.linewidth']   = 0.8

MEDIUM  = '#1FB6B6'   # teal
HIGH    = '#C9189E'   # magenta
WIDTH   = 0.32
ALPHA   = 0.88

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
fig.subplots_adjust(wspace=0.38)

# ── Panel izquierdo: entropías ────────────────────────────────────────────────
labels_ent = [r'$H_1$', r'$H_2$', r'$H_3$']
med_ent    = [3.0441, 5.4235, 7.2420]
high_ent   = [3.1991, 5.6971, 7.5956]
x = np.arange(len(labels_ent))

b1 = ax1.bar(x - WIDTH/2, med_ent,  WIDTH, label='Medium',
             color=MEDIUM, alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)
b2 = ax1.bar(x + WIDTH/2, high_ent, WIDTH, label='High',
             color=HIGH,   alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(labels_ent, fontsize=12)
ax1.set_ylabel('Entropía de $n$-gramas (bits)', fontsize=10)
ax1.set_title('Entropía por orden de $n$-grama', fontsize=11, pad=8)
ax1.set_ylim(0, 9.0)
ax1.tick_params(labelsize=9, length=3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
ax1.set_axisbelow(True)
leg1 = ax1.legend(fontsize=9, framealpha=0.0, borderpad=0.4)

for bar in b1:
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.12,
             f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=7.5, color='#333333')
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.12,
             f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=7.5, color='#333333')

# ── Panel derecho: Distinct-4 y Gzip ratio ────────────────────────────────────
labels_rat = ['Distinct-4', 'Gzip ratio']
med_rat    = [0.1649, 0.3778]
high_rat   = [0.2303, 0.4190]
x2 = np.arange(len(labels_rat))

b3 = ax2.bar(x2 - WIDTH/2, med_rat,  WIDTH, label='Medium',
             color=MEDIUM, alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)
b4 = ax2.bar(x2 + WIDTH/2, high_rat, WIDTH, label='High',
             color=HIGH,   alpha=ALPHA, edgecolor='white', linewidth=0.6, zorder=3)

ax2.set_xticks(x2)
ax2.set_xticklabels(labels_rat, fontsize=10.5)
ax2.set_ylabel('Proporción', fontsize=10)
ax2.set_title('Diversidad léxica y compresibilidad', fontsize=11, pad=8)
ax2.set_ylim(0, 0.58)
ax2.tick_params(labelsize=9, length=3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
leg2 = ax2.legend(fontsize=9, framealpha=0.0, borderpad=0.4)

for bar in b3:
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.007,
             f'{bar.get_height():.4f}',
             ha='center', va='bottom', fontsize=7.5, color='#333333')
for bar in b4:
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.007,
             f'{bar.get_height():.4f}',
             ha='center', va='bottom', fontsize=7.5, color='#333333')

plt.savefig('fig_diversity.pdf', bbox_inches='tight', dpi=200)
plt.savefig('fig_diversity.png', bbox_inches='tight', dpi=180)
print("fig_diversity OK")
