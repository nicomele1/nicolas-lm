#!/usr/bin/env python3
"""
Genera fig_diversity.pdf y fig_results.pdf usando solo stdlib de Python.
PDF directo con operadores gráficos: sin numpy, sin matplotlib.
"""

import io

# ── Paleta ────────────────────────────────────────────────────────────────────
def _h(s):
    return (int(s[1:3],16)/255, int(s[3:5],16)/255, int(s[5:7],16)/255)

TEAL  = _h('#1FB6B6')   # Medium
MAGE  = _h('#C9189E')   # High
AXIS  = (0.25, 0.25, 0.25)
GRID  = (0.87, 0.87, 0.87)
TEXT  = (0.10, 0.10, 0.10)
WHITE = (1.0,  1.0,  1.0)

def rgb(t): return f'{t[0]:.4f} {t[1]:.4f} {t[2]:.4f}'

# ── Constructor de streams PDF ────────────────────────────────────────────────
class PDF:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.S = []          # content stream lines

    # estado gráfico
    def save(self):    self.S.append('q')
    def restore(self): self.S.append('Q')
    def lw(self, w):   self.S.append(f'{w:.2f} w')
    def fill(self, c): self.S.append(f'{rgb(c)} rg')
    def stroke(self, c): self.S.append(f'{rgb(c)} RG')

    # primitivas
    def rect(self, x, y, w, h, op='f'):
        self.S.append(f'{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}')

    def line(self, x1, y1, x2, y2):
        self.S.append(f'{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S')

    # texto (fuente F1=Helvetica, F2=Helvetica-Oblique)
    # align: 'l'=left, 'c'=center, 'r'=right
    def text(self, x, y, s, sz=9, font='F1', col=TEXT, align='l'):
        ew = len(s) * sz * 0.50     # ancho estimado
        if align == 'c': x -= ew / 2
        if align == 'r': x -= ew
        self.S += ['BT', f'{rgb(col)} rg', f'/{font} {sz:.1f} Tf',
                   f'{x:.2f} {y:.2f} Td', f'({s}) Tj', 'ET']

    # texto rotado 90° (para etiquetas del eje y)
    def text_rot(self, x, y, s, sz=9, col=TEXT):
        ew = len(s) * sz * 0.50
        self.S += ['BT', f'{rgb(col)} rg', f'/F1 {sz:.1f} Tf',
                   f'0 1 -1 0 {x:.2f} {y - ew/2:.2f} Tm',
                   f'({s}) Tj', 'ET']

    # ── ensamblar PDF ─────────────────────────────────────────────────────────
    def build(self):
        content = '\n'.join(self.S).encode('latin-1')
        objs = {
            1: b'<< /Type /Catalog /Pages 2 0 R >>',
            2: b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
            3: (f'<< /Type /Page /Parent 2 0 R '
                f'/MediaBox [0 0 {self.W:.2f} {self.H:.2f}] '
                f'/Contents 4 0 R '
                f'/Resources << /ProcSet [/PDF /Text] '
                f'/Font << /F1 5 0 R /F2 6 0 R >> >> >>').encode(),
            4: (f'<< /Length {len(content)} >>\nstream\n').encode()
                + content + b'\nendstream',
            5: b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica'
               b' /Encoding /WinAnsiEncoding >>',
            6: b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique'
               b' /Encoding /WinAnsiEncoding >>',
        }
        out = io.BytesIO()
        out.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
        offs = {}
        for n in sorted(objs):
            offs[n] = out.tell()
            out.write(f'{n} 0 obj\n'.encode())
            out.write(objs[n])
            out.write(b'\nendobj\n')
        xp = out.tell()
        N = max(objs) + 1
        out.write(f'xref\n0 {N}\n'.encode())
        out.write(b'0000000000 65535 f \n')
        for i in range(1, N):
            out.write((f'{offs[i]:010d} 00000 n \n' if i in offs
                       else b'0000000000 65535 f \n').encode()
                      if isinstance(f'{offs.get(i,""):010d} 00000 n \n', str)
                      else f'{offs[i]:010d} 00000 n \n'.encode())
        out.write(f'trailer\n<< /Size {N} /Root 1 0 R >>\n'
                  f'startxref\n{xp}\n%%EOF\n'.encode())
        return out.getvalue()


# ── Función de panel de barras agrupadas ──────────────────────────────────────
def bar_panel(p, x0, x1, y0, y1,
              groups, vals_med, vals_high,
              y_min, y_max, y_ticks, tick_fmt,
              title, y_label=None,
              bw=17, bg=4, gg=18):
    """
    Dibuja un panel de barras agrupadas (Medium=TEAL, High=MAGE).
    x0,x1,y0,y1 definen el área de gráfico (pt).
    groups: lista de etiquetas de grupos.
    vals_med / vals_high: lista de valores por grupo.
    """
    pw, ph = x1 - x0, y1 - y0
    yr = y_max - y_min
    ys = ph / yr                    # pt por unidad
    y0_data = y0 - y_min * ys      # posición y del cero de datos

    n = len(groups)
    gw = 2*bw + bg                  # ancho de un grupo
    total_w = n*gw + (n-1)*gg
    xs = x0 + (pw - total_w) / 2   # x de inicio del primer grupo

    # Fondo blanco del área
    p.save()
    p.fill(WHITE); p.rect(x0, y0, pw, ph, 'f')
    p.restore()

    # Grid horizontal
    p.save()
    p.lw(0.35); p.stroke(GRID)
    for t in y_ticks:
        if y_min <= t <= y_max:
            yp = y0_data + t * ys
            p.line(x0, yp, x1, yp)
    p.restore()

    # Ejes
    p.save()
    p.lw(0.7); p.stroke(AXIS)
    p.line(x0, y0, x0, y1)              # eje y
    p.line(x0, y0_data, x1, y0_data)   # eje x (en el cero de datos)
    p.restore()

    # Marcas y etiquetas del eje y
    p.save(); p.lw(0.4); p.stroke(AXIS)
    for t in y_ticks:
        if y_min <= t <= y_max:
            yp = y0_data + t * ys
            p.line(x0 - 3.5, yp, x0, yp)
            p.text(x0 - 5, yp - 3.5, tick_fmt(t), sz=7.5, align='r')
    p.restore()

    # Barras y etiquetas de valor
    for i, (g, vm, vh) in enumerate(zip(groups, vals_med, vals_high)):
        gx = xs + i * (gw + gg)

        for val, col, bx in [(vm, TEAL, gx), (vh, MAGE, gx + bw + bg)]:
            bh = val * ys
            by = y0_data if val >= 0 else y0_data + bh
            bh_abs = abs(bh)

            # barra con fondo blanco (para separación visual leve)
            p.save()
            p.fill(col)
            p.rect(bx + 0.5, by, bw - 1, bh_abs, 'f')
            p.restore()

            # etiqueta de valor
            lbl = (f'{val:+.4f}' if abs(val) < 0.1 and val != 0
                   else f'{val:.4f}' if abs(val) < 1
                   else f'{val:.3f}' if abs(val) < 10
                   else f'{val:.2f}')
            lx = bx + bw / 2
            ly = (y0_data + bh + 2.5 if val >= 0
                  else y0_data + bh - 8)
            p.text(lx, ly, lbl, sz=6.2, align='c', col=TEXT)

        # etiqueta de grupo (bajo el eje x)
        gcx = gx + gw / 2
        p.text(gcx, y0 - 16, g, sz=8.0, align='c')

    # Título del panel
    pcx = (x0 + x1) / 2
    p.text(pcx, y1 + 6, title, sz=9.5, align='c', col=AXIS)

    # Etiqueta eje y (rotada)
    if y_label:
        p.text_rot(x0 - 32, (y0 + y1) / 2, y_label, sz=8.5, col=AXIS)


def legend(p, x, y, labels, colors, sz=8.5):
    xc = x
    for lbl, col in zip(labels, colors):
        p.save(); p.fill(col)
        p.rect(xc, y, sz, sz, 'f')
        p.restore()
        p.text(xc + sz + 2.5, y + 0.5, lbl, sz=sz, align='l')
        xc += sz + 2.5 + len(lbl) * sz * 0.50 + 9


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 1: Métricas de diversidad
# ═══════════════════════════════════════════════════════════════════════════════
def make_diversity():
    W, H = 495.0, 168.0
    p = PDF(W, H)

    # área de plot: y0=45, y1=138 → altura=93pt
    Y0, Y1 = 45.0, 138.0

    # Panel izquierdo: entropías H1, H2, H3
    bar_panel(p, 48, 238, Y0, Y1,
              groups=['H_1', 'H_2', 'H_3'],
              vals_med=[3.1008, 5.5460, 7.4058],
              vals_high=[3.1308, 5.6084, 7.5880],
              y_min=0, y_max=8.5,
              y_ticks=[0, 2, 4, 6, 8],
              tick_fmt=lambda t: str(int(t)),
              title='Entropía por orden de n-grama',
              y_label='Entropia (bits)',
              bw=18, bg=4, gg=18)

    # Panel derecho: Distinct-4 y Gzip ratio
    bar_panel(p, 268, 487, Y0, Y1,
              groups=['Distinct-4', 'Gzip ratio'],
              vals_med=[0.0378, 0.3579],
              vals_high=[0.0592, 0.4066],
              y_min=0, y_max=0.55,
              y_ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
              tick_fmt=lambda t: f'{t:.1f}',
              title='Diversidad léxica y compresibilidad (proporcion)',
              y_label=None,
              bw=22, bg=5, gg=35)

    # Leyenda global centrada arriba
    legend(p, 170, H - 14, ['Medium', 'High'], [TEAL, MAGE], sz=9)

    return p.build()


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA 2: Resultados
# ═══════════════════════════════════════════════════════════════════════════════
def make_results():
    W, H = 495.0, 170.0
    p = PDF(W, H)

    Y0, Y1 = 45.0, 138.0
    ARCHS = ['Transformer', 'LLaMA-style']

    # Panel 1: Pérdida de test
    bar_panel(p, 45, 192, Y0, Y1,
              groups=ARCHS,
              vals_med=[1.8355, 1.4638],
              vals_high=[1.9597, 1.7010],
              y_min=0, y_max=2.5,
              y_ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5],
              tick_fmt=lambda t: f'{t:.1f}',
              title='Pérdida de test',
              y_label='Pérdida (entropía cruzada)',
              bw=17, bg=4, gg=20)

    # Panel 2: Perplejidad
    bar_panel(p, 210, 355, Y0, Y1,
              groups=ARCHS,
              vals_med=[6.27, 4.32],
              vals_high=[7.10, 5.48],
              y_min=0, y_max=9.0,
              y_ticks=[0, 2, 4, 6, 8],
              tick_fmt=lambda t: str(int(t)),
              title='Perplejidad (PPL)',
              y_label=None,
              bw=17, bg=4, gg=20)

    # Panel 3: Brecha de generalización
    bar_panel(p, 372, 490, Y0, Y1,
              groups=ARCHS,
              vals_med=[0.0676, 0.0993],
              vals_high=[0.0398, 0.0742],
              y_min=0, y_max=0.14,
              y_ticks=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
              tick_fmt=lambda t: f'{t:.2f}',
              title='Brecha de generalización (gap)',
              y_label=None,
              bw=14, bg=3, gg=16)

    legend(p, 170, H - 14, ['Medium', 'High'], [TEAL, MAGE], sz=9)

    return p.build()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    with open('fig_diversity.pdf', 'wb') as f:
        f.write(make_diversity())
    print('fig_diversity.pdf OK')

    with open('fig_results.pdf', 'wb') as f:
        f.write(make_results())
    print('fig_results.pdf OK')
