"""
111_sst_escala_crisp.py — escala de placa e apontamento das séries CRISP 2024-10-08 a partir do cabeçalho FITS.

Para os eixos da Fig. A6 (109) em arcsec. Lê o WCS de cada arquivo:
  - se HPLN/HPLT forem -TAB (como no CRISP2, script 92): mesma conta do 92 (ψ = CRVAL + CDELT·PC·(p − CRPIX),
    índice na tabela, interpolação bilinear, média em λ) e ajuste afim pixel → (HPLN, HPLT) por varredura;
    SEM a troca para "cantos da tabela" do 92 (aquela verificação supunha 0,0503″/px; aqui a escala é a incógnita);
  - senão, CDELT/PC (ou CD) do cabeçalho, convertidos para arcsec.
Também imprime CTYPE/CUNIT/CDELT para conferência. Só leitura; sem critério (é um parâmetro de figura).
Diagnóstico: dispersão da escala entre varreduras (< 1% esperado) e comparação com o CRISP2 (0,0487″/px, 92).

Uso:   python Scripts/111_sst_escala_crisp.py Data/RAW_SST/<arquivo CRISP>.fits [mais arquivos]
Saída: Outputs/Reports/111_sst_escala_crisp.json (lido pelo 109 para a Fig. A6)
"""
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Scripts'))

import numpy as np
from astropy.io import fits

import _saidas_sst as SA

UNID = {'arcsec': 1.0, 'deg': 3600.0, 'arcmin': 60.0, 'rad': 206264.806}


def eixos(hd):
    e = {}
    for i in range(1, hd['NAXIS'] + 1):
        c = str(hd.get(f'CTYPE{i}', '')).upper()
        if c.startswith('HPLN'):
            e['x'] = i
        if c.startswith('HPLT'):
            e['y'] = i
    return e


def afins_tab(h, ix, iy):
    hd = h[0].header
    nt, _, nl, ny, nx = h[0].data.shape
    ext, col = hd[f'PS{ix}_0'], hd[f'PS{ix}_1']
    arr = np.asarray(h[ext].data[col][0], dtype=np.float64)
    nomes = [n.strip().upper() for n in str(col).split('+')]
    mx, my = nomes.index('HPLN'), nomes.index('HPLT')
    K2, K1 = arr.shape[-3], arr.shape[-2]

    def indice(i, K):
        nome_idx = hd.get(f'PS{i}_2')
        if nome_idx:
            try:
                return np.asarray(h[ext].data[nome_idx][0], dtype=np.float64).ravel()
            except Exception:
                pass
        return np.arange(1, K + 1, dtype=np.float64)

    ups_x, ups_y = indice(ix, K1), indice(iy, K2)

    def psi(i, px, py):
        pc = lambda a, b: hd.get(f'PC{a}_{b}', 1.0 if a == b else 0.0)
        p = {ix: px + 1.0, iy: py + 1.0}
        s = sum(pc(i, j) * (p[j] - hd.get(f'CRPIX{j}', 0.0)) for j in (ix, iy))
        return hd.get(f'CRVAL{i}', 0.0) + hd.get(f'CDELT{i}', 1.0) * s

    gy, gx = np.meshgrid(np.linspace(0, ny - 1, 6), np.linspace(0, nx - 1, 6), indexing='ij')
    gx, gy = gx.ravel(), gy.ravel()
    out = []
    for t in range(nt):
        C = arr[t].mean(axis=0) if arr.ndim == 5 else arr.mean(axis=0) if arr.ndim == 4 else arr
        fx = (psi(ix, gx, gy) - ups_x[0]) / (ups_x[-1] - ups_x[0]) * (K1 - 1)
        fy = (psi(iy, gx, gy) - ups_y[0]) / (ups_y[-1] - ups_y[0]) * (K2 - 1)
        x0 = np.clip(np.floor(fx).astype(int), 0, K1 - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, K2 - 2)
        wx, wy = fx - x0, fy - y0

        def bil(m):
            return (C[y0, x0, m] * (1 - wx) * (1 - wy) + C[y0, x0 + 1, m] * wx * (1 - wy)
                    + C[y0 + 1, x0, m] * (1 - wx) * wy + C[y0 + 1, x0 + 1, m] * wx * wy)
        X = np.vstack([gx, gy, np.ones_like(gx)]).T
        cx, *_ = np.linalg.lstsq(X, bil(mx), rcond=None)
        cy, *_ = np.linalg.lstsq(X, bil(my), rcond=None)
        A = np.vstack([cx, cy])
        out.append({'t': t, 'modo': 'WCS-TAB',
                    'escala_arcsec_px': float(np.sqrt(abs(np.linalg.det(A[:, :2])))),
                    'angulo_eixo_x_graus': float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
                    'centro': [float(v) for v in A @ np.array([(nx - 1) / 2, (ny - 1) / 2, 1.0])]})
    return out


def linear(hd, ix, iy, ny, nx):
    u = UNID.get(str(hd.get(f'CUNIT{ix}', 'arcsec')).strip().lower(), 1.0)
    if f'CD{ix}_{ix}' in hd:
        M = np.array([[hd.get(f'CD{ix}_{ix}', 0.0), hd.get(f'CD{ix}_{iy}', 0.0)],
                      [hd.get(f'CD{iy}_{ix}', 0.0), hd.get(f'CD{iy}_{iy}', 0.0)]])
    else:
        pc = lambda a, b: hd.get(f'PC{a}_{b}', 1.0 if a == b else 0.0)
        M = np.array([[hd.get(f'CDELT{ix}', 1.0) * pc(ix, ix), hd.get(f'CDELT{ix}', 1.0) * pc(ix, iy)],
                      [hd.get(f'CDELT{iy}', 1.0) * pc(iy, ix), hd.get(f'CDELT{iy}', 1.0) * pc(iy, iy)]])
    M = M * u
    cen = np.array([hd.get(f'CRVAL{ix}', 0.0), hd.get(f'CRVAL{iy}', 0.0)]) * u + M @ (
        np.array([(nx - 1) / 2 + 1 - hd.get(f'CRPIX{ix}', 0.0), (ny - 1) / 2 + 1 - hd.get(f'CRPIX{iy}', 0.0)]))
    return [{'t': 0, 'modo': 'linear (CDELT/PC ou CD)', 'escala_arcsec_px': float(np.sqrt(abs(np.linalg.det(M)))),
             'angulo_eixo_x_graus': float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 'centro': [float(c) for c in cen]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    out = {'arquivos': []}
    for c in a.arquivos:
        nome = Path(c).name
        with fits.open(c, memmap=True) as h:
            hd = h[0].header
            shp = h[0].data.shape
            e = eixos(hd)
            print(f'\n{nome}\n  forma {shp};  ' + '; '.join(
                f"eixo {i}: {hd.get(f'CTYPE{i}', '')} [{hd.get(f'CUNIT{i}', '')}] CDELT {hd.get(f'CDELT{i}', '—')}"
                for i in range(1, hd['NAXIS'] + 1)))
            if 'x' not in e or 'y' not in e:
                print('  sem eixos HPLN/HPLT no cabeçalho primário — escala não determinada')
                continue
            ix, iy = e['x'], e['y']
            tab = str(hd.get(f'CTYPE{ix}', '')).upper().endswith('-TAB')
            g = afins_tab(h, ix, iy) if tab else linear(hd, ix, iy, shp[-2], shp[-1])
        esc = np.array([q['escala_arcsec_px'] for q in g])
        r = {'arquivo': nome, 'etiqueta': SA.etiqueta(nome), 'geometria': g,
             'escala_mediana_arcsec_px': float(np.median(esc)),
             'dispersao_relativa': float(np.std(esc) / np.median(esc)) if len(esc) > 1 else 0.0}
        out['arquivos'].append(r)
        print(f"  modo {g[0]['modo']}: escala mediana {r['escala_mediana_arcsec_px']:.5f}″/px "
              f"(dispersão entre varreduras {100 * r['dispersao_relativa']:.2f}%), eixo x a "
              f"{np.median([q['angulo_eixo_x_graus'] for q in g]):+.2f}°, centro da varredura 0 "
              f"({g[0]['centro'][0]:.1f}, {g[0]['centro'][1]:.1f})″;  razão para o CRISP2 (0,0487″/px): "
              f"{r['escala_mediana_arcsec_px'] / 0.0487:.3f}")
    SA.REP.mkdir(parents=True, exist_ok=True)
    p = SA.REP / '111_sst_escala_crisp.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nRelatório: {p}')


if __name__ == '__main__':
    main()
