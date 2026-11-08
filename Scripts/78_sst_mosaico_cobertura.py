"""
78_sst_mosaico_cobertura.py — as 4 "varreduras" de um cubo CRISP2 são o mesmo campo ou posições de mosaico?

Para cada passo de tempo t (eixo 0 do cubo (t, s, λ, y, x)):
  - fração do campo com Stokes I válido (contínuo, λ índice 0) e com Stokes V válido;
para cada par (t1, t2):
  - fração de pixels válidos em comum (sobreposição);
  - correlação de I no contínuo entre t1 e t2 na sobreposição.

Critério (definido antes de rodar):
  - MESMO CAMPO: sobreposição ≥ 0,8 e correlação ≥ 0,7 em todos os pares consecutivos
    → ruído pode ser medido pela diferença entre varreduras (scripts 64/65).
  - MOSAICO: sobreposição < 0,2 em pares consecutivos
    → cada pixel tem uma observação; ruído terá de vir de V no contínuo de regiões calmas.
  - Outro caso: INDEFINIDO (olhar a figura).

Lê com memmap; usa só o contínuo (λ índice 0) de I e V, reamostrado a cada `passo` pixels.
Saída: Outputs/Reports/78_sst_mosaico_cobertura.json e Outputs/Figures/cel78_<arquivo>.png

Uso:  python Scripts/78_sst_mosaico_cobertura.py Data/RAW_SST/<arquivo>.fits [--passo 4]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'


def analisar(caminho, passo):
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data                                   # (t, s, λ, y, x)
        nt = d.shape[0]
        I = np.array([np.asarray(d[t, 0, 0, ::passo, ::passo], dtype=np.float32) for t in range(nt)])
        V = np.array([np.asarray(d[t, 3, 0, ::passo, ::passo], dtype=np.float32) for t in range(nt)])
    okI = np.isfinite(I) & (I != 0)
    okV = np.isfinite(V) & (V != 0)
    res = {'arquivo': Path(caminho).name, 'passo': passo, 'shape_amostrado': list(I.shape),
           'por_varredura': [], 'pares': []}
    for t in range(nt):
        res['por_varredura'].append({'t': t, 'frac_I_valido': float(okI[t].mean()),
                                     'frac_V_valido': float(okV[t].mean())})
    for a in range(nt):
        for b in range(a + 1, nt):
            comum = okI[a] & okI[b]
            uniao = okI[a] | okI[b]
            sob = float(comum.sum() / max(uniao.sum(), 1))
            if comum.sum() > 100:
                r = float(np.corrcoef(I[a][comum], I[b][comum])[0, 1])
            else:
                r = None
            res['pares'].append({'t1': a, 't2': b, 'sobreposicao': sob, 'correlacao_I': r})
    cons = [p for p in res['pares'] if p['t2'] == p['t1'] + 1]
    if all(p['sobreposicao'] >= 0.8 and (p['correlacao_I'] or 0) >= 0.7 for p in cons):
        dec = 'MESMO CAMPO'
    elif all(p['sobreposicao'] < 0.2 for p in cons):
        dec = 'MOSAICO'
    else:
        dec = 'INDEFINIDO'
    res['decisao'] = dec

    fig, axs = plt.subplots(2, nt, figsize=(3 * nt, 6))
    for t in range(nt):
        v = I[t][okI[t]]
        lo, hi = (np.percentile(v, [1, 99]) if v.size else (0, 1))
        axs[0, t].imshow(np.where(okI[t], I[t], np.nan), origin='lower', cmap='gray', vmin=lo, vmax=hi)
        axs[0, t].set_title(f't={t}  I contínuo')
        axs[1, t].imshow(okV[t], origin='lower', cmap='viridis', vmin=0, vmax=1)
        axs[1, t].set_title(f't={t}  V válido ({okV[t].mean():.0%})')
        for ax in axs[:, t]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{Path(caminho).name[:40]}…  →  {dec}')
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    nome = 'cel78_' + Path(caminho).stem[:30].replace(':', '_') + '.png'
    fig.savefig(FIG / nome, dpi=110)
    plt.close(fig)
    res['figura'] = str(FIG / nome)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    ap.add_argument('--passo', type=int, default=4)
    a = ap.parse_args()
    todos = []
    for c in a.arquivos:
        r = analisar(c, a.passo)
        todos.append(r)
        print('=' * 78)
        print(r['arquivo'])
        for p in r['por_varredura']:
            print(f"  t={p['t']}: I válido {p['frac_I_valido']:.1%}   V válido {p['frac_V_valido']:.1%}")
        for p in r['pares']:
            rr = 'n/d' if p['correlacao_I'] is None else f"{p['correlacao_I']:+.3f}"
            print(f"  t{p['t1']}–t{p['t2']}: sobreposição {p['sobreposicao']:.2f}   correlação I {rr}")
        print(f"  → {r['decisao']}   (figura: {r['figura']})")
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '78_sst_mosaico_cobertura.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos}, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório: {REP / '78_sst_mosaico_cobertura.json'}")


if __name__ == '__main__':
    main()
