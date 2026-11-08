"""
83_sst_ruido_v_sem_borda.py — ruído de Stokes V nas asas extremas, filtrado (como o 82) e SEM a borda do campo.

Script 82 (mapas cel82_mapas_S.png): o padrão simétrico S = (V₀ + V₁₄)/2 se concentra em faixas
estreitas nas bordas esquerda e direita do campo circular do CRISP2, com até ~4×10⁻² I (08:01) e
~1×10⁻² I (10:48), além da região que o pipeline já mascara. Faixas estreitas e de borda abrupta
explicam por que ε₊ dependia da largura do filtro (C2) e por que σ₋/σ₊ ficava em 0,95.
Os resultados do 82 ficam registrados como estão; este é um teste novo com a borda excluída.

Mudança em relação ao 82 (o resto é idêntico, inclusive C1, C2 e os limiares):
  - distância de cada pixel à borda do campo válido (V finito e ≠ 0), por transformada de distância;
  - exclui pixels a menos de d px da borda. d = 150 px (≈ 7,5″) é o valor da decisão;
    d = 300 px é só sensibilidade (descritivo).
  - mapa de S suave só no interior, com escala de cor própria (cel83_mapas_S_interior.png).

Critério (definido antes de rodar), com d = 150 px, por posição:
  C1: |σ₊/σ_ind − 1| < 0,05 no resíduo com w = 16 px
  C2: |ε₊(w = 16)/ε₊(w = 32) − 1| < 0,03
  Arquivo VÁLIDO se C1 e C2 valem em ≥ 3 das 4 posições → ε₊(w = 16) é a medida de ruído de V.

Uso:  python Scripts/83_sst_ruido_v_sem_borda.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
Saída: Outputs/Reports/83_sst_ruido_v_sem_borda.json,
       Outputs/Figures/cel83_ruido_filtrado.png e cel83_mapas_S_interior.png

Saídas por arquivo (2026-10-26, _saidas_sst.py): relatório <prefixo>_<AAAAMMDDThhmmss>.json e figuras
<nome>_<AAAAMMDDThhmmss>.png, mais <prefixo>_resumo.json com todos os arquivos já analisados.
Critério por arquivo em proporção: "≥ 3 das 4 posições" ≡ ≥ 75% das posições/varreduras (idêntico nos
mosaicos; fixado antes de reexecutar nas séries do CRISP, que têm 14 e 10 varreduras).
"""
import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saidas_sst as SA

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'
K_MAD = 1.4826
L_AZUL, L_VERM = 0, 14
LARGURAS = (16, 32)
D_BORDA = 150          # px, valor da decisão
D_SENS = 300           # px, só sensibilidade


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def sigmas(v0, v14):
    s0, s14 = sigma_rob(v0), sigma_rob(v14)
    smais = sigma_rob(v0 + v14) / np.sqrt(2)
    smenos = sigma_rob(v0 - v14) / np.sqrt(2)
    sind = float(np.sqrt((s0 ** 2 + s14 ** 2) / 2))
    return {'sigma_0': s0, 'sigma_14': s14, 'sigma_mais': smais, 'sigma_menos': smenos,
            'sigma_ind': sind, 'razao_mais_ind': smais / sind, 'razao_menos_mais': smenos / smais}


def suave(V, m, w):
    num = gaussian_filter(np.where(m, V, 0.0), w, mode='constant')
    den = gaussian_filter(m.astype(np.float64), w, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den, den


def analisar(caminho, d_borda):
    out = {'arquivo': Path(caminho).name, 'posicoes': []}
    mapas = []
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data                                      # (t, s, λ, y, x)
        nt, ns, nl = d.shape[:3]
        r0 = h[0].header.get('ATMOS_R0', None)
        out['ATMOS_R0_header'] = None if r0 is None else float(np.nanmean(r0))
        for t in range(nt):
            amostra = np.array([np.asarray(d[t, 0, l, ::8, ::8], dtype=np.float32) for l in range(nl)])
            okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
            lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(nl)]))
            le = lambda s, l: np.asarray(d[t, s, l], dtype=np.float64)
            I0, I14, Ic = le(0, L_AZUL), le(0, L_VERM), le(0, lnuc)
            V0, V14 = le(3, L_AZUL), le(3, L_VERM)
            ok = (np.isfinite(I0) & (I0 != 0) & np.isfinite(I14) & np.isfinite(Ic)
                  & np.isfinite(V0) & (V0 != 0) & np.isfinite(V14) & (V14 != 0))
            asa = I0 / np.median(I0[ok])
            nuc = Ic / np.median(Ic[ok])
            dist = distance_transform_edt(binary_fill_holes(ok))   # buracos internos não contam como borda
            interior = dist >= d_borda
            calmo = ok & interior & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)
            mI = float(np.mean((I0[calmo] + I14[calmo]) / 2))
            sem = sigmas(V0[calmo], V14[calmo])
            pos = {'t': t, 'd_borda': d_borda, 'frac_campo_interior': float(interior.sum() / max(ok.sum(), 1)),
                   'n_calmo': int(calmo.sum()), 'I_asas_medio': mI, 'sem_filtro': sem,
                   'eps_mais_sem_filtro': sem['sigma_mais'] / mI, 'eps_menos_sem_filtro': sem['sigma_menos'] / mI}
            for w in LARGURAS:
                b0, den = suave(V0, calmo, w)
                b14, _ = suave(V14, calmo, w)
                usa = calmo & (den > 0.5)
                s = sigmas((V0 - b0)[usa], (V14 - b14)[usa])
                s['n_usados'] = int(usa.sum())
                s['eps_mais'] = s['sigma_mais'] / mI
                s['eps_menos'] = s['sigma_menos'] / mI
                s['amplitude_S_suave'] = sigma_rob(((b0 + b14) / 2)[usa]) / mI
                s['amplitude_D_suave'] = sigma_rob(((b14 - b0) / 2)[usa]) / mI
                pos[f'w{w}'] = s
                if w == LARGURAS[0]:
                    Smap = np.where(usa, (b0 + b14) / 2 / mI, np.nan)
                    mapas.append(Smap[::8, ::8])
            f16, f32 = pos['w16'], pos['w32']
            pos['C1'] = bool(abs(f16['razao_mais_ind'] - 1) < 0.05)
            pos['razao_eps_16_32'] = f16['eps_mais'] / f32['eps_mais']
            pos['C2'] = bool(abs(pos['razao_eps_16_32'] - 1) < 0.03)
            out['posicoes'].append(pos)
            print(f"  t={t}: d = {d_borda} px (interior {pos['frac_campo_interior']:.0%} do campo)  sem filtro  σ₊/σ_ind = {sem['razao_mais_ind']:.3f}  ε₊ = {pos['eps_mais_sem_filtro']:.2e}  "
                  f"ε₋ = {pos['eps_menos_sem_filtro']:.2e}")
            for w in LARGURAS:
                s = pos[f'w{w}']
                print(f"        w = {w:2d} px   σ₊/σ_ind = {s['razao_mais_ind']:.3f}  σ₋/σ₊ = {s['razao_menos_mais']:.3f}  "
                      f"ε₊ = {s['eps_mais']:.2e}  ε₋ = {s['eps_menos']:.2e}  padrão suave S: {s['amplitude_S_suave']:.2e}")
            print(f"        ε₊(16)/ε₊(32) = {pos['razao_eps_16_32']:.3f}   C1 {'passa' if pos['C1'] else 'FALHA'}   "
                  f"C2 {'passa' if pos['C2'] else 'FALHA'}")
    n_ok = sum(p['C1'] and p['C2'] for p in out['posicoes'])
    out['n_posicoes_ok'] = n_ok
    out['decisao'] = 'VÁLIDO' if SA.passa_fracao(n_ok, len(out['posicoes'])) else 'REPROVADO'
    boas = [p['w16']['eps_mais'] for p in out['posicoes'] if p['C1'] and p['C2']]
    out['eps_V'] = ({'mediano': float(np.median(boas)), 'faixa': [float(min(boas)), float(max(boas))]}
                    if out['decisao'] == 'VÁLIDO' else None)
    return out, mapas


def figuras(todos, todos_mapas, caminhos):
    cores = ['#0072B2', '#D55E00']
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, r in enumerate(todos):
        ts = np.array([p['t'] for p in r['posicoes']]) + (k - 0.5) * 0.2
        lab = r['arquivo'][8:27]
        ax.plot(ts, [p['eps_mais_sem_filtro'] * 1e3 for p in r['posicoes']], 's', color=cores[k % 2], mfc='white', ms=7,
                label=f'{lab}  ε₊ sem filtro')
        ax.plot(ts, [p['w16']['eps_mais'] * 1e3 for p in r['posicoes']], 'o', color=cores[k % 2], ms=8,
                label=f'{lab}  ε₊ filtrado (16 px)')
        ax.plot(ts, [p['w32']['eps_mais'] * 1e3 for p in r['posicoes']], 'x', color=cores[k % 2], ms=8, mew=2,
                label=f'{lab}  ε₊ filtrado (32 px)')
        ax.plot(ts, [p['eps_menos_sem_filtro'] * 1e3 for p in r['posicoes']], 'v', color=cores[k % 2], mfc='white', ms=7,
                label=f'{lab}  ε₋ sem filtro')
    ax.axhline(3.2, color='0.4', ls=':', lw=1, label='CRISP 2024 (artigo B)')
    nmax = max(len(r['posicoes']) for r in todos)
    ax.set_xticks(range(nmax)); ax.set_xlabel('posição do mosaico ou varredura (t)'); ax.set_ylabel('ε_V  (×10⁻³)')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminhos[0], dpi=120)
    plt.close(fig)

    nlin = len(todos)
    fig, axs = plt.subplots(nlin, 4, figsize=(12, 3.2 * nlin), squeeze=False)
    vm = np.nanmax([np.nanpercentile(np.abs(m), 99) for ms in todos_mapas for m in ms])
    for k, (r, ms) in enumerate(zip(todos, todos_mapas)):
        idx = list(range(len(ms))) if len(ms) <= 4 else [int(round(x)) for x in np.linspace(0, len(ms) - 1, 4)]
        for j, t in enumerate(idx):                     # séries temporais: 4 varreduras igualmente espaçadas
            im = axs[k, j].imshow(ms[t] * 1e3, origin='lower', cmap='RdBu_r', vmin=-vm * 1e3, vmax=vm * 1e3)
            axs[k, j].set_title(f"{r['arquivo'][8:27]}  t={t}", fontsize=8)
            axs[k, j].set_xticks([]); axs[k, j].set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label='S suave / ⟨I⟩  (×10⁻³)')
    fig.savefig(caminhos[1], dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    todos, todos_mapas, sens = [], [], []
    for c in a.arquivos:
        print('=' * 78)
        print(Path(c).name, f'   [decisão, d = {D_BORDA} px]')
        r, mapas = analisar(c, D_BORDA)
        todos.append(r)
        todos_mapas.append(mapas)
        print(f"  posições com C1 e C2: {r['n_posicoes_ok']} de {len(r['posicoes'])}  →  {r['decisao']}")
        if r['eps_V']:
            e = r['eps_V']
            print(f"  ε_V (ruído, filtrado 16 px, sem borda): {e['mediano']:.2e}  (faixa {e['faixa'][0]:.2e}–{e['faixa'][1]:.2e})")
        print(f'  --- sensibilidade (descritivo), d = {D_SENS} px ---')
        rs, _ = analisar(c, D_SENS)
        sens.append(rs)
        for p150, p300 in zip(r['posicoes'], rs['posicoes']):
            print(f"  t={p150['t']}: ε₊(16) d=150: {p150['w16']['eps_mais']:.2e}   d=300: {p300['w16']['eps_mais']:.2e}   "
                  f"razão {p300['w16']['eps_mais'] / p150['w16']['eps_mais']:.3f}")
        rel = SA.gravar('83_sst_ruido_v_sem_borda', r, {'sensibilidade_d300': [rs]})   # antes das figuras
        cfs = [SA.figura(n, r['etiqueta']) for n in ('cel83_ruido_filtrado', 'cel83_mapas_S_interior')]
        figuras([r], [mapas], cfs)
        print(f"  Relatório: {rel}\n  Figuras: {cfs[0]}\n           {cfs[1]}")
    SA.resumo('83_sst_ruido_v_sem_borda')


if __name__ == '__main__':
    main()
