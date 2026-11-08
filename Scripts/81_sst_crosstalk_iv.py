"""
81_sst_crosstalk_iv.py — a componente de V simétrica nas asas (script 80) é crosstalk I→V?

Script 80: em todas as posições σ₊ > σ_ind e σ₋ < σ₊, ou seja, V₀ (≈ −658 mÅ) e V₁₄ (≈ +692 mÅ)
estão positivamente correlacionados (ρ ≈ 0,2–0,3). O sinal solar de campo fraco é antissimétrico;
uma componente com o mesmo sinal nas duas asas pode ser crosstalk I→V (V_λ += α I_λ).

Para cada posição do mosaico, no Sol calmo (máscara idêntica aos scripts 79 e 80):
  S = (V₀ + V₁₄)/2,  D = (V₁₄ − V₀)/2,  I_w = (I₀ + I₁₄)/2
  - ajuste S = α I_w + c (mínimos quadrados, pixels com |S − mediana| < 5σ): α e correlação r;
  - α também por quadrante do campo (2 × 2), para ver se o crosstalk varia no campo (descritivo);
  - correção global: V₀ ← V₀ − α I₀,  V₁₄ ← V₁₄ − α I₁₄; repete σ₀, σ₁₄, σ₊, σ₋ do script 80;
  - autocorrelação espacial de S e de D nos deslocamentos 1, 4 e 16 px (descritivo: crosstalk
    segue a granulação; ruído de pixel decai em ~1 px).

Critério (definido antes de rodar), por arquivo:
  CROSSTALK CONFIRMADO se |r| ≥ 0,3 em ≥ 3 das 4 posições
                       E, após a correção global, σ₊/σ_ind < 1,05 em ≥ 3 das 4 posições.
    → ε_V corrigido = σ₊/⟨I_w⟩ nas posições que passam vale como medida de ruído.
  Caso contrário: NÃO CONFIRMADO → fica registrado só σ₋/⟨I_w⟩ (script 80), que é limite superior do
  ruído SE o ruído for independente entre λ; se a própria componente comum for ruído correlacionado
  entre λ, σ₋ SUBESTIMA o ruído (σ₋² = σ²(1 − ρ)) e não serve como limite.

Uso:  python Scripts/81_sst_crosstalk_iv.py Data/RAW_SST/<arquivo>.fits [mais arquivos] [--passo 1]
Saída: Outputs/Reports/81_sst_crosstalk_iv.json e Outputs/Figures/cel81_crosstalk_iv.png

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saidas_sst as SA

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'
K_MAD = 1.4826
L_AZUL, L_VERM = 0, 14
LAGS = (1, 4, 16)


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def sigmas(v0, v14):
    s0, s14 = sigma_rob(v0), sigma_rob(v14)
    smais = sigma_rob(v0 + v14) / np.sqrt(2)
    smenos = sigma_rob(v0 - v14) / np.sqrt(2)
    sind = float(np.sqrt((s0 ** 2 + s14 ** 2) / 2))
    return {'sigma_0': s0, 'sigma_14': s14, 'sigma_mais': smais, 'sigma_menos': smenos,
            'razao_mais_ind': smais / sind, 'razao_menos_mais': smenos / smais}


def ajuste(S, Iw):
    s = sigma_rob(S)
    m = np.abs(S - np.median(S)) < 5 * s
    alfa, c = np.polyfit(Iw[m], S[m], 1)
    r = float(np.corrcoef(Iw[m], S[m])[0, 1])
    return float(alfa), float(c), r


def autocorr(mapa, msk, k):
    a, b = mapa[:, :-k], mapa[:, k:]
    m = msk[:, :-k] & msk[:, k:]
    if m.sum() < 1000:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def analisar(caminho, passo):
    out = {'arquivo': Path(caminho).name, 'passo': passo, 'posicoes': []}
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data                                      # (t, s, λ, y, x)
        nt, ns, nl = d.shape[:3]
        r0 = h[0].header.get('ATMOS_R0', None)
        out['ATMOS_R0_header'] = None if r0 is None else float(np.nanmean(r0))
        for t in range(nt):
            # λ do núcleo pelas medianas numa amostra esparsa
            amostra = np.array([np.asarray(d[t, 0, l, ::8, ::8], dtype=np.float32) for l in range(nl)])
            okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
            lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(nl)]))
            le = lambda s, l: np.asarray(d[t, s, l, ::passo, ::passo], dtype=np.float64)
            I0, I14, Ic = le(0, L_AZUL), le(0, L_VERM), le(0, lnuc)
            V0, V14 = le(3, L_AZUL), le(3, L_VERM)
            ok = (np.isfinite(I0) & (I0 != 0) & np.isfinite(I14) & np.isfinite(Ic)
                  & np.isfinite(V0) & (V0 != 0) & np.isfinite(V14) & (V14 != 0))
            asa = I0 / np.median(I0[ok])
            nuc = Ic / np.median(Ic[ok])
            calmo = ok & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)
            Iw = (I0 + I14) / 2
            S = (V0 + V14) / 2
            D = (V14 - V0) / 2
            mI = float(np.mean(Iw[calmo]))
            antes = sigmas(V0[calmo], V14[calmo])
            alfa, c, r = ajuste(S[calmo], Iw[calmo])
            # por quadrante
            ny, nx = calmo.shape
            quad = []
            for qy in range(2):
                for qx in range(2):
                    m = np.zeros_like(calmo)
                    m[qy * ny // 2:(qy + 1) * ny // 2, qx * nx // 2:(qx + 1) * nx // 2] = True
                    m &= calmo
                    if m.sum() > 5000:
                        aq, _, rq = ajuste(S[m], Iw[m])
                        quad.append({'qy': qy, 'qx': qx, 'alfa': aq, 'r': rq, 'n': int(m.sum())})
            V0c, V14c = V0 - alfa * I0, V14 - alfa * I14
            depois = sigmas(V0c[calmo], V14c[calmo])
            Sc = (V0c + V14c) / 2
            ac = {f'S_lag{k}': autocorr(np.where(calmo, S, 0), calmo, k) for k in LAGS}
            ac.update({f'D_lag{k}': autocorr(np.where(calmo, D, 0), calmo, k) for k in LAGS})
            ac.update({f'S_corrigido_lag{k}': autocorr(np.where(calmo, Sc, 0), calmo, k) for k in LAGS})
            pos = {'t': t, 'n_calmo': int(calmo.sum()), 'lambda_nucleo_idx': lnuc, 'I_asas_medio': mI,
                   'alfa': alfa, 'r': r, 'std_Iw_rel': float(np.std(Iw[calmo]) / mI), 'quadrantes': quad,
                   'antes': antes, 'depois': depois, 'autocorr': ac,
                   'eps_mais_antes': antes['sigma_mais'] / mI, 'eps_menos_antes': antes['sigma_menos'] / mI,
                   'eps_mais_depois': depois['sigma_mais'] / mI, 'eps_menos_depois': depois['sigma_menos'] / mI,
                   'r_ok': bool(abs(r) >= 0.3), 'C1_depois': bool(depois['razao_mais_ind'] < 1.05)}
            out['posicoes'].append(pos)
            aq = ' '.join(f"{q['alfa']:+.4f}" for q in quad)
            print(f"  t={t}: α = {alfa:+.4f}  r = {r:+.3f}   α por quadrante: {aq}")
            print(f"        antes:  σ₊/σ_ind = {antes['razao_mais_ind']:.3f}  σ₋/σ₊ = {antes['razao_menos_mais']:.3f}  "
                  f"ε₊ = {pos['eps_mais_antes']:.2e}  ε₋ = {pos['eps_menos_antes']:.2e}")
            print(f"        depois: σ₊/σ_ind = {depois['razao_mais_ind']:.3f}  σ₋/σ₊ = {depois['razao_menos_mais']:.3f}  "
                  f"ε₊ = {pos['eps_mais_depois']:.2e}  ε₋ = {pos['eps_menos_depois']:.2e}")
            print(f"        autocorr. lag 1/4/16 px:  S {ac['S_lag1']:+.2f}/{ac['S_lag4']:+.2f}/{ac['S_lag16']:+.2f}   "
                  f"D {ac['D_lag1']:+.2f}/{ac['D_lag4']:+.2f}/{ac['D_lag16']:+.2f}   "
                  f"S corrigido {ac['S_corrigido_lag1']:+.2f}/{ac['S_corrigido_lag4']:+.2f}/{ac['S_corrigido_lag16']:+.2f}")
    n_r = sum(p['r_ok'] for p in out['posicoes'])
    n_c = sum(p['C1_depois'] for p in out['posicoes'])
    out['n_r_ok'], out['n_C1_depois'] = n_r, n_c
    npos = len(out['posicoes'])
    out['decisao'] = ('CROSSTALK CONFIRMADO' if (SA.passa_fracao(n_r, npos) and SA.passa_fracao(n_c, npos))
                      else 'NÃO CONFIRMADO')
    boas = [p['eps_mais_depois'] for p in out['posicoes'] if p['C1_depois']]
    out['eps_V_corrigido'] = ({'mediano': float(np.median(boas)), 'faixa': [float(min(boas)), float(max(boas))]}
                              if out['decisao'] == 'CROSSTALK CONFIRMADO' else None)
    tetos = [p['eps_menos_antes'] for p in out['posicoes']]
    out['limite_superior_eps'] = {'mediano': float(np.median(tetos)), 'faixa': [float(min(tetos)), float(max(tetos))]}
    return out


def figura(todos, caminho):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    nmax = max(len(r['posicoes']) for r in todos)
    cores = ['#0072B2', '#D55E00']
    for k, r in enumerate(todos):
        cor = cores[k % 2]
        ts = np.array([p['t'] for p in r['posicoes']]) + (k - 0.5) * 0.15
        lab = r['arquivo'][8:27]
        axs[0].plot(ts, [p['antes']['razao_mais_ind'] for p in r['posicoes']], 'o', color=cor, mfc='white', ms=8,
                    label=f'{lab} antes')
        axs[0].plot(ts, [p['depois']['razao_mais_ind'] for p in r['posicoes']], 'o', color=cor, ms=8,
                    label=f'{lab} depois')
        for p, x in zip(r['posicoes'], ts):
            axs[1].plot([x] * len(p['quadrantes']), [q['alfa'] for q in p['quadrantes']], '.', color=cor, ms=8)
            axs[1].plot(x, p['alfa'], '_', color=cor, ms=16, mew=2)
            lags = list(LAGS)
            axs[2].plot(lags, [p['autocorr'][f'S_lag{j}'] for j in lags], '-o', color=cor, ms=4, lw=1)
            axs[2].plot(lags, [p['autocorr'][f'D_lag{j}'] for j in lags], ':o', color=cor, ms=4, lw=1, mfc='white')
    axs[0].axhline(1.05, color='0.3', ls='--', lw=1); axs[0].axhline(1.0, color='0.6', ls=':', lw=1)
    axs[0].set_xlabel('posição do mosaico ou varredura (t)'); axs[0].set_ylabel('σ₊ / σ_ind'); axs[0].set_xticks(range(nmax))
    axs[0].legend(fontsize=7)
    axs[1].axhline(0, color='0.6', lw=1)
    axs[1].set_xlabel('posição do mosaico ou varredura (t)'); axs[1].set_ylabel('α  (traço: global; pontos: quadrantes)')
    axs[1].set_xticks(range(nmax))
    axs[2].set_xscale('log'); axs[2].set_xticks(list(LAGS)); axs[2].set_xticklabels([str(j) for j in LAGS])
    axs[2].axhline(0, color='0.6', lw=1)
    axs[2].set_xlabel('deslocamento (px)'); axs[2].set_ylabel('autocorrelação  (— S, ··· D)')
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    ap.add_argument('--passo', type=int, default=1)
    a = ap.parse_args()
    todos = []
    for c in a.arquivos:
        print('=' * 78)
        print(Path(c).name)
        r = analisar(c, a.passo)
        todos.append(r)
        print(f"  |r| ≥ 0,3 em {r['n_r_ok']} de {len(r['posicoes'])};  σ₊/σ_ind < 1,05 após correção em {r['n_C1_depois']} de {len(r['posicoes'])}"
              f"  →  {r['decisao']}")
        if r['eps_V_corrigido']:
            e = r['eps_V_corrigido']
            print(f"  ε_V corrigido: {e['mediano']:.2e}  (faixa {e['faixa'][0]:.2e}–{e['faixa'][1]:.2e})")
        u = r['limite_superior_eps']
        print(f"  σ₋/⟨I⟩ sem correção (teto só se o ruído for independente entre λ): {u['mediano']:.2e}  (faixa {u['faixa'][0]:.2e}–{u['faixa'][1]:.2e})")
        rel = SA.gravar('81_sst_crosstalk_iv', r)                # relatório antes da figura
        cf = SA.figura('cel81_crosstalk_iv', r['etiqueta'])
        figura([r], cf)
        print(f"  Relatório: {rel}\n  Figura: {cf}")
    SA.resumo('81_sst_crosstalk_iv')


if __name__ == '__main__':
    main()
