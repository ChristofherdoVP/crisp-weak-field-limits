"""
80_sst_ruido_v_asas.py — ruído de Stokes V do CRISP2 medido nas asas extremas, no Sol calmo (cubos de mosaico 2025-08-19).

Motivação (script 79): Q e U não servem como medida de ruído nestes cubos (desvios de calibração
com rampa em λ e eficiências de modulação diferentes: σQ/σU ≈ 0,89, σV/σQU ≈ 1,1–1,2). Aqui o ruído
de V é medido no próprio V, nas duas asas mais afastadas do núcleo (λ índice 0 ≈ −658 mÅ e 14 ≈ +692 mÅ).

No regime de campo fraco V ∝ dI/dλ, que tem sinais opostos nas asas azul e vermelha. Por isso:
  σ₊ = σ(V₀ + V₁₄)/√2   o sinal solar se cancela (estimador principal do ruído)
  σ₋ = σ(V₀ − V₁₄)/√2   o sinal solar se soma   (indicador de sinal, só informativo)
  σ₀, σ₁₄              dispersão em cada asa
σ por desvio absoluto mediano × 1,4826 (um desvio constante no campo não altera σ).
ε_V = σ₊ / ⟨I⟩ (média de I₀ e I₁₄ no Sol calmo).
Máscara de Sol calmo idêntica à do script 79 (só com I).

Critério de validade (definido antes de rodar), por posição do mosaico:
  C1: |σ₊ / √((σ₀² + σ₁₄²)/2) − 1| < 0,10
      (as três estimativas concordam: sinal solar desprezível e ruído independente entre as asas)
  Arquivo VÁLIDO se C1 vale em ≥ 3 das 4 posições; posições que falham são excluídas e reportadas.

A comparação 08:01 (r₀ 13 cm) × 10:48 (r₀ 6 cm) posição a posição é descritiva (sem critério).
Limitação: com só duas asas não há lei de escala σ ∝ I^p; o ruído vale para o nível de I das asas.

Uso:  python Scripts/80_sst_ruido_v_asas.py Data/RAW_SST/<arquivo>.fits [mais arquivos] [--passo 1]
Saída: Outputs/Reports/80_sst_ruido_v_asas.json e Outputs/Figures/cel80_ruido_v_asas.png

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


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def analisar(caminho, passo):
    out = {'arquivo': Path(caminho).name, 'passo': passo, 'posicoes': []}
    hist = []
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data                                      # (t, s, λ, y, x)
        nt, ns, nl = d.shape[:3]
        r0 = h[0].header.get('ATMOS_R0', None)
        out['ATMOS_R0_header'] = None if r0 is None else float(np.nanmean(r0))
        for t in range(nt):
            Iall = np.array([np.asarray(d[t, 0, l, ::passo, ::passo], dtype=np.float64) for l in range(nl)])
            V0 = np.asarray(d[t, 3, L_AZUL, ::passo, ::passo], dtype=np.float64)
            V14 = np.asarray(d[t, 3, L_VERM, ::passo, ::passo], dtype=np.float64)
            ok = (np.all(np.isfinite(Iall), axis=0) & (Iall[0] != 0)
                  & np.isfinite(V0) & (V0 != 0) & np.isfinite(V14) & (V14 != 0))
            Imed = np.array([np.median(Iall[l][ok]) for l in range(nl)])
            lnuc = int(np.argmin(Imed))
            asa = Iall[0] / Imed[0]
            nuc = Iall[lnuc] / Imed[lnuc]
            calmo = ok & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)
            v0, v14 = V0[calmo], V14[calmo]
            mI = float(np.mean((Iall[L_AZUL][calmo] + Iall[L_VERM][calmo]) / 2))
            s0, s14 = sigma_rob(v0), sigma_rob(v14)
            smais = sigma_rob(v0 + v14) / np.sqrt(2)
            smenos = sigma_rob(v0 - v14) / np.sqrt(2)
            sind = np.sqrt((s0 ** 2 + s14 ** 2) / 2)
            c1 = abs(smais / sind - 1) < 0.10
            pos = {'t': t, 'n_calmo': int(calmo.sum()), 'frac_calmo': float(calmo.sum() / max(ok.sum(), 1)),
                   'I_asas_medio': mI, 'sigma_0': s0, 'sigma_14': s14, 'sigma_mais': smais, 'sigma_menos': smenos,
                   'razao_mais_ind': float(smais / sind), 'razao_menos_mais': float(smenos / smais),
                   'mediana_V0_sobre_sigma': float(np.median(v0) / s0), 'mediana_V14_sobre_sigma': float(np.median(v14) / s14),
                   'eps_V': float(smais / mI), 'eps_V0': s0 / mI, 'eps_V14': s14 / mI, 'C1': bool(c1)}
            out['posicoes'].append(pos)
            z = (v0 + v14 - np.median(v0 + v14)) / np.sqrt(2) / smais
            hist.append(np.histogram(z, bins=np.linspace(-6, 6, 121))[0].tolist())
            print(f"  t={t}: Sol calmo {pos['frac_calmo']:.0%}  ε_V = {pos['eps_V']:.2e}  "
                  f"(asa azul {pos['eps_V0']:.2e}, asa vermelha {pos['eps_V14']:.2e})  "
                  f"σ₊/σ_ind = {pos['razao_mais_ind']:.3f}  σ₋/σ₊ = {pos['razao_menos_mais']:.3f}  "
                  f"→ {'passa' if c1 else 'FALHA'}")
    n_ok = sum(p['C1'] for p in out['posicoes'])
    out['n_posicoes_ok'] = n_ok
    out['decisao'] = 'VÁLIDO' if SA.passa_fracao(n_ok, len(out['posicoes'])) else 'REPROVADO'
    boas = [p['eps_V'] for p in out['posicoes'] if p['C1']]
    out['resumo'] = {'eps_V_mediano': float(np.median(boas)) if boas else None,
                     'eps_V_faixa': [float(min(boas)), float(max(boas))] if boas else None}
    out['_hist'] = hist
    return out


def figura(todos, caminho):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    cores = ['#0072B2', '#D55E00']
    larg = 0.38
    for k, r in enumerate(todos):
        ts = [p['t'] for p in r['posicoes']]
        eps = [p['eps_V'] * 1e3 for p in r['posicoes']]
        lab = f"{r['arquivo'][8:27]}  r₀ = {r['ATMOS_R0_header'] * 100:.1f} cm" if r['ATMOS_R0_header'] else r['arquivo'][8:27]
        b = axs[0].bar(np.array(ts) + (k - 0.5 * (len(todos) - 1)) * larg, eps, width=larg,
                       color=cores[k % 2], edgecolor='white', linewidth=1, label=lab)
        for bi, p in zip(b, r['posicoes']):
            if not p['C1']:
                bi.set_hatch('//')
        for h in r['_hist']:
            x = np.linspace(-6, 6, 121)
            c = 0.5 * (x[1:] + x[:-1])
            hh = np.array(h, dtype=float)
            axs[1].semilogy(c, hh / hh.sum() / (x[1] - x[0]), color=cores[k % 2], lw=1, alpha=0.7)
    c = np.linspace(-6, 6, 400)
    axs[1].semilogy(c, np.exp(-c ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', lw=1.2, label='gaussiana')
    axs[0].axhline(3.2, color='0.4', ls=':', lw=1, label='CRISP 2024 (artigo B)')
    axs[0].set_xlabel('posição do mosaico ou varredura (t)'); axs[0].set_ylabel('ε_V = σ₊ / ⟨I⟩ asas  (×10⁻³)')
    axs[0].set_xticks(range(max(len(r['posicoes']) for r in todos))); axs[0].legend(fontsize=7, loc='lower right')
    axs[1].set_xlabel('(V₀ + V₁₄)/√2 / σ₊'); axs[1].set_ylabel('densidade')
    axs[1].set_ylim(1e-5, 1); axs[1].legend(fontsize=7)
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
        s = r['resumo']
        print(f"  posições que passam em C1: {r['n_posicoes_ok']} de {len(r['posicoes'])}  →  {r['decisao']}")
        if s['eps_V_mediano'] is not None:
            print(f"  ε_V mediano (posições válidas): {s['eps_V_mediano']:.2e}  "
                  f"(faixa {s['eps_V_faixa'][0]:.2e}–{s['eps_V_faixa'][1]:.2e})")
        hist = r.pop('_hist')
        rel = SA.gravar('80_sst_ruido_v_asas', r)                # relatório antes da figura
        r['_hist'] = hist
        cf = SA.figura('cel80_ruido_v_asas', r['etiqueta'])
        figura([r], cf)
        r.pop('_hist')
        print(f"  Relatório: {rel}\n  Figura: {cf}")
    if len(todos) == 2 and len(todos[0]['posicoes']) == len(todos[1]['posicoes']):
        print('=' * 78)
        print('Comparação posição a posição (descritiva): ε_V(2º arquivo) / ε_V(1º arquivo)')
        for p1, p2 in zip(todos[0]['posicoes'], todos[1]['posicoes']):
            print(f"  t={p1['t']}: {p2['eps_V'] / p1['eps_V']:.3f}")
    SA.resumo('80_sst_ruido_v_asas')


if __name__ == '__main__':
    main()
