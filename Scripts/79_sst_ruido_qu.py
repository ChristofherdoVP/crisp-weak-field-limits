"""
79_sst_ruido_qu.py — ruído do CRISP2 medido por Stokes Q e U no Sol calmo (cubos de mosaico 2025-08-19).

Motivação (script 78): cada posição do mosaico foi observada uma vez, então o ruído não pode ser
medido pela diferença entre varreduras. No centro do disco, em Sol calmo, o sinal de polarização
linear em Ca II 854,2 nm fica muito abaixo do ruído: a dispersão de Q e U ali é essencialmente ruído.

Para cada arquivo, cada posição do mosaico (t) e cada comprimento de onda (λ):
  - máscara de Sol calmo (só com I, para não enviesar a polarização):
      I na asa azul (λ índice 0) entre 0,92 e 1,08 × mediana da posição  (exclui mancha e poros)
      I no núcleo (λ de menor I médio) < 1,15 × mediana do núcleo          (exclui plage/rede brilhante)
  - σ_Q, σ_U, σ_V por desvio absoluto mediano × 1,4826 (robusto a pixels com sinal);
  - ⟨I⟩ no Sol calmo; razões ε_Q = σ_Q/⟨I⟩, ε_U = σ_U/⟨I⟩.
Lei de escala: ajuste log σ = log ε₀ + p log(⟨I⟩/⟨I⟩_asa) usando os 15 λ (σ = média de σ_Q e σ_U).

Critério de validade de Q,U como medida de ruído (definido antes de rodar), por posição e λ:
  (1) |mediana(Q)| < 0,1 σ_Q e |mediana(U)| < 0,1 σ_U   (média compatível com zero)
  (2) |σ_Q/σ_U − 1| < 0,10                                (Q e U com a mesma dispersão)
  Se (1) e (2) valem em ≥ 80% dos pares (posição, λ) de um arquivo → VÁLIDO; senão → REPROVADO
  (há sinal ou crosstalk residual e o método é descartado para esse arquivo).

Uso:  python Scripts/79_sst_ruido_qu.py Data/RAW_SST/<arquivo>.fits [mais arquivos] [--passo 2]
Saída: Outputs/Reports/79_sst_ruido_qu.json e Outputs/Figures/cel79_ruido_qu.png

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


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def ler_lambdas(h):
    """Δλ (pm) a partir da tabela WCS-TAB, se existir; senão índices."""
    for hdu in h[1:]:
        cols = getattr(hdu, 'columns', None)
        if cols is None:
            continue
        for nome in cols.names:
            if 'WAVE' in nome.upper():
                w = np.asarray(hdu.data[nome]).ravel()
                w = np.unique(w[np.isfinite(w)])
                if w.size >= 2:
                    return w
    return None


def analisar(caminho, passo):
    out = {'arquivo': Path(caminho).name, 'passo': passo, 'posicoes': []}
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data                                      # (t, s, λ, y, x)
        nt, ns, nl = d.shape[:3]
        r0 = None
        try:
            r0 = float(np.nanmean(h[0].header.get('ATMOS_R0', np.nan)))
        except Exception:
            pass
        out['ATMOS_R0_header'] = r0
        lam = ler_lambdas(h)
        out['lambda_nm'] = None if lam is None else [float(x) for x in lam[:nl]]
        for t in range(nt):
            Iall = np.array([np.asarray(d[t, 0, l, ::passo, ::passo], dtype=np.float64) for l in range(nl)])
            V0 = np.asarray(d[t, 3, 0, ::passo, ::passo], dtype=np.float64)
            ok = np.all(np.isfinite(Iall), axis=0) & (Iall[0] != 0) & np.isfinite(V0) & (V0 != 0)
            Imed = np.array([np.median(Iall[l][ok]) for l in range(nl)])
            lnuc = int(np.argmin(Imed))
            asa = Iall[0] / Imed[0]
            nuc = Iall[lnuc] / Imed[lnuc]
            calmo = ok & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)
            pos = {'t': t, 'frac_valida': float(ok.mean()), 'frac_calmo': float(calmo.sum() / max(ok.sum(), 1)),
                   'n_calmo': int(calmo.sum()), 'lambda_nucleo_idx': lnuc, 'por_lambda': []}
            for l in range(nl):
                Q = np.asarray(d[t, 1, l, ::passo, ::passo], dtype=np.float64)[calmo]
                U = np.asarray(d[t, 2, l, ::passo, ::passo], dtype=np.float64)[calmo]
                V = np.asarray(d[t, 3, l, ::passo, ::passo], dtype=np.float64)[calmo]
                I = Iall[l][calmo]
                sQ, sU, sV = sigma_rob(Q), sigma_rob(U), sigma_rob(V)
                mI = float(np.mean(I))
                c1 = abs(np.median(Q)) < 0.1 * sQ and abs(np.median(U)) < 0.1 * sU
                c2 = abs(sQ / sU - 1) < 0.10
                pos['por_lambda'].append({
                    'l': l, 'I_medio': mI, 'sigma_Q': sQ, 'sigma_U': sU, 'sigma_V': sV,
                    'eps_Q': sQ / mI, 'eps_U': sU / mI, 'eps_V': sV / mI,
                    'mediana_Q_sobre_sigma': float(np.median(Q) / sQ), 'mediana_U_sobre_sigma': float(np.median(U) / sU),
                    'razao_sQ_sU': sQ / sU, 'criterio_1': bool(c1), 'criterio_2': bool(c2)})
            # lei de escala σ ∝ I^p ao longo da linha
            Ib = np.array([p['I_medio'] for p in pos['por_lambda']])
            sb = np.array([(p['sigma_Q'] + p['sigma_U']) / 2 for p in pos['por_lambda']])
            x = np.log(Ib / Ib[0])
            y = np.log(sb / Ib[0])
            p_exp, log_e0 = np.polyfit(x, y, 1)
            pos['lei_escala'] = {'p': float(p_exp), 'eps_na_asa': float(np.exp(log_e0)),
                                 'eps_mediano': float(np.median(sb / Ib))}
            out['posicoes'].append(pos)
            print(f"  t={t}: Sol calmo {pos['frac_calmo']:.0%} ({pos['n_calmo']} px)  "
                  f"ε mediano = {pos['lei_escala']['eps_mediano']:.2e}  p = {p_exp:.2f}  "
                  f"σV/σQU (mediana) = {np.median([q['sigma_V'] / ((q['sigma_Q'] + q['sigma_U']) / 2) for q in pos['por_lambda']]):.2f}")
    pares = [q for p in out['posicoes'] for q in p['por_lambda']]
    frac_ok = float(np.mean([q['criterio_1'] and q['criterio_2'] for q in pares]))
    out['frac_pares_ok'] = frac_ok
    out['decisao'] = 'VÁLIDO' if frac_ok >= 0.8 else 'REPROVADO'
    eps_all = [p['lei_escala']['eps_mediano'] for p in out['posicoes']]
    p_all = [p['lei_escala']['p'] for p in out['posicoes']]
    out['resumo'] = {'eps_mediano_posicoes': float(np.median(eps_all)), 'eps_faixa': [float(min(eps_all)), float(max(eps_all))],
                     'p_mediano': float(np.median(p_all)), 'p_faixa': [float(min(p_all)), float(max(p_all))]}
    return out


def figura(todos, caminho):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    cores = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for k, r in enumerate(todos):
        ls = '-' if k == 0 else '--'
        for p in r['posicoes']:
            Ib = np.array([q['I_medio'] for q in p['por_lambda']])
            e = np.array([(q['eps_Q'] + q['eps_U']) / 2 for q in p['por_lambda']])
            axs[0].plot(range(len(e)), e, ls=ls, color=cores[p['t'] % 4], marker='o', ms=3,
                        label=f"{r['arquivo'][8:27]} t={p['t']}")
            sb = e * Ib
            axs[1].loglog(Ib / Ib[0], sb / Ib[0], ls=ls, color=cores[p['t'] % 4], marker='o', ms=3)
    axs[0].set_xlabel('índice de λ'); axs[0].set_ylabel('σ_QU / ⟨I⟩  (Sol calmo)')
    axs[1].set_xlabel('⟨I⟩ / ⟨I⟩ asa'); axs[1].set_ylabel('σ_QU / ⟨I⟩ asa')
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    ap.add_argument('--passo', type=int, default=2)
    a = ap.parse_args()
    todos = []
    for c in a.arquivos:
        print('=' * 78)
        print(Path(c).name)
        r = analisar(c, a.passo)
        todos.append(r)
        s = r['resumo']
        print(f"  pares (posição, λ) que passam no critério: {r['frac_pares_ok']:.0%}  →  {r['decisao']}")
        print(f"  ε mediano entre posições: {s['eps_mediano_posicoes']:.2e}  (faixa {s['eps_faixa'][0]:.2e}–{s['eps_faixa'][1]:.2e})"
              f"   p mediano: {s['p_mediano']:.2f}  (faixa {s['p_faixa'][0]:.2f}–{s['p_faixa'][1]:.2f})")
        rel = SA.gravar('79_sst_ruido_qu', r)                   # relatório antes da figura
        cf = SA.figura('cel79_ruido_qu', r['etiqueta'])
        figura([r], cf)
        print(f"  Relatório: {rel}\n  Figura: {cf}")
    SA.resumo('79_sst_ruido_qu')


if __name__ == '__main__':
    main()
