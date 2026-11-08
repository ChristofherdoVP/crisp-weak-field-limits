"""
108_regressao_bf_v3.py — calibração sintética do mapa de B_los·f do artigo A com o motor v3 (o 72 refeito).

Por que: o 72 calibrou o estimador de campo fraco com o motor v2, cujo núcleo é largo demais (105/106). O 107 mostrou
que, com o núcleo v3, o filtro de campo fraco devolve â/(B·f) = 1,04–1,05 sem ruído (v2: 1,005): a derivada por
np.gradient na grade grossa (75 mÅ) erra mais quando o núcleo é estreito. Aqui se mede isso para os DOIS estimadores
usados no A, nos quatro instrumentos do v3, e se verifica se o σ formal descreve o erro.

Estimadores:
  WF-pixel  : Src.sst_analise.campo_fraco (script 65 e o 72): mínimos quadrados ponderados com o dI/dλ do PRÓPRIO
              perfil (ruidoso), σ_B formal = (Σ w U²)^−½. O ruído em I entra no template.
  WF-mapa   : filtro dos mapas do A (89/100): template t = −C g λ₀² dĨ/dλ do perfil MÉDIO (aqui: sem ruído e sem
              campo), pesos u = t/σ², â = Σ u V / Σ u t, σ = s = √(Σu²σ²)/|Σut|.
Montagem (como o 72): varredura em B·f do 54 (Quieta, Rede, Mancha; 400 por classe; θ = 0), mas TODOS os perfis em
cada instrumento (sem sorteio de instrumento), motor v3 (Src/physics_v3: núcleo do atlas, V_CORE_KMS_V3,
INSTRUMENTAL_V3, ruído potencia_I no CRISP2), grade nativa. Para cada perfil, gerador próprio (semente, i): o ruído é
sorteado uma vez e escalado por k. k = 1 (uma varredura) e k = 1/√11 ≈ 0,30 (média de 11). Duas faixas no WF-pixel:
todos os pontos e |Δλ| ≤ 600 mÅ do núcleo (núcleo v3: λ0·(1 + v_core/c)).

Métricas (por estimador, k, instrumento, classe): ganho = mediana de estimativa/verdade sem ruído (B·f > 1 G);
pull = (estimativa − verdade)/σ, média (mediana) e desvio robusto (1,4826·MAD); cobertura |erro| ≤ σ (ideal 68,3%).
Critério (definido antes de rodar; o mesmo do 72), faixa completa, Quieta + Rede, por estimador:
  |ganho − 1| < 0,05 e desvio do pull em [0,9; 1,1], em k = 1 e k = 0,30, nos QUATRO instrumentos.
  Se falhar só pelo ganho: a correção é dividir pela g medida aqui (sem ruído, determinística); o pull com a
  estimativa corrigida é mostrado como descritivo (não reabre o critério).
Mancha (B·f até 2500 G): só descritivo (limite do campo fraco com o núcleo v3).

Uso:   python Scripts/108_regressao_bf_v3.py             (≈ 3–5 min)
       python Scripts/108_regressao_bf_v3.py --rapido    (teste: 100 por classe)
Saída: Outputs/Reports/108_regressao_bf_v3.json, Outputs/Figures/cel108_regressao_bf_v3.png
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Src.physics as P
import Src.physics_v3 as V3
from Src.config import FIGURES_DIR
from Src.dataset import SEED_BASE
from Src.evaluation import amostrar_varredura_bf
from Src.sst_analise import campo_fraco, C_ZEE_A, G_EFF, LAMBDA0_AR_NM
from Src.report_utils import save_report

INSTR = V3.INSTRUMENTAL_V3
SEED_VAR, SEED_DEG = SEED_BASE + 10800, SEED_BASE + 10801
K_GRADE = [1.0, 1 / np.sqrt(11)]
CLASSES = ['Quieta', 'Rede', 'Mancha']
LAM_NUCLEO_V3 = P.CALIB['lambda0'] * (1.0 + V3.V_CORE_KMS_V3 / P.CALIB['c_light'])


def gerar(B, F, ins):
    out = []
    for i in range(len(B)):
        st = V3.get_stokes(float(B[i]), filling=float(F[i]))
        d0 = V3.degradar_stokes(st, P.LAM, ins, rng_local=np.random.default_rng([SEED_DEG, i]), fator_ruido=0.0)
        d1 = V3.degradar_stokes(st, P.LAM, ins, rng_local=np.random.default_rng([SEED_DEG, i]), fator_ruido=1.0)
        I0, V0 = d0['I'].astype(np.float64), d0['V'].astype(np.float64)
        out.append({'lam': np.asarray(d1['lam_obs'], np.float64), 'I0': I0, 'V0': V0,
                    'nI': d1['I'].astype(np.float64) - I0, 'nV': d1['V'].astype(np.float64) - V0,
                    'sig': np.asarray(d1['sigma_por_ponto'], np.float64)})
    return out


def filtro_mapa(ins):
    d = V3.degradar_stokes(V3.get_stokes(0.0), P.LAM, ins, rng_local=np.random.default_rng(0), fator_ruido=0.0)
    I = d['I'].astype(np.float64)
    t = -C_ZEE_A * G_EFF * (LAMBDA0_AR_NM * 10.0) ** 2 * np.gradient(I, np.asarray(d['lam_obs'], np.float64) * 10.0)
    sig = V3.sigma_por_ponto(ins, I)
    u = t / sig ** 2
    return u, float(np.sum(u * t)), float(np.sqrt(np.sum(u ** 2 * sig ** 2)) / abs(np.sum(u * t)))


def estimar_pixel(perfis, k, faixa):
    est, sB = np.empty(len(perfis)), np.empty(len(perfis))
    for i, p in enumerate(perfis):
        I = p['I0'] + k * p['nI']
        V = p['V0'] + k * p['nV']
        est[i], sB[i] = campo_fraco(I, V, p['lam'], sigma=p['sig'] * max(k, 1e-9), faixa_mA=faixa,
                                    lam_nucleo_nm=LAM_NUCLEO_V3)
    return est, sB


def estimar_mapa(perfis, k, fm):
    u, ut, s = fm
    est = np.array([(p['V0'] + k * p['nV']) @ u / ut for p in perfis])
    return est, np.full(len(perfis), k * s if k > 0 else np.nan)


def rstd(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')


def metricas(est0, est, sB, verdade, m):
    mg = m & (verdade > 1.0)
    pull = (est - verdade) / sB
    return {'n': int(m.sum()),
            'ganho': float(np.median(est0[mg] / verdade[mg])) if mg.any() else float('nan'),
            'pull_media': float(np.median(pull[m])), 'pull_std': rstd(pull[m]),
            'cobertura_1sigma': float(np.mean(np.abs(est[m] - verdade[m]) <= sB[m])),
            'sigma_mediano_G': float(np.median(sB[m]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rapido', action='store_true')
    args = ap.parse_args()
    n_cls = 100 if args.rapido else 400
    t0 = time.time()
    B, F, y = amostrar_varredura_bf(n_cls, SEED_VAR, classes=(0, 1, 2))
    verdade = B * F
    print(f'{len(B)} perfis por instrumento | núcleo v3 em {LAM_NUCLEO_V3:.5f} nm | k: '
          + ', '.join(f'{k:.3f}' for k in K_GRADE))

    ESTS = {'WF-pixel (todos)': ('pixel', None), 'WF-pixel (±600 mÅ)': ('pixel', 600.0), 'WF-mapa': ('mapa', None)}
    res, fig_dados, ok = {}, {}, {e: True for e in ('WF-pixel (todos)', 'WF-mapa')}
    for nm, ins in INSTR.items():
        perfis = gerar(B, F, ins)
        fm = filtro_mapa(ins)
        res[nm] = {}
        for en, (tipo, faixa) in ESTS.items():
            f_est = (lambda k: estimar_pixel(perfis, k, faixa)) if tipo == 'pixel' else (lambda k: estimar_mapa(perfis, k, fm))
            est0, _ = f_est(0.0)
            res[nm][en] = {}
            for k in K_GRADE:
                est, sB = f_est(k)
                r = {cl: metricas(est0, est, sB, verdade, y == c) for c, cl in enumerate(CLASSES)}
                r['Quieta+Rede'] = metricas(est0, est, sB, verdade, y <= 1)
                g = r['Quieta+Rede']['ganho']
                r['Quieta+Rede (corrigido por g)'] = metricas(est0 / g, est / g, sB / g, verdade, y <= 1)
                res[nm][en][f'{k:.3f}'] = r
                if en in ok:
                    q = r['Quieta+Rede']
                    ok[en] &= bool(abs(q['ganho'] - 1) < 0.05 and 0.9 <= q['pull_std'] <= 1.1)
                if en == 'WF-mapa' and abs(k - 1) < 1e-9:
                    fig_dados[nm] = (est, sB)
        print(f'  {nm:11s} concluído ({time.time() - t0:.0f} s)')

    for en in ESTS:
        print(f'\n{"=" * 100}\nESTIMADOR: {en}\n{"=" * 100}')
        print(f'  {"k":>5s} {"instrumento":11s} {"classe":30s} {"n":>4s} {"ganho":>6s} {"σ [G]":>7s} '
              f'{"pull: média":>11s} {"desvio":>7s} {"cob. 1σ":>8s}')
        for nm in INSTR:
            for ks, r in res[nm][en].items():
                for cl in CLASSES + ['Quieta+Rede', 'Quieta+Rede (corrigido por g)']:
                    v = r[cl]
                    marca = ''
                    if cl == 'Quieta+Rede' and en in ok:
                        passa = abs(v['ganho'] - 1) < 0.05 and 0.9 <= v['pull_std'] <= 1.1
                        marca = '  ✓' if passa else '  ✗'
                    print(f'  {float(ks):5.3f} {nm:11s} {cl:30s} {v["n"]:4d} {v["ganho"]:6.3f} {v["sigma_mediano_G"]:7.1f} '
                          f'{v["pull_media"]:+11.2f} {v["pull_std"]:7.2f} {100 * v["cobertura_1sigma"]:7.0f}%{marca}')
            print()
    print('  Critério (faixa completa, Quieta + Rede, k = 1 e 0,30, quatro instrumentos: |ganho − 1| < 0,05 e '
          'desvio do pull em [0,9; 1,1]):')
    for en, v in ok.items():
        print(f'    {en:18s}: ' + ('ATENDIDO' if v else 'NÃO ATENDIDO'))
    print('  Ganho sem ruído (fator de correção dos mapas, Quieta + Rede): '
          + ', '.join(f'{nm} {res[nm]["WF-mapa"]["1.000"]["Quieta+Rede"]["ganho"]:.3f}' for nm in INSTR))

    fig, axs = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True, sharey=True)
    cores = ['#8b949e', '#2f81f7', '#f85149']
    for ax, nm in zip(axs, INSTR):
        est, sB = fig_dados[nm]
        for c, cl in enumerate(CLASSES):
            m = y == c
            ax.errorbar(verdade[m], est[m], yerr=sB[m], fmt='.', ms=3, lw=0.4, color=cores[c], alpha=0.6, label=cl)
        lim = [0.05, 3000]
        ax.plot(lim, lim, 'k:', lw=0.8)
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=10)
        ax.set_xlim(lim); ax.set_xlabel('B·f verdadeiro [G]'); ax.set_title(f'{nm} — WF-mapa, k = 1')
    axs[0].set_ylabel('B_los·f estimado [G]')
    axs[0].legend(fontsize=8)
    fig_path = Path(FIGURES_DIR) / 'cel108_regressao_bf_v3.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    save_report('108_regressao_bf_v3', {'n_por_classe': n_cls, 'k_grade': K_GRADE, 'rapido': args.rapido,
                                        'lam_nucleo_v3_nm': LAM_NUCLEO_V3,
                                        'criterio_atendido': ok, 'resultados': res})
    print(f'\n  Figura: {fig_path.name}   Tempo: {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
