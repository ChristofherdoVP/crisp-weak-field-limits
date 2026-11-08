"""
106_recalibracao_motor_atlas.py — recalibração da forma do núcleo do motor físico contra o atlas FTS/Neckel.

Script 105: o núcleo do motor (Src/physics.py, CALIB) é ~40% mais largo que o observado nos 4 conjuntos SST; nenhum
parâmetro de instrumento corrige (o instrumento só alarga). Contra o atlas (Data/Reference/atlas_fts_neckel_hs5000_8542.txt,
extraído da Célula 3 do Colab): o motor acerta I_min e a FWHM total (2,53–2,55 Å), únicos números usados na calibração
antiga, mas erra a forma interna: I(±0,33 Å) = 0,26 contra 0,42 do atlas (até 0,16 Ic de diferença).

Etapas (o motor em Src/physics.py NÃO é alterado; os parâmetros novos são aplicados só dentro deste script):
  1) Ajuste ao atlas, |Δλ| ≤ 1,5 Å do núcleo, perda robusta (soft-L1, escala 0,01 Ic — linhas de mistura do atlas):
     σ_f, γ_f, σ_c, γ_c, peso do núcleo w_c, profundidade e a velocidade do núcleo. Descritivo: mediana e P95 do
     |resíduo| em ±1,5 Å, máximo em ±0,4 Å.
  2) K1 (critério da calibração): o que importa para os limites é o desvio do filtro casado. Com o modelo CRISP/SST
     (LSF 10,5 pm, espúria 6%) nas grades reais do CRISP e do CRISP2, σ_k = ε Ĩ_k, template de campo fraco
     t_k = −C g λ₀² dĨ/dλ do perfil degradado: s(motor novo) contra s(atlas degradado do mesmo jeito).
     CALIBRAÇÃO SUFICIENTE PARA OS LIMITES se |s_motor/s_atlas − 1| ≤ 0,05 nas duas grades.
  3) C1' (repetição do C1 do 105 com o motor novo): perfis observados (relatório do 105) ajustados com FWHM, luz
     espúria e Δv. PARÂMETROS DO CRISP CONFIRMADOS se, nas duas séries, FWHM em 7,35–13,65 pm e espúria em 0,03–0,09.
  4) C2' (repetição do 104 com o motor novo e os parâmetros do CRISP): a₅₀ previsto (mesmo ruído e fator de população
     do 104) dentro da faixa aceita do 104 → MODELO REPRESENTA OS DADOS, para o CRISP e para o CRISP2.
Critérios definidos antes de rodar. Se K1 passa, os parâmetros novos viram uma versão nova do motor (v3) em
Src/physics.py, com a v2 mantida para a reprodutibilidade do artigo B (como no achado 16).

Uso:  python Scripts/106_recalibracao_motor_atlas.py
Saída: Outputs/Reports/106_recalibracao_motor_atlas.json, Outputs/Figures/cel106_atlas_motor.png
"""
import sys
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saidas_sst as SA
import Src.physics as P

REP = SA.REP
ATLAS = ROOT / 'Data' / 'Reference' / 'atlas_fts_neckel_hs5000_8542.txt'
C_ZEEMAN = 4.6686e-13
G_EFF = 1.10
LAMBDA0 = 8542.09
Z50 = 1.96
B_TAB2_BRANQ = 47.5
CHAVES = ['sigma_f', 'gamma_f', 'sigma_c', 'gamma_c', 'weight_c', 'depth']
GRADES = {'CRISP': (np.array([-650, -500, -350, -275, -200, -125, -50, 25, 100, 175, 250, 325, 400, 550, 700.]), 25.0),
          'CRISP2': (np.array([-656, -506, -356, -281, -206, -131, -56, 19, 94, 169, 244, 319, 394, 544, 694.]), 19.0)}
ORIG = {k: P.CALIB[k] for k in CHAVES}
V0 = P.V_CORE_KMS


def usar(par):
    for k in CHAVES:
        P.CALIB[k] = par[k]


def inst(instr, fwhm=10.5, stray=0.06):
    i = copy.deepcopy(P.INSTRUMENTAL_V2['CRISP/SST'])
    off, c = GRADES[instr]
    i['grade_dl_pm'] = (off - c) / 10.0
    i['fwhm_pm'], i['stray_light'] = float(fwhm), float(stray)
    return i


def degradar_I(I_lam, ins):
    z = np.zeros_like(I_lam)
    d = P.degradar_stokes({'I': I_lam, 'V': z, 'Q': z, 'U': z}, P.LAM, ins, rng_local=np.random.default_rng(0),
                          fator_ruido=0.0)
    lamA = d['lam_obs'] * 10.0 if np.median(d['lam_obs']) < 2000 else d['lam_obs']
    return d['I'].astype(float), lamA


def s_wfa(I, lamA, eps):
    t = -C_ZEEMAN * G_EFF * LAMBDA0 ** 2 * np.gradient(I, lamA)
    sig = eps * I
    return float(1.0 / np.sqrt(np.sum(t ** 2 / sig ** 2)))


def s_wfa_p(I, lamA, eps, p):
    t = -C_ZEEMAN * G_EFF * LAMBDA0 ** 2 * np.gradient(I, lamA)
    Iw = (I[0] + I[-1]) / 2
    sig = eps * Iw * (I / Iw) ** p
    u = t / (eps * I) ** 2
    return float(np.sqrt(np.sum(u ** 2 * sig ** 2)) / abs(np.sum(u * t)))


def motor_I(par, dv=0.0):
    usar(par)
    I = P.get_stokes(0.0, v_km_s=par['v_core'] + dv)['I'].astype(float)
    return I


def main():
    a = np.loadtxt(ATLAS)
    At = np.interp(P.LAM, a[:, 0], a[:, 1])
    kc = int(np.argmin(At))
    c = P.LAM[kc]
    m15 = np.abs(P.LAM - c) <= 0.15
    m04 = np.abs(P.LAM - c) <= 0.04

    # 1) ajuste ao atlas
    def res(x):
        par = dict(zip(CHAVES, x[:6]))
        par['v_core'] = x[6]
        return (motor_I(par) - At)[m15]
    x0 = [ORIG[k] for k in CHAVES] + [V0]
    sol = least_squares(res, x0, loss='soft_l1', f_scale=0.01,
                        bounds=([0.005, 0.0, 0.002, 0.0, 0.05, 0.5, -10.0], [0.6, 0.5, 0.2, 0.2, 0.98, 0.99, 10.0]),
                        x_scale=[0.05, 0.02, 0.01, 0.01, 0.1, 0.05, 0.5])
    novo = dict(zip(CHAVES, map(float, sol.x[:6])))
    novo['v_core'] = float(sol.x[6])
    antigo = dict(ORIG)
    antigo['v_core'] = V0
    In, Io = motor_I(novo), motor_I(antigo)
    r_n, r_o = In - At, Io - At
    desc = lambda r: {'mediana_abs_1p5A': float(np.median(np.abs(r[m15]))), 'P95_abs_1p5A': float(np.percentile(np.abs(r[m15]), 95)),
                      'max_abs_0p4A': float(np.max(np.abs(r[m04])))}
    out = {'parametros_antigos': antigo, 'parametros_novos': novo, 'residuo_antigo': desc(r_o), 'residuo_novo': desc(r_n)}
    print('1) ajuste ao atlas (±1,5 Å)')
    print('   parâmetros novos: ' + ', '.join(f"{k} = {v:.4f}" for k, v in novo.items()))
    for nome, d in (('antigo', out['residuo_antigo']), ('novo', out['residuo_novo'])):
        print(f"   motor {nome}: |resíduo| mediana {d['mediana_abs_1p5A']:.4f}, P95 {d['P95_abs_1p5A']:.4f} (±1,5 Å);  "
              f"máx {d['max_abs_0p4A']:.4f} (±0,4 Å)")

    # 2) K1
    r104 = json.loads((REP / '104_sst_limites_previsto_medido.json').read_text(encoding='utf-8'))
    eps_c, eps_c2, p_c2 = r104['eps_crisp'], r104['eps_crisp2'], r104['p_crisp2']
    k1 = {}
    print('2) K1: desvio do filtro casado (campo fraco), motor × atlas degradados (CRISP: 10,5 pm, 6%)')
    for instr in GRADES:
        ins = inst(instr)
        Ia, lamA = degradar_I(At, ins)
        Imn, _ = degradar_I(In, ins)
        Imo, _ = degradar_I(Io, ins)
        eps = eps_c if instr == 'CRISP' else eps_c2
        sa, sn, so = s_wfa(Ia, lamA, eps), s_wfa(Imn, lamA, eps), s_wfa(Imo, lamA, eps)
        k1[instr] = {'s_atlas': sa, 's_motor_novo': sn, 's_motor_antigo': so, 'razao_novo': sn / sa, 'razao_antigo': so / sa,
                     'passa': bool(abs(sn / sa - 1) <= 0.05)}
        print(f"   {instr}: s(motor antigo)/s(atlas) = {so / sa:.3f};  s(motor novo)/s(atlas) = {sn / sa:.3f}  "
              f"→ {'passa' if k1[instr]['passa'] else 'FALHA'}")
    out['K1'] = k1
    out['K1_decisao'] = ('CALIBRAÇÃO SUFICIENTE PARA OS LIMITES' if all(v['passa'] for v in k1.values())
                         else 'CALIBRAÇÃO INSUFICIENTE')
    print(f"   → {out['K1_decisao']}")

    # 3) C1' — perfis observados (relatório do 105)
    r105 = json.loads((REP / '105_sst_perfil_observado_modelo.json').read_text(encoding='utf-8'))
    print("3) C1': ajuste dos perfis observados com o motor novo")
    c1 = {}
    for tag, v in r105['conjuntos'].items():
        instr = v['instrumento']
        Iobs = np.array(v['perfil_obs'])

        def res_obs(x):
            Id, _ = degradar_I(motor_I(novo, x[2]), inst(instr, x[0], x[1]))
            return Id / ((Id[0] + Id[-1]) / 2) - Iobs
        dvs = np.arange(-3.0, 3.001, 0.25)
        dv0 = float(dvs[np.argmin([np.sum(res_obs([10.5, 0.06, d]) ** 2) for d in dvs])]) or 0.05
        s = least_squares(res_obs, [10.5, 0.07, np.clip(dv0, -2.9, 2.9)], bounds=([3.0, 0.0, -3.0], [30.0, 0.4, 3.0]),
                          x_scale=[2.0, 0.02, 0.2], diff_step=1e-3)
        c1[tag] = {'instrumento': instr, 'fwhm_pm': float(s.x[0]), 'stray': float(s.x[1]), 'dv_km_s': float(s.x[2]),
                   'rms': float(np.sqrt(np.mean(s.fun ** 2))), 'rms_105_motor_antigo': v['ajuste']['rms']}
        print(f"   {instr} {tag}: FWHM {s.x[0]:.1f} pm, espúria {s.x[1]:.3f}, Δv {s.x[2]:+.2f} km/s, rms "
              f"{c1[tag]['rms']:.4f} (motor antigo: {v['ajuste']['rms']:.4f})")
    cr = [x for x in c1.values() if x['instrumento'] == 'CRISP']
    out['C1'] = c1
    out['C1_decisao'] = ('PARÂMETROS DO CRISP CONFIRMADOS' if len(cr) == 2 and all(
        7.35 <= x['fwhm_pm'] <= 13.65 and abs(x['stray'] - 0.06) <= 0.03 for x in cr) else 'PARÂMETROS DO CRISP NÃO CONFIRMADOS')
    print(f"   → {out['C1_decisao']}")

    # 4) C2' — a₅₀ previsto com o motor novo (parâmetros do CRISP), como no 104
    usar(antigo)
    insB = copy.deepcopy(P.INSTRUMENTAL_V2['CRISP/SST'])
    Ib, lamB = degradar_I(Io, insB)
    f_pop = B_TAB2_BRANQ / (Z50 * s_wfa(Ib, lamB, 3.2e-3))
    c2 = {}
    print("4) C2': a₅₀ previsto com o motor novo (z = 1,96, fator de população do 104)")
    for instr in GRADES:
        Id, lamA = degradar_I(In, inst(instr))
        if instr == 'CRISP':
            a50 = Z50 * f_pop * s_wfa(Id, lamA, eps_c)
        else:
            a50 = Z50 * f_pop * s_wfa_p(Id, lamA, eps_c2, p_c2)
        faixa = r104['decisao'][instr]['faixa_G']
        c2[instr] = {'a50_previsto_G': a50, 'a50_previsto_104_G': r104['decisao'][instr]['previsto_G'], 'faixa_G': faixa,
                     'a50_N11_G': a50 / np.sqrt(11),
                     'decisao': 'MODELO REPRESENTA OS DADOS' if faixa[0] <= a50 <= faixa[1] else 'NÃO REPRESENTA'}
        print(f"   {instr}: {r104['decisao'][instr]['previsto_G']:.1f} G (motor antigo) → {a50:.1f} G (motor novo);  "
              f"faixa {faixa[0]:.1f}–{faixa[1]:.1f} G  →  {c2[instr]['decisao']};  média de 11: {a50 / np.sqrt(11):.1f} G")
    out['C2'] = c2
    out['f_pop'] = f_pop
    usar(antigo)

    REP.mkdir(parents=True, exist_ok=True)
    (REP / '106_recalibracao_motor_atlas.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    x = (P.LAM - c) * 1e4
    axs[0].plot(x, At, 'k', lw=1, label='atlas FTS/Neckel')
    axs[0].plot(x, Io, color='#D55E00', lw=1.2, label='motor atual')
    axs[0].plot(x, In, '--', color='#0072B2', lw=1.2, label='motor recalibrado')
    axs[0].set_xlim(-2000, 2000); axs[0].set_xlabel('Δλ do núcleo (mÅ)'); axs[0].set_ylabel('I/Ic')
    axs[0].legend(fontsize=8)
    axs[1].plot(x, r_o, color='#D55E00', lw=1, label='motor atual − atlas')
    axs[1].plot(x, r_n, color='#0072B2', lw=1, label='recalibrado − atlas')
    axs[1].axhline(0, color='0.5', lw=0.8); axs[1].set_xlim(-2000, 2000)
    axs[1].set_xlabel('Δλ do núcleo (mÅ)'); axs[1].set_ylabel('resíduo (Ic)'); axs[1].legend(fontsize=8)
    fig.suptitle(f"K1: {out['K1_decisao']};  C1': {out['C1_decisao']}", fontsize=9)
    fig.tight_layout()
    cf = SA.figura('cel106_atlas_motor', 'todos')
    fig.savefig(cf, dpi=120)
    plt.close(fig)
    print(f"\nRelatório: {REP / '106_recalibracao_motor_atlas.json'}\nFigura: {cf}")


if __name__ == '__main__':
    main()
