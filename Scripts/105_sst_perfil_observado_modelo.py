"""
105_sst_perfil_observado_modelo.py — por que o modelo prevê a₅₀ 1,6–1,9× maior que o medido no CRISP2 (script 104)?

Script 104: CRISP previsto 44,9 G × medido 36,4/54,3 G → REPRESENTA; CRISP2 previsto 34,7 G × medido 18,1–21,8 G
→ NÃO REPRESENTA. O ruído entra medido nos dois (ε do 83, lei p do 93); o a₅₀ "medido" do CRISP2 (94) usa o perfil
médio OBSERVADO com esse ruído. Logo a diferença está no PERFIL: o do modelo (motor físico degradado com os
parâmetros do CRISP: LSF 10,5 pm, luz espúria 6%) contra o observado.

Para cada conjunto: perfil médio de Sol calmo observado Ĩ_k = ⟨I_k⟩/⟨(I₀ + I₁₄)/2⟩ na máscara dos scripts 79–104
(CRISP: varreduras 0, meio e última; CRISP2: as 4 posições). Métricas: profundidade 1 − min Ĩ, largura a meia
profundidade (interpolada), max |dĨ/dλ| (Å⁻¹). Ajuste do modelo ao perfil observado (mínimos quadrados em Ĩ_k):
FWHM da LSF (3–30 pm), luz espúria (0–0,40) e deslocamento Doppler do núcleo (±3 km/s em torno de V_CORE_KMS).
Com os parâmetros ajustados: a₅₀ previsto como no 104 (mesmo ruído, mesmo fator de população da Tabela 2 do B).

Critérios (definidos antes de rodar):
  C1 (controle) PARÂMETROS DO CRISP CONFIRMADOS se, nas duas séries, a FWHM ajustada está em 10,5 pm ± 30%
     (7,35–13,65 pm) e a luz espúria em 0,06 ± 0,03.
  C2 MODELO DO CRISP2 CALIBRADO REPRESENTA OS DADOS se o a₅₀ previsto com os parâmetros ajustados no CRISP2 cai
     na faixa aceita do 104 (verificação de consistência da cadeia: o ajuste é ao perfil, do qual o a₅₀ depende).
  Se o ajuste não reproduz o perfil (resíduo rms > 0,01 em Ĩ), registrar que a forma do modelo é o limite.
Descritivo: perfis observados × modelo com os parâmetros do CRISP; o CRISP2 olha Sol calmo e o CRISP uma região
ativa (máscara de intensidade "calma"): instrumento e alvo não se separam completamente com estes dados.

Uso:  python Scripts/105_sst_perfil_observado_modelo.py Data/RAW_SST/<arquivos dos 4 conjuntos>
Saída: Outputs/Reports/105_sst_perfil_observado_modelo.json, Outputs/Figures/cel105_perfis.png
"""
import sys
import copy
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saidas_sst as SA
from Src.physics import INSTRUMENTAL_V2, LAM, V_CORE_KMS, get_stokes, degradar_stokes

REP = SA.REP
Z50 = 1.96
C_ZEEMAN = 4.6686e-13
G_EFF = 1.10
LAMBDA0 = 8542.09
D_BORDA = 150
B_TAB2_BRANQ = 47.5
GRADES = {'CRISP': (np.array([-650, -500, -350, -275, -200, -125, -50, 25, 100, 175, 250, 325, 400, 550, 700.]), 25.0),
          'CRISP2': (np.array([-656, -506, -356, -281, -206, -131, -56, 19, 94, 169, 244, 319, 394, 544, 694.]), 19.0)}
FWHM_CRISP, STRAY_CRISP = 10.5, 0.06


def perfil_modelo(instr, fwhm_pm, stray, dv, B=0.0):
    off, c = GRADES[instr]
    inst = copy.deepcopy(INSTRUMENTAL_V2['CRISP/SST'])
    inst['grade_dl_pm'] = (off - c) / 10.0
    inst['fwhm_pm'] = float(fwhm_pm)
    inst['stray_light'] = float(stray)
    d = degradar_stokes(get_stokes(B, filling=1.0, v_km_s=V_CORE_KMS + dv), LAM, inst,
                        rng_local=np.random.default_rng(0), fator_ruido=0.0)
    return d


def norm_asas(I):
    return I / ((I[0] + I[-1]) / 2)


def metricas(I, lam_A):
    I = np.asarray(I, float)
    k = int(np.argmin(I))
    prof = 1 - I[k]
    meia = 1 - prof / 2
    xs = np.linspace(lam_A[0], lam_A[-1], 4000)
    Ii = np.interp(xs, lam_A, I)
    abaixo = xs[Ii <= meia]
    largura = float(abaixo[-1] - abaixo[0]) * 1e3 if abaixo.size else float('nan')
    return {'profundidade': float(prof), 'largura_meia_prof_mA': largura,
            'max_abs_dIdl_por_A': float(np.max(np.abs(np.gradient(I, lam_A))))}


def perfil_obs(caminho):
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt, nl = d.shape[0], d.shape[2]
        serie = nt > 4
        ts = sorted({0, nt // 2, nt - 1}) if serie else list(range(nt))
        perfis = []
        for t in ts:
            amostra = np.array([np.asarray(d[t, 0, l, ::8, ::8], dtype=np.float32) for l in range(nl)])
            okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
            lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(nl)]))
            I0, I14, Ic = (np.asarray(d[t, 0, l], dtype=np.float64) for l in (0, 14, lnuc))
            V0 = np.asarray(d[t, 3, 0], dtype=np.float64)
            ok = np.isfinite(I0) & (I0 != 0) & np.isfinite(I14) & np.isfinite(Ic) & np.isfinite(V0) & (V0 != 0)
            interior = distance_transform_edt(binary_fill_holes(ok)) >= D_BORDA
            asa = I0 / np.median(I0[ok])
            nuc = Ic / np.median(Ic[ok])
            m = ok & interior & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)
            Iw = float(np.mean(((I0 + I14) / 2)[m]))
            perfis.append(np.array([np.mean(np.asarray(d[t, 0, l], dtype=np.float64)[m]) for l in range(nl)]) / Iw)
    return np.array(perfis), ts


def ajustar(instr, Iobs):
    def res(p):
        return norm_asas(perfil_modelo(instr, p[0], p[1], p[2])['I'].astype(float)) - Iobs
    dvs = np.arange(-3.0, 3.001, 0.25)                 # partida: melhor Δv com os parâmetros do CRISP
    dv0 = float(dvs[np.argmin([np.sum(res([FWHM_CRISP, STRAY_CRISP, v]) ** 2) for v in dvs])])
    dv0 = float(np.clip(dv0 if dv0 != 0 else 0.05, -2.9, 2.9))
    sol = least_squares(res, x0=[FWHM_CRISP, STRAY_CRISP + 0.01, dv0], bounds=([3.0, 0.0, -3.0], [30.0, 0.40, 3.0]),
                        x_scale=[2.0, 0.02, 0.2], diff_step=1e-3)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return {'fwhm_pm': float(sol.x[0]), 'stray': float(sol.x[1]), 'dv_km_s': float(sol.x[2]), 'rms': rms}


def s_previsto(instr, fwhm, stray, dv, eps, p=None):
    d0 = perfil_modelo(instr, fwhm, stray, dv, 0.0)
    d1 = perfil_modelo(instr, fwhm, stray, dv, 100.0)
    I = d0['I'].astype(float)
    t = d1['V'].astype(float) / 100.0
    if p is None:
        sig = eps * I
        u = t / sig ** 2
    else:
        Iw = (I[0] + I[-1]) / 2
        sig = eps * Iw * (I / Iw) ** p
        u = t / (eps * I) ** 2                          # pesos dos mapas do 89
    return float(np.sqrt(np.sum(u ** 2 * sig ** 2)) / abs(np.sum(u * t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    r104 = json.loads((REP / '104_sst_limites_previsto_medido.json').read_text(encoding='utf-8'))
    eps_c, eps_c2, p_c2 = r104['eps_crisp'], r104['eps_crisp2'], r104['p_crisp2']
    # fator de população (P1 do 104): Tabela 2 do B / analítico na grade do B
    inst = copy.deepcopy(INSTRUMENTAL_V2['CRISP/SST'])
    rng = np.random.default_rng(0)
    d0 = degradar_stokes(get_stokes(0.0, v_km_s=V_CORE_KMS), LAM, inst, rng_local=rng, fator_ruido=0.0)
    d1 = degradar_stokes(get_stokes(100.0, v_km_s=V_CORE_KMS), LAM, inst, rng_local=rng, fator_ruido=0.0)
    I, t = d0['I'].astype(float), d1['V'].astype(float) / 100.0
    s1 = 1 / np.sqrt(np.sum(t ** 2 / (3.2e-3 * I) ** 2))
    f_pop = B_TAB2_BRANQ / (Z50 * s1)
    out = {'f_pop': f_pop, 'conjuntos': {}}
    figs = {}
    for c in a.arquivos:
        nome = Path(c).name
        tag = SA.etiqueta(nome)
        instr = 'CRISP2' if '2025-08-19' in nome else 'CRISP'
        off, cen = GRADES[instr]
        lamA = LAMBDA0 + off / 1000.0
        P, ts = perfil_obs(c)
        Iobs = P.mean(axis=0)
        mod = norm_asas(perfil_modelo(instr, FWHM_CRISP, STRAY_CRISP, 0.0)['I'].astype(float))
        aj = ajustar(instr, Iobs)
        Iaj = norm_asas(perfil_modelo(instr, aj['fwhm_pm'], aj['stray'], aj['dv_km_s'])['I'].astype(float))
        if instr == 'CRISP':
            a50_par = Z50 * f_pop * s_previsto(instr, FWHM_CRISP, STRAY_CRISP, 0.0, eps_c)
            a50_aj = Z50 * f_pop * s_previsto(instr, aj['fwhm_pm'], aj['stray'], aj['dv_km_s'], eps_c)
        else:
            a50_par = Z50 * f_pop * s_previsto(instr, FWHM_CRISP, STRAY_CRISP, 0.0, eps_c2, p_c2)
            a50_aj = Z50 * f_pop * s_previsto(instr, aj['fwhm_pm'], aj['stray'], aj['dv_km_s'], eps_c2, p_c2)
        r = {'instrumento': instr, 'varreduras_ou_posicoes': ts, 'perfil_obs': Iobs.tolist(),
             'dispersao_entre_varreduras': float(np.max(P.std(axis=0))),
             'metricas_obs': metricas(Iobs, lamA), 'metricas_modelo_parametros_CRISP': metricas(mod, lamA),
             'ajuste': aj, 'metricas_ajuste': metricas(Iaj, lamA),
             'a50_previsto_parametros_CRISP_G': a50_par, 'a50_previsto_ajustado_G': a50_aj}
        out['conjuntos'][tag] = r
        figs[tag] = (lamA, Iobs, mod, Iaj, instr)
        mo, mm = r['metricas_obs'], r['metricas_modelo_parametros_CRISP']
        print(f"{tag} ({instr}):  profundidade obs {mo['profundidade']:.3f} × modelo {mm['profundidade']:.3f};  largura "
              f"{mo['largura_meia_prof_mA']:.0f} × {mm['largura_meia_prof_mA']:.0f} mÅ;  max|dĨ/dλ| "
              f"{mo['max_abs_dIdl_por_A']:.2f} × {mm['max_abs_dIdl_por_A']:.2f} Å⁻¹")
        print(f"     ajuste: FWHM {aj['fwhm_pm']:.1f} pm, luz espúria {aj['stray']:.3f}, Δv {aj['dv_km_s']:+.2f} km/s, "
              f"rms {aj['rms']:.4f};  a₅₀ previsto {a50_par:.1f} G (parâmetros do CRISP) → {a50_aj:.1f} G (ajustado)")
    cr = [v for v in out['conjuntos'].values() if v['instrumento'] == 'CRISP']
    c1 = len(cr) == 2 and all(7.35 <= v['ajuste']['fwhm_pm'] <= 13.65 and abs(v['ajuste']['stray'] - 0.06) <= 0.03
                              for v in cr)
    out['C1'] = 'PARÂMETROS DO CRISP CONFIRMADOS' if c1 else 'PARÂMETROS DO CRISP NÃO CONFIRMADOS'
    faixa = r104['decisao']['CRISP2']['faixa_G']
    c2v = [v['a50_previsto_ajustado_G'] for v in out['conjuntos'].values() if v['instrumento'] == 'CRISP2']
    c2 = bool(c2v) and all(faixa[0] <= x <= faixa[1] for x in c2v)
    out['C2'] = ('MODELO DO CRISP2 CALIBRADO REPRESENTA OS DADOS' if c2
                 else 'MODELO DO CRISP2 CALIBRADO NÃO REPRESENTA')
    ruins = [k for k, v in out['conjuntos'].items() if v['ajuste']['rms'] > 0.01]
    out['ajustes_ruins'] = ruins
    print(f"\nC1: {out['C1']}\nC2: {out['C2']}  (faixa do 104: {faixa[0]:.1f}–{faixa[1]:.1f} G)")
    if ruins:
        print(f"Ajuste com rms > 0,01 (forma do modelo não reproduz o perfil): {ruins}")
    REP.mkdir(parents=True, exist_ok=True)
    (REP / '105_sst_perfil_observado_modelo.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                              encoding='utf-8')
    fig, axs = plt.subplots(1, len(figs), figsize=(4 * len(figs), 3.6), squeeze=False)
    for ax, (tag, (lamA, Io, Im, Ia, instr)) in zip(axs[0], figs.items()):
        x = (lamA - LAMBDA0) * 1e3
        ax.plot(x, Io, 'ko', ms=4, label='observado')
        ax.plot(x, Im, '-', color='#D55E00', lw=1.2, label='modelo (parâmetros do CRISP)')
        ax.plot(x, Ia, '--', color='#0072B2', lw=1.2, label='modelo ajustado')
        aj = out['conjuntos'][tag]['ajuste']
        ax.set_title(f"{instr} {tag}\nFWHM {aj['fwhm_pm']:.1f} pm, espúria {aj['stray']:.2f}", fontsize=8)
        ax.set_xlabel('Δλ (mÅ, de 8542,09 Å)'); ax.set_ylabel('Ĩ / asas')
        ax.legend(fontsize=6)
    fig.tight_layout()
    cf = SA.figura('cel105_perfis', 'todos')
    fig.savefig(cf, dpi=120)
    plt.close(fig)
    print(f"Relatório: {REP / '105_sst_perfil_observado_modelo.json'}\nFigura: {cf}")


if __name__ == '__main__':
    main()
