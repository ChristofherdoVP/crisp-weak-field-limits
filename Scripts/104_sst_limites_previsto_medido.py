"""
104_sst_limites_previsto_medido.py — artigo A: limites de detecção do modelo sintético nas grades REAIS do CRISP e
do CRISP2, contra os medidos nos dados.

Briefing do A: "a grade do CRISP é provisória até chegarem os λ reais; todos os números do CRISP mudam".
Os λ reais (WCS-TAB) e o ruído real estão medidos (scripts 83, 93, 96, 97, 100, 101, 103). Aqui:

PREVISTO (modelo; perfil de Sol calmo do motor físico degradado sem ruído pelo modelo CRISP/SST — LSF 10,5 pm e luz
espúria 6% também para o CRISP2, hipótese registrada; z₅₀ = 1,96 do ruído gaussiano):
  P1  CRISP, grade provisória do B (±90 pm), σ_k = 3,2e-3 Ĩ_k  → referência; fator de população
      f_pop = valor da Tabela 2 do B / valor analítico (perfil único), para branqueado e sem pesos (97);
  P2  CRISP, grade real 2024-10-08, σ_k = ε Ĩ_k com ε = média do script 83 nas duas séries;
  P3  CRISP2, grade real 2025-08-19, σ_k = ε Ĩ_w (Ĩ_k/Ĩ_w)^p (ε do 83, p mediano do 93 via relatório do 94,
      Ĩ_w = média das asas extremas), pesos dos mapas do 89 (u_k = t_k/(ε Ĩ_k)²) e também branqueado.
  a₅₀ = 1,96 · s · f_pop (filtro branqueado = o dos mapas do CRISP; s = (Σ u t)⁻¹ √(Σ u² σ²) em geral);
  planejamento: a₅₀ com média de 11 varreduras = a₅₀/√11 (ganho √N validado no 101).
MEDIDO:
  CRISP 09:49 e 10:10: a₅₀ = z_c · σ₁ (σ₁ do 101, z_c empírico do 103 para 5% de falso alarme; uma varredura) e
  a₅₀ da média de todas as varreduras = z_c(média) · σ₁/√N; extrapolado para N = 11.
  CRISP2 (sem varreduras repetidas): a₅₀ = 1,96 · s com o ruído real por λ (script 94, pesos do 89), posições
  sem quadros excluídos; z_c não medido no CRISP2 (no CRISP o gaussiano subestima em 5–12%).

Critério (definido antes de rodar), por instrumento, uma varredura:
  MODELO REPRESENTA OS DADOS se o a₅₀ previsto (P2 para o CRISP, P3 com os pesos do 89 para o CRISP2) está em
  [0,75 × menor medido; 1,25 × maior medido].  Senão: NÃO REPRESENTA (registrar a razão).
  (Para o CRISP o 97 já mostrava 44 G contra 34–48 G medidos com z = 1,96: o critério foi fixado conhecendo essa
   ordem de grandeza; para o CRISP2 a previsão é nova.)

Uso:  python Scripts/104_sst_limites_previsto_medido.py
Saída: Outputs/Reports/104_sst_limites_previsto_medido.json, Outputs/Reports/104_tabela_a50.csv,
       Outputs/Figures/cel104_a50_previsto_medido.png
"""
import sys
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saidas_sst as SA
from Src.physics import INSTRUMENTAL_V2, LAM, V_CORE_KMS, get_stokes, degradar_stokes

REP = SA.REP
Z50 = 1.96
N_PLANEJ = 11
B_TAB2 = {'branqueado': 47.5, 'sem_pesos': 57.8}     # CRISP, Tabela 2 do artigo B (G)
GRADE_CRISP = (np.array([-650, -500, -350, -275, -200, -125, -50, 25, 100, 175, 250, 325, 400, 550, 700.]), 25.0)
GRADE_CRISP2 = (np.array([-656, -506, -356, -281, -206, -131, -56, 19, 94, 169, 244, 319, 394, 544, 694.]), 19.0)
TAGS_CRISP = ['20241008T094951', '20241008T101050']
TAGS_CRISP2 = ['20250819T080131', '20250819T104836']


def ler(prefixo, tag):
    p = REP / f'{prefixo}_{tag}.json'
    if not p.exists():
        raise SystemExit(f'falta {p.name}')
    return json.loads(p.read_text(encoding='utf-8'))['data'][0]


def perfil(grade_pm):
    inst = copy.deepcopy(INSTRUMENTAL_V2['CRISP/SST'])
    if grade_pm is not None:
        inst['grade_dl_pm'] = grade_pm
    rng = np.random.default_rng(0)
    d0 = degradar_stokes(get_stokes(0.0, filling=1.0, v_km_s=V_CORE_KMS), LAM, inst, rng_local=rng, fator_ruido=0.0)
    d1 = degradar_stokes(get_stokes(100.0, filling=1.0, v_km_s=V_CORE_KMS), LAM, inst, rng_local=rng, fator_ruido=0.0)
    I = d0['I'].astype(float)
    t = d1['V'].astype(float) / 100.0
    M = np.column_stack([np.interp(LAM, d1['lam_obs'], np.eye(len(I))[k], left=0.0, right=0.0)
                         for k in range(len(I))])
    return I, t, M


def s_geral(u, t, sig):
    return float(np.sqrt(np.sum(u ** 2 * sig ** 2)) / abs(np.sum(u * t)))


def previsoes(eps_crisp, eps_crisp2, p_crisp2):
    out = {}
    # P1: grade do B
    I, t, M = perfil(None)
    sig = 3.2e-3 * I
    s_w = s_geral(t / sig ** 2, t, sig)
    tL = M @ t
    u_sp = M.T @ (tL / (tL @ tL))
    s_u = s_geral(u_sp, t, sig)
    f_pop = {'branqueado': B_TAB2['branqueado'] / (Z50 * s_w), 'sem_pesos': B_TAB2['sem_pesos'] / (Z50 * s_u)}
    out['P1_CRISP_grade_B'] = {'a50_branqueado_G': B_TAB2['branqueado'], 'a50_sem_pesos_G': B_TAB2['sem_pesos'],
                               'a50_analitico_branqueado_G': Z50 * s_w, 'a50_analitico_sem_pesos_G': Z50 * s_u,
                               'f_pop': f_pop}
    # P2: CRISP grade real
    off, c = GRADE_CRISP
    I, t, M = perfil((off - c) / 10.0)
    sig = eps_crisp * I
    s_w = s_geral(t / sig ** 2, t, sig)
    tL = M @ t
    s_u = s_geral(M.T @ (tL / (tL @ tL)), t, sig)
    out['P2_CRISP_grade_real'] = {'eps': eps_crisp, 'a50_branqueado_G': Z50 * s_w * f_pop['branqueado'],
                                  'a50_sem_pesos_G': Z50 * s_u * f_pop['sem_pesos']}
    # P3: CRISP2 grade real, lei σ ∝ I^p
    off, c = GRADE_CRISP2
    I, t, M = perfil((off - c) / 10.0)
    Iw = (I[0] + I[-1]) / 2
    sig = eps_crisp2 * Iw * (I / Iw) ** p_crisp2
    u89 = t / (eps_crisp2 * I) ** 2
    s_89 = s_geral(u89, t, sig)
    s_w = s_geral(t / sig ** 2, t, sig)
    s_mod = s_geral(t / (eps_crisp2 * I) ** 2, t, eps_crisp2 * I)
    out['P3_CRISP2_grade_real'] = {'eps': eps_crisp2, 'p': p_crisp2,
                                   'a50_pesos89_G': Z50 * s_89 * f_pop['branqueado'],
                                   'a50_branqueado_G': Z50 * s_w * f_pop['branqueado'],
                                   'a50_se_ruido_prop_I_G': Z50 * s_mod * f_pop['branqueado']}
    for k in ('P2_CRISP_grade_real', 'P3_CRISP2_grade_real'):
        for kk in list(out[k]):
            if kk.startswith('a50'):
                out[k][kk.replace('a50', 'a50_N11')] = out[k][kk] / np.sqrt(N_PLANEJ)
    return out


def medidos():
    crisp = {}
    for tag in TAGS_CRISP:
        r101, r103 = ler('101_sst_media_temporal', tag), ler('103_sst_falso_alarme', tag)
        s1, nt = r101['sigma_1_medido_G'], r101['n_varreduras']
        zc1, zcm = r103['zc_95_empirico_mediano'], r103['media']['zc_95_empirico']
        crisp[tag] = {'sigma_1_G': s1, 'z_c_uma': zc1, 'z_c_media': zcm, 'n_varreduras': nt,
                      'a50_uma_G': zc1 * s1, 'a50_uma_gauss_G': Z50 * s1,
                      'a50_media_todas_G': zcm * s1 / np.sqrt(nt), 'a50_N11_G': zcm * s1 / np.sqrt(N_PLANEJ)}
    crisp2 = {}
    p94 = REP / '94_sst_filtro_branqueado.json'
    if not p94.exists():
        raise SystemExit('falta 94_sst_filtro_branqueado.json')
    for r in json.loads(p94.read_text(encoding='utf-8'))['data']:
        tag = SA.etiqueta(r['arquivo'])
        pos = [q for q in r['posicoes'] if not q['quadros_excluidos']]
        a50 = [Z50 * q['s_antigo_ruido_real_G'] for q in pos]
        crisp2[tag] = {'eps': r['eps'], 'p': r['p'], 'posicoes_sem_exclusao': [q['t'] for q in pos],
                       'a50_uma_G': a50, 'a50_uma_min_G': min(a50), 'a50_uma_max_G': max(a50)}
    return crisp, crisp2


def main():
    eps_c = float(np.mean([ler('83_sst_ruido_v_sem_borda', t)['eps_V']['mediano'] for t in TAGS_CRISP]))
    eps_c2 = float(np.mean([ler('83_sst_ruido_v_sem_borda', t)['eps_V']['mediano'] for t in TAGS_CRISP2]))
    crisp, crisp2 = medidos()
    p_c2 = float(np.median([v['p'] for v in crisp2.values()]))
    prev = previsoes(eps_c, eps_c2, p_c2)
    P1, P2, P3 = prev['P1_CRISP_grade_B'], prev['P2_CRISP_grade_real'], prev['P3_CRISP2_grade_real']
    print(f"PREVISTO (modelo, z = 1,96)")
    print(f"  P1 CRISP, grade do B:   branqueado {P1['a50_branqueado_G']:.1f} G, sem pesos {P1['a50_sem_pesos_G']:.1f} G "
          f"(Tabela 2 do B; analítico {P1['a50_analitico_branqueado_G']:.1f} / {P1['a50_analitico_sem_pesos_G']:.1f} G)")
    print(f"  P2 CRISP, grade real (ε = {eps_c:.2e}):   branqueado {P2['a50_branqueado_G']:.1f} G, sem pesos "
          f"{P2['a50_sem_pesos_G']:.1f} G;  média de 11: {P2['a50_N11_branqueado_G']:.1f} / {P2['a50_N11_sem_pesos_G']:.1f} G")
    print(f"  P3 CRISP2, grade real (ε = {eps_c2:.2e}, p = {p_c2:.2f}):   pesos do 89 {P3['a50_pesos89_G']:.1f} G, "
          f"branqueado {P3['a50_branqueado_G']:.1f} G (com σ ∝ I: {P3['a50_se_ruido_prop_I_G']:.1f} G);  média de 11: "
          f"{P3['a50_N11_pesos89_G']:.1f} G")
    print("MEDIDO")
    for tag, m in crisp.items():
        print(f"  CRISP {tag}: σ₁ = {m['sigma_1_G']:.1f} G;  uma varredura {m['a50_uma_G']:.1f} G (z_c = {m['z_c_uma']:.2f}; "
              f"gaussiano {m['a50_uma_gauss_G']:.1f} G);  média de {m['n_varreduras']}: {m['a50_media_todas_G']:.1f} G;  "
              f"N = 11: {m['a50_N11_G']:.1f} G")
    for tag, m in crisp2.items():
        print(f"  CRISP2 {tag}: uma varredura {m['a50_uma_min_G']:.1f}–{m['a50_uma_max_G']:.1f} G "
              f"(posições sem exclusão {m['posicoes_sem_exclusao']}; z = 1,96)")
    dec = {}
    mc = [m['a50_uma_G'] for m in crisp.values()]
    lo, hi = 0.75 * min(mc), 1.25 * max(mc)
    v = P2['a50_branqueado_G']
    dec['CRISP'] = {'previsto_G': v, 'faixa_G': [lo, hi],
                    'decisao': 'MODELO REPRESENTA OS DADOS' if lo <= v <= hi else 'NÃO REPRESENTA',
                    'razao_previsto_medido': [v / x for x in mc]}
    mc2 = [x for m in crisp2.values() for x in m['a50_uma_G']]
    lo, hi = 0.75 * min(mc2), 1.25 * max(mc2)
    v = P3['a50_pesos89_G']
    dec['CRISP2'] = {'previsto_G': v, 'faixa_G': [lo, hi],
                     'decisao': 'MODELO REPRESENTA OS DADOS' if lo <= v <= hi else 'NÃO REPRESENTA',
                     'razao_previsto_medido': [v / min(mc2), v / max(mc2)]}
    for k, d in dec.items():
        print(f"  {k}: previsto {d['previsto_G']:.1f} G, faixa aceita {d['faixa_G'][0]:.1f}–{d['faixa_G'][1]:.1f} G  →  "
              f"{d['decisao']}")
    out = {'previsto': prev, 'medido_crisp': crisp, 'medido_crisp2': crisp2, 'decisao': dec,
           'eps_crisp': eps_c, 'eps_crisp2': eps_c2, 'p_crisp2': p_c2}
    REP.mkdir(parents=True, exist_ok=True)
    (REP / '104_sst_limites_previsto_medido.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                              encoding='utf-8')
    linhas = ['conjunto,tipo,a50_uma_varredura_G,a50_media_G,N_media,a50_N11_G,observacao',
              f"CRISP grade do B,previsto,{P1['a50_branqueado_G']:.1f},,,{P1['a50_branqueado_G'] / np.sqrt(11):.1f},Tabela 2 do B (branqueado)",
              f"CRISP grade real,previsto,{P2['a50_branqueado_G']:.1f},,,{P2['a50_N11_branqueado_G']:.1f},eps={eps_c:.2e}",
              f"CRISP2 grade real,previsto,{P3['a50_pesos89_G']:.1f},,,{P3['a50_N11_pesos89_G']:.1f},eps={eps_c2:.2e} p={p_c2:.2f}"]
    for tag, m in crisp.items():
        linhas.append(f"CRISP {tag},medido,{m['a50_uma_G']:.1f},{m['a50_media_todas_G']:.1f},{m['n_varreduras']},"
                      f"{m['a50_N11_G']:.1f},z_c={m['z_c_uma']:.2f}/{m['z_c_media']:.2f}")
    for tag, m in crisp2.items():
        linhas.append(f"CRISP2 {tag},medido,{m['a50_uma_min_G']:.1f}-{m['a50_uma_max_G']:.1f},,,,z=1.96 (z_c não medido)")
    (REP / '104_tabela_a50.csv').write_text('\n'.join(linhas) + '\n', encoding='utf-8')

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = 0
    ticks, rot = [], []
    for nome, prv, med in (('CRISP', P2['a50_branqueado_G'], [m['a50_uma_G'] for m in crisp.values()]),
                           ('CRISP2', P3['a50_pesos89_G'], mc2)):
        ax.plot([x], [prv], 's', color='#0072B2', ms=9, label='previsto (modelo)' if x == 0 else None)
        ax.plot([x + 0.25] * len(med), med, 'o', color='#D55E00', ms=6, label='medido (1 varredura)' if x == 0 else None)
        d = dec[nome]
        ax.add_patch(plt.Rectangle((x - 0.2, d['faixa_G'][0]), 0.65, d['faixa_G'][1] - d['faixa_G'][0], color='0.9', zorder=0))
        ticks.append(x + 0.12)
        rot.append(f"{nome}\n{d['decisao']}")
        x += 1
    ax.plot([0], [P1['a50_branqueado_G']], 'x', color='0.4', ms=8, label='CRISP, grade provisória do B')
    ax.set_xticks(ticks); ax.set_xticklabels(rot, fontsize=8)
    ax.set_ylabel('a₅₀ por pixel, uma varredura (G)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    cf = SA.figura('cel104_a50_previsto_medido', 'todos')
    fig.savefig(cf, dpi=120)
    plt.close(fig)
    print(f"\nRelatório: {REP / '104_sst_limites_previsto_medido.json'}\nTabela: {REP / '104_tabela_a50.csv'}\nFigura: {cf}")


if __name__ == '__main__':
    main()
