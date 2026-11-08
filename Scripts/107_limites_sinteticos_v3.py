"""
107_limites_sinteticos_v3.py — limites de detecção sintéticos do artigo A com o motor v3 (só filtro, sem D²NN).

Por que: o 105/106 mostraram que o núcleo do motor v2 é largo demais (desvio do filtro 1,63× o do atlas). Os números
sintéticos do A (B·f₅₀ por instrumento e planejamento do CRISP, script 54) vinham do v2. Aqui eles são refeitos com
Src/physics_v3.py (núcleo recalibrado + INSTRUMENTAL_V3), com o MESMO estimador aplicado aos dados reais:
  template de campo fraco t_k = −C g λ₀² dĨ/dλ (np.gradient do perfil sem ruído e sem campo, na grade NATIVA do
  instrumento), pesos u_k = t_k/σ_k², â = Σ u_k V_k / Σ u_k t_k (scripts 89/100).
Src/physics.py, Src/dataset.py e o artigo B não são tocados: os perfis são gerados aqui, com as funções do v3.

Configurações (mesmos B, f e sementes em todas; ruído sorteado uma vez e escalado por k — exato):
  v3        : motor v3 + INSTRUMENTAL_V3 (IBIS, CRISP grade real, CRISP2, ViSP)      → números do A
  v2-núcleo : motor v2 + os MESMOS instrumentos do v3                                  → efeito só do núcleo
  B         : motor v2 + INSTRUMENTAL_V2 (IBIS, CRISP grade provisória, ViSP)          → razão v3/B (decisão do B)
  Doppler   : v3, perfis em V_CORE + Δv (Δv = 0,5; 1; 2 km/s), template fixo em Δv = 0 → descritivo
Pontos de operação: k = 1 (uma varredura) e k = 1/√11 (média de 11; ganho √N validado nos dados pelo 101).

Por instrumento: 1500 perfis sem campo (limiar a 5% de falso alarme = quantil 95% de |â|) e a varredura Quieta + Rede
do 54 (amostrar_varredura_bf, 1000 por classe, B·f 0,05–150 G, θ = 0). B·f₅₀ com limiar_bf50_fisico (β = 1, IC 95%
por bootstrap). Teoria: B·f₅₀ = z₅₀·s/g, com s = √(Σu²σ²)/|Σut| analítico e g = mediana de â/(B·f) sem ruído.

Critérios (definidos antes de rodar):
  K-a (cadeia; mesmo critério dos scripts 54/69/70): no v3, 4 instrumentos × (k = 1, k = 1/√11) = 8 casos;
      teoria dentro do IC 95% em ≥ 7 dos 8 e mediana de |simulado/teoria − 1| < 0,05.
      (Revisado após o teste rápido e ANTES da rodada completa: a primeira versão, |razão − 1| ≤ 0,05 em todos os 8
      casos, é mais estreita que a dispersão estatística do estimador — IC 95% de ~±6% com 2000 perfis.)
  K-b (consistência com o 106): CRISP v3 (k = 1) a até 5% do a₅₀ previsto no C2' do 106 (27,9 G).
      CRISP2: o C2' do 106 usou a LSF e a luz espúria do CRISP (10,5 pm, 6%); o CRISP2 v3 usa os valores ajustados
      no próprio 106 (11,9 pm, 0%). Por isso a referência do CRISP2 é a faixa aceita do 104 (medida nos mapas):
      CRISP2 v3 (k = 1) dentro dela.
  Se K-a falhar, a cadeia v3 tem erro e os números NÃO devem ser usados.
Descritivo: razões v3/v2-núcleo e v3/B; Doppler B·f₅₀(Δv)/B·f₅₀(0).

Uso:   python Scripts/107_limites_sinteticos_v3.py            (≈ 5–10 min)
       python Scripts/107_limites_sinteticos_v3.py --rapido   (teste: 300 nulos, 200 por classe, 40 bootstraps)
Saída: Outputs/Reports/107_limites_sinteticos_v3.json, Outputs/Figures/cel107_limites_v3.png
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Src.physics as P
import Src.physics_v3 as V3
from Src.config import FIGURES_DIR, REPORTS_DIR
from Src.dataset import SEED_BASE
from Src.evaluation import amostrar_varredura_bf, limiar_bf50_fisico
from Src.report_utils import save_report

C_ZEEMAN, G_EFF, LAMBDA0_A = 4.6686e-13, 1.10, 8542.09
FAP = 0.05
Z_C = norm.isf(FAP / 2)
Z50 = brentq(lambda m: norm.sf(Z_C - m) + norm.cdf(-Z_C - m) - 0.5, 0.0, 10.0)
K_PONTOS = {'1 varredura': 1.0, 'média de 11': 1.0 / np.sqrt(11)}
DV_DOPPLER = [0.5, 1.0, 2.0]
SEED_VAR, SEED_NULO, SEED_RUIDO = SEED_BASE + 10700, SEED_BASE + 10701, SEED_BASE + 10702
A50_106_CRISP = 27.9            # usado só se o relatório do 106 não estiver disponível
FAIXA_104_CRISP2 = None         # idem (sem relatório do 104 → K-b do CRISP2 fica "sem referência")


def stokes_v3(B, f, dv=0.0):
    return V3.get_stokes(float(B), filling=float(f), v_km_s=V3.V_CORE_KMS_V3 + dv)


def stokes_v2(B, f, dv=0.0):
    return P.get_stokes(float(B), filling=float(f), v_km_s=P.V_CORE_KMS + dv)


CONFIGS = {
    'v3': (stokes_v3, V3.INSTRUMENTAL_V3),
    'v2-nucleo': (stokes_v2, V3.INSTRUMENTAL_V3),
    'B': (stokes_v2, P.INSTRUMENTAL_V2),
}


def filtro(stokes_fn, ins):
    """Template e pesos do filtro de campo fraco a partir do perfil sem ruído e sem campo (como nos dados)."""
    d = V3.degradar_stokes(stokes_fn(0.0, 1.0), P.LAM, ins, rng_local=np.random.default_rng(0), fator_ruido=0.0)
    I = d['I'].astype(np.float64)
    lamA = np.asarray(d['lam_obs'], np.float64) * 10.0
    t = -C_ZEEMAN * G_EFF * LAMBDA0_A ** 2 * np.gradient(I, lamA)
    sig = V3.sigma_por_ponto(ins, I)
    u = t / sig ** 2
    s = float(np.sqrt(np.sum(u ** 2 * sig ** 2)) / abs(np.sum(u * t)))
    return {'u': u, 'ut': float(np.sum(u * t)), 's': s, 'n_lam': len(I)}


def gerar_V(stokes_fn, ins, B, F, seed, dv=0.0):
    """V sem ruído e ruído unitário×σ (k = 1) na grade nativa; mesma semente → mesmo ruído entre configurações."""
    V0, nV = [], []
    for i in range(len(B)):
        st = stokes_fn(B[i], F[i], dv)
        d0 = V3.degradar_stokes(st, P.LAM, ins, rng_local=np.random.default_rng([seed, i]), fator_ruido=0.0)
        d1 = V3.degradar_stokes(st, P.LAM, ins, rng_local=np.random.default_rng([seed, i]), fator_ruido=1.0)
        v0 = d0['V'].astype(np.float64)
        V0.append(v0)
        nV.append(d1['V'].astype(np.float64) - v0)
    return np.array(V0), np.array(nV)


def avaliar(fil, V0_var, nV_var, nV_nulo, bf, rng, n_boot, k_pontos):
    a0 = V0_var @ fil['u'] / fil['ut']
    m = bf > 1.0
    g = float(np.median(a0[m] / bf[m]))
    out = {'ganho': g, 's_analitico_G': fil['s']}
    for rot, k in k_pontos.items():
        a_nulo = k * (nV_nulo @ fil['u'] / fil['ut'])
        a_var = a0 + k * (nV_var @ fil['u'] / fil['ut'])
        thr = float(np.quantile(np.abs(a_nulo), 1 - FAP))
        det = (np.abs(a_var) > thr).astype(float)
        est, lo, hi, motivo = limiar_bf50_fisico(bf, det, rng, n_boot=n_boot, fap=FAP)
        teoria = Z50 * k * fil['s'] / g
        out[rot] = {'k': k, 'limiar_G': thr, 'sigma_a_sim_G': float(np.std(a_nulo)), 'sigma_a_teoria_G': k * fil['s'],
                    'bf50': est, 'ic95': [lo, hi], 'motivo': motivo, 'bf50_teoria': teoria,
                    'razao_sim_teoria': est / teoria if np.isfinite(est) else float('nan'),
                    'curva': curva(bf, det)}
    return out


def curva(bf, det, n=19):
    bins = np.logspace(np.log10(0.05), np.log10(150), n)
    idx = np.digitize(bf, bins) - 1
    return [{'bf': float(np.sqrt(bins[i] * bins[i + 1])), 'n': int((idx == i).sum()),
             'p': float(det[idx == i].mean()) if (idx == i).any() else float('nan')} for i in range(n - 1)]


def ler_referencias():
    a50_crisp, faixa_c2, fonte = A50_106_CRISP, FAIXA_104_CRISP2, 'constantes do script'
    try:
        r106 = json.loads((Path(REPORTS_DIR) / '106_recalibracao_motor_atlas.json').read_text(encoding='utf-8'))
        a50_crisp = float(r106['C2']['CRISP']['a50_previsto_G'])
        fonte = '106'
        r104 = json.loads((Path(REPORTS_DIR) / '104_sst_limites_previsto_medido.json').read_text(encoding='utf-8'))
        faixa_c2 = [float(x) for x in r104['decisao']['CRISP2']['faixa_G']]
        fonte = '104 e 106'
    except (FileNotFoundError, KeyError) as e:
        print(f'  AVISO: relatório não encontrado/sem chave ({e}); usando {fonte}.')
    return a50_crisp, faixa_c2, fonte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rapido', action='store_true')
    args = ap.parse_args()
    n_nulo, n_cls, n_boot = (300, 200, 40) if args.rapido else (1500, 1000, 200)
    t0 = time.time()
    rng = np.random.default_rng(SEED_BASE + 10703)
    a50_crisp_106, faixa_c2_104, fonte = ler_referencias()

    Bv, Fv, yv = amostrar_varredura_bf(n_cls, SEED_VAR, classes=(0, 1))
    bf = Bv * Fv
    B0 = np.zeros(n_nulo)
    print(f'Varredura Quieta + Rede: {len(bf)} perfis (B·f {bf.min():.2f}–{bf.max():.0f} G); sem campo: {n_nulo}; '
          f'z₅₀ = {Z50:.3f} a {100 * FAP:.0f}% de falso alarme')

    res = {}
    for cfg, (fn, instr) in CONFIGS.items():
        res[cfg] = {}
        for nm, ins in instr.items():
            fil = filtro(fn, ins)
            V0v, nVv = gerar_V(fn, ins, Bv, Fv, SEED_RUIDO)
            _, nV0 = gerar_V(fn, ins, B0, B0, SEED_NULO)
            res[cfg][nm] = avaliar(fil, V0v, nVv, nV0, bf, rng, n_boot, K_PONTOS)
            res[cfg][nm]['n_lam'] = fil['n_lam']
            print(f'  {cfg:10s} {nm:11s} concluído ({time.time() - t0:.0f} s)')

    # Doppler (v3, template fixo em Δv = 0; nulo e varredura no mesmo Δv)
    dop = {}
    for nm, ins in V3.INSTRUMENTAL_V3.items():
        fil = filtro(stokes_v3, ins)
        dop[nm] = {}
        for dv in DV_DOPPLER:
            V0v, nVv = gerar_V(stokes_v3, ins, Bv, Fv, SEED_RUIDO, dv)
            _, nV0 = gerar_V(stokes_v3, ins, B0, B0, SEED_NULO, dv)
            r = avaliar(fil, V0v, nVv, nV0, bf, rng, max(n_boot // 2, 20), {'1 varredura': 1.0})
            b0 = res['v3'][nm]['1 varredura']['bf50']
            dop[nm][f'{dv:.1f}'] = {'ganho': r['ganho'], 'bf50': r['1 varredura']['bf50'],
                                    'relativo': r['1 varredura']['bf50'] / b0 if np.isfinite(b0) else float('nan')}
        print(f'  Doppler    {nm:11s} concluído ({time.time() - t0:.0f} s)')

    # --- Tabelas ---
    print(f'\n{"=" * 104}\nB·f₅₀ A 5% DE FALSO ALARME [G] — filtro de campo fraco ponderado (estimador dos dados)\n{"=" * 104}')
    print(f'  {"config":10s} {"instrumento":11s} {"nλ":>4s} {"g":>6s} {"s [G]":>7s} │ '
          + ' │ '.join(f'{rot + ": sim [IC95]":>27s} {"teoria":>7s} {"razão":>6s}' for rot in K_PONTOS))
    ka_dentro, ka_desvio = [], []
    for cfg in res:
        for nm, r in res[cfg].items():
            cel = []
            for rot in K_PONTOS:
                q = r[rot]
                ic = f'[{q["ic95"][0]:.1f}–{q["ic95"][1]:.1f}]' if np.isfinite(q['ic95'][0]) else ''
                txt = f'{q["bf50"]:6.1f} {ic:>14s}' if np.isfinite(q['bf50']) else f'{q["motivo"][:20]:>21s}'
                cel.append(f'{txt:>27s} {q["bf50_teoria"]:7.1f} {q["razao_sim_teoria"]:6.3f}')
                if cfg == 'v3':
                    lo, hi = q['ic95']
                    ka_dentro.append(bool(np.isfinite(lo) and lo <= q['bf50_teoria'] <= hi))
                    ka_desvio.append(abs(q['razao_sim_teoria'] - 1) if np.isfinite(q['razao_sim_teoria']) else np.inf)
            print(f'  {cfg:10s} {nm:11s} {r["n_lam"]:4d} {r["ganho"]:6.3f} {r["s_analitico_G"]:7.2f} │ ' + ' │ '.join(cel))
        print()

    print('  Razões de B·f₅₀ (k = 1, simulado):')
    for nm in V3.INSTRUMENTAL_V3:
        v3 = res['v3'][nm]['1 varredura']['bf50']
        v2n = res['v2-nucleo'][nm]['1 varredura']['bf50']
        bB = res['B'].get(nm, {}).get('1 varredura', {}).get('bf50', float('nan'))
        print(f'    {nm:11s} v3/v2-núcleo = {v3 / v2n:5.3f}' + (f'   v3/B = {v3 / bB:5.3f}' if np.isfinite(bB) else ''))

    print('\n  Doppler (v3, template fixo; B·f₅₀(Δv)/B·f₅₀(0), k = 1):')
    for nm, d in dop.items():
        print(f'    {nm:11s} ' + '   '.join(f'Δv = {dv} km/s: {v["relativo"]:5.3f} (g {v["ganho"]:.3f})' for dv, v in d.items()))

    c_v3 = res['v3']['CRISP/SST']['1 varredura']['bf50']
    c2_v3 = res['v3']['CRISP2/SST']['1 varredura']['bf50']
    kb_crisp = bool(np.isfinite(c_v3) and abs(c_v3 / a50_crisp_106 - 1) <= 0.05)
    kb_crisp2 = (None if faixa_c2_104 is None
                 else bool(np.isfinite(c2_v3) and faixa_c2_104[0] <= c2_v3 <= faixa_c2_104[1]))
    n_dentro, desvio_med = int(sum(ka_dentro)), float(np.median(ka_desvio))
    ka_ok = bool(n_dentro >= 7 and desvio_med < 0.05)
    print(f'\n  K-a (v3: teoria no IC 95% em {n_dentro} de {len(ka_dentro)} casos; mediana |sim/teoria − 1| = '
          f'{100 * desvio_med:.1f}%; critério ≥ 7 e < 5%): '
          + ('ATENDIDO — cadeia v3 consistente' if ka_ok else 'NÃO ATENDIDO — NÃO usar os números'))
    print(f'  K-b CRISP  (v3 {c_v3:.1f} G × 106 {a50_crisp_106:.1f} G, ±5%; referência: {fonte}): '
          + ('ATENDIDO' if kb_crisp else 'NÃO ATENDIDO'))
    if kb_crisp2 is None:
        print(f'  K-b CRISP2 (v3 {c2_v3:.1f} G): sem a faixa do 104 — não avaliado')
    else:
        print(f'  K-b CRISP2 (v3 {c2_v3:.1f} G × faixa medida do 104 {faixa_c2_104[0]:.1f}–{faixa_c2_104[1]:.1f} G): '
              + ('ATENDIDO' if kb_crisp2 else 'NÃO ATENDIDO'))

    # --- Figura: curvas de detecção v3 (k = 1 e média de 11) com a forma física ---
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True, sharey=True)
    xx = np.logspace(np.log10(0.05), np.log10(150), 300)
    for ax, nm in zip(axs, V3.INSTRUMENTAL_V3):
        for rot, cor in zip(K_PONTOS, ('#1f77b4', '#d62728')):
            q = res['v3'][nm][rot]
            c = [p for p in q['curva'] if p['n'] > 0]
            ax.plot([p['bf'] for p in c], [p['p'] for p in c], 'o', ms=3, color=cor, label=f'{rot}: {q["bf50"]:.1f} G')
            s_eff = q['bf50_teoria'] / Z50
            ax.plot(xx, norm.sf(Z_C - xx / s_eff) + norm.cdf(-xx / s_eff - Z_C), color=cor, lw=0.8)
            q2 = res['v2-nucleo'][nm][rot]
            ax.axvline(q2['bf50'], color=cor, ls=':', lw=0.8)
        ax.axhline(0.5, color='0.6', lw=0.5)
        ax.set_xscale('log'); ax.set_title(nm); ax.set_xlabel('B·f [G]')
        ax.legend(fontsize=7, loc='upper left')
    axs[0].set_ylabel('P(detecção), 5% FA')
    fig.suptitle('Motor v3: curvas de detecção do filtro ponderado (pontos: simulação; linha: forma física; '
                 'pontilhado: B·f₅₀ com o núcleo v2)', fontsize=9)
    fig_path = Path(FIGURES_DIR) / 'cel107_limites_v3.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)

    for cfg in res:
        for nm in res[cfg]:
            for rot in K_PONTOS:
                res[cfg][nm][rot].pop('curva', None) if cfg != 'v3' else None
    save_report('107_limites_sinteticos_v3', {
        'fap': FAP, 'z_c': Z_C, 'z50': Z50, 'k_pontos': K_PONTOS, 'n_nulo': n_nulo, 'n_por_classe': n_cls,
        'n_boot': n_boot, 'rapido': args.rapido, 'estimador': 'campo fraco ponderado (u = t/σ²) + limiar_bf50_fisico',
        'referencias': {'a50_crisp_106_G': a50_crisp_106, 'faixa_crisp2_104_G': faixa_c2_104, 'fonte': fonte},
        'criterios': {'K_a': ka_ok, 'K_a_n_dentro_ic': n_dentro, 'K_a_desvio_mediano': desvio_med, 'K_b_crisp': kb_crisp, 'K_b_crisp2': kb_crisp2},
        'resultados': res, 'doppler': dop})
    print(f'\n  Figura: {fig_path.name}   Tempo: {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
