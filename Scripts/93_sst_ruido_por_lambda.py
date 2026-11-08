"""
93_sst_ruido_por_lambda.py — de onde vem o excesso de ruído do filtro casado (script 92: R = 1,55–1,82)?

Nas sobreposições entre ladrilhos vizinhos do mesmo horário (só os pares com alinhamento válido no 92), o mesmo
pedaço do Sol foi visto duas vezes com ~20 s de intervalo. Para cada par de comprimentos de onda espelhados
(k, 14 − k), com ΔV = V_a − V_b (b amostrado por vizinho mais próximo, sem interpolar) e passa-alta (subtrai a
suavização gaussiana de 32 px, o que remove α(x, y) — diferente em cada posição do detector — e a evolução
lenta):
  σ₊,k = σ(ΔV_k + ΔV_{14−k}) / 2   o sinal de campo fraco (antissimétrico) se cancela → só ruído
  σ₋,k = σ(ΔV_k − ΔV_{14−k}) / 2   ruído + diferença de sinal (seeing, alinhamento, evolução)
  (cada ΔV é diferença de duas observações; a soma de duas tem variância 4σ² → divide por 2)
Com σ₊,k atribuído aos dois λ do par (ruído independente entre λ), recalcula o desvio esperado de â de cada
ladrilho com os MESMOS pesos usados nos mapas (script 89: u_k = t_k/(ε Ĩ_k)², quadros ruins excluídos):
  s_ruído = √(Σ u_k² σ₊,k²) / Σ u_k t_k
e compara com o observado na mesma região e com o mesmo passa-alta:
  Q = σ(Δâ) / √(s_ruído,a² + s_ruído,b²)
Também ajusta a lei σ₊,k ∝ Ĩ_k^p ao longo da linha (descritivo).

Critérios (definidos antes de rodar), por par de ladrilhos com alinhamento válido:
  C0 (o método mede ruído): σ₊,0 / (ε₈₃ Ĩ_0) entre 0,8 e 1,25 nas asas extremas (ε₈₃ do script 83);
  EXCESSO É RUÍDO se C0 e Q entre 0,85 e 1,15
     (o ruído por λ, medido, explica o ruído do filtro: o modelo ∝ I com o ε das asas é que estava errado);
  EXCESSO É DIFERENÇA DE SINAL se C0, Q > 1,15 e N > 1,15, onde N = σ(Δâ_u)/σ(ẑ_u): com os mesmos pesos u_k
     (exclusões dos dois ladrilhos), Δâ_u = Σ u_k ΔV_k / D e o filtro nulo ẑ_u = Σ sgn_k u_k ΔV_k / D
     (sgn = −1 na asa azul, +1 na vermelha). Como t_k é antissimétrico, ẑ cancela o sinal de campo fraco e
     guarda o ruído com a mesma variância: N > 1 indica diferença de sinal entre as duas observações
     (o que sobra vem de seeing/alinhamento/evolução, não de ruído);
     [σ₋/σ₊ por par de λ fica como descritivo: no teste sintético ele mal se moveu (1,03–1,07) com diferença
      de seeing, porque por λ o ruído domina; o filtro nulo combina os λ coerentemente.]
  INDEFINIDO nos demais casos.
Por arquivo: a classificação de ≥ 2/3 dos pares válidos (mínimo 1). Em qualquer caso, s_ruído é o desvio de â
devido só ao ruído, e a₅₀ ≈ 1,96 s_ruído é a sensibilidade por pixel do filtro.

Uso:  python Scripts/93_sst_ruido_por_lambda.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
      (precisa dos mapas e relatórios dos scripts 89 e 92)
Saída: Outputs/Reports/93_sst_ruido_por_lambda.json, Outputs/Figures/cel93_ruido_por_lambda.png
"""
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'
DAT = ROOT / 'Outputs' / 'Data'


def carregar(nome):
    esp = importlib.util.spec_from_file_location(nome.replace('.py', ''), ROOT / 'Scripts' / nome)
    m = importlib.util.module_from_spec(esp)
    esp.loader.exec_module(m)
    return m


S92 = carregar('92_sst_sobreposicao_ladrilhos.py')
S89 = carregar('89_sst_mapas_blos.py')
sigma_rob, suave_nan, amostrar = S92.sigma_rob, S92.suave_nan, S92.amostrar
LIM_FRACO = S92.LIM_FRACO
W_HP = 32


def hp(x):
    return x - suave_nan(x, W_HP)


def pesos(Imed, lam, eps, excl):
    tk = -S89.C_ZEEMAN * S89.G_EFF * S89.LAMBDA0 ** 2 * np.gradient(Imed, lam)
    uk = np.where([l not in excl for l in range(len(Imed))], tk / (eps * Imed) ** 2, 0.0)
    return tk, uk, float(np.sum(uk * tk))


def analisar(caminho, rel92):
    nome = Path(caminho).name
    hora = nome[19:27]
    rot = f'{nome[8:18].replace("-", "")}T{hora.replace("_", "")}'
    eps = S89.EPS[hora]
    ruins = S89.RUINS.get(hora, {})
    validos = [p for p in rel92['pares'] if p.get('sobreposicao') and p['alinhamento_valido']]
    out = {'arquivo': nome, 'eps_83': eps, 'pares': []}
    if not validos:
        print('  nenhum par com alinhamento válido no script 92')
        return out
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt, _, nl, ny, nx = d.shape
        lam = S89.ler_grade(h, nl)
        afins, _ = S92.afins_wcs(h)
        As = {t: np.load(DAT / f'blos_f_crisp2_{rot}_t{t}.npy').astype(np.float64) for t in range(nt)}
        for p in validos:
            a, b = p['par']
            dy, dx = p['desloc_refino_px']
            T = S92.a_para_b(afins[a], afins[b])
            Tr = T.copy()
            Tr[:, 2] = T[:, 2] - T[:, :2] @ np.array([dx, dy], dtype=float)
            ys, xs = np.arange(0, ny, 8), np.arange(0, nx, 8)
            cob = np.isfinite(amostrar(As[b], Tr, ys, xs, 0)) & np.isfinite(As[a][::8, ::8][:len(ys), :len(xs)])
            yy, xx = np.where(cob)
            y0, y1 = max(0, ys[yy.min()] - 16), min(ny, ys[yy.max()] + 16)
            x0, x1 = max(0, xs[xx.min()] - 16), min(nx, xs[xx.max()] + 16)
            ysF, xsF = np.arange(y0, y1), np.arange(x0, x1)
            Aa = As[a][y0:y1, x0:x1]
            Ab = amostrar(As[b], Tr, ysF, xsF, 0)
            fraco = np.isfinite(Aa) & np.isfinite(Ab) & (np.abs(Aa) < LIM_FRACO) & (np.abs(Ab) < LIM_FRACO)
            Ia = [np.asarray(d[a, 0, l, y0:y1, x0:x1], dtype=np.float64) for l in range(nl)]
            Iw = float(np.mean(((Ia[0] + Ia[14]) / 2)[fraco]))
            Imed = np.array([np.mean(I[fraco]) for I in Ia]) / Iw
            dV = []
            for l in range(nl):
                Va = np.asarray(d[a, 3, l, y0:y1, x0:x1], dtype=np.float64)
                Vb = amostrar(np.asarray(d[b, 3, l], dtype=np.float64), Tr, ysF, xsF, 0)
                dV.append(hp(np.where(fraco, (Va - Vb) / Iw, np.nan)))
            excl = set(ruins.get(a, [])) | set(ruins.get(b, []))
            sig_mais, sig_menos = np.full(nl, np.nan), np.full(nl, np.nan)
            linhas = []
            for k in range(8):
                j = 14 - k
                if k in excl or j in excl:
                    continue
                m = np.isfinite(dV[k]) & np.isfinite(dV[j])
                sp = sigma_rob((dV[k] + dV[j])[m]) / 2 if j != k else sigma_rob(dV[k][m]) / np.sqrt(2)
                sm = sigma_rob((dV[k] - dV[j])[m]) / 2 if j != k else np.nan
                sig_mais[k] = sig_mais[j] = sp
                sig_menos[k] = sig_menos[j] = sm
                linhas.append({'par_lambda': [k, j], 'I_rel': float((Imed[k] + Imed[j]) / 2), 'sigma_mais': sp,
                               'sigma_menos': None if np.isnan(sm) else sm,
                               'razao_menos_mais': None if np.isnan(sm) else sm / sp})
            # lei de escala σ₊ ∝ Ĩ^p
            ok = np.isfinite(sig_mais)
            p_exp = float(np.polyfit(np.log(Imed[ok]), np.log(sig_mais[ok]), 1)[0]) if ok.sum() >= 3 else None
            c0 = float(sig_mais[0] / (eps * Imed[0])) if np.isfinite(sig_mais[0]) else None
            # pesos dos mapas (89) para cada ladrilho; ruído por λ medido
            res_t = {}
            for t in (a, b):
                tk, uk, D = pesos(Imed, lam, eps, ruins.get(t, []))
                usar = uk != 0
                s_teo = 1 / np.sqrt(D)
                if np.any(usar & ~np.isfinite(sig_mais)):
                    s_ruido = np.nan          # algum λ usado no mapa sem σ medido (excluído no outro ladrilho)
                    faltam = [int(l) for l in np.where(usar & ~np.isfinite(sig_mais))[0]]
                else:
                    s_ruido = float(np.sqrt(np.sum((uk * np.nan_to_num(sig_mais)) ** 2)) / D)
                    faltam = []
                res_t[t] = {'s_teorico': float(s_teo), 's_ruido': s_ruido, 'lambdas_sem_sigma': faltam}
            if any(np.isnan(res_t[t]['s_ruido']) for t in (a, b)):
                # aproximação: σ₊ dos λ faltantes interpolado em Ĩ pela lei de escala
                for t in (a, b):
                    if np.isnan(res_t[t]['s_ruido']) and p_exp is not None:
                        tk, uk, D = pesos(Imed, lam, eps, ruins.get(t, []))
                        k0 = np.where(ok)[0][0]
                        sig_full = np.where(ok, sig_mais, sig_mais[k0] * (Imed / Imed[k0]) ** p_exp)
                        res_t[t]['s_ruido'] = float(np.sqrt(np.sum((uk * sig_full) ** 2)) / D)
                        res_t[t]['s_ruido_com_interpolacao'] = True
            dA = hp(np.where(fraco, Aa - Ab, np.nan))
            s_obs = sigma_rob(dA[np.isfinite(dA)])
            s_pred = float(np.hypot(res_t[a]['s_ruido'], res_t[b]['s_ruido']))
            s_teo2 = float(np.hypot(res_t[a]['s_teorico'], res_t[b]['s_teorico']))
            Q = s_obs / s_pred
            nuc = [x['razao_menos_mais'] for x in linhas if x['par_lambda'][0] in (5, 6) and x['razao_menos_mais']]
            r_nuc = float(np.mean(nuc)) if nuc else None
            # filtro nulo com os pesos da união das exclusões
            _, uu, Du = pesos(Imed, lam, eps, sorted(excl))
            sgn = np.sign(np.arange(nl) - 7.0)
            da_u = sum(uu[l] * np.nan_to_num(dV[l]) for l in range(nl) if uu[l] != 0) / Du
            z_u = sum(sgn[l] * uu[l] * np.nan_to_num(dV[l]) for l in range(nl) if uu[l] != 0) / Du
            mz = np.all([np.isfinite(dV[l]) for l in range(nl) if uu[l] != 0], axis=0)
            Nn = sigma_rob(da_u[mz]) / sigma_rob(z_u[mz])
            c0_ok = c0 is not None and 0.8 <= c0 <= 1.25
            if c0_ok and 0.85 <= Q <= 1.15:
                cls = 'EXCESSO É RUÍDO'
            elif c0_ok and Q > 1.15 and Nn > 1.15:
                cls = 'EXCESSO É DIFERENÇA DE SINAL'
            else:
                cls = 'INDEFINIDO'
            # correlação do ruído entre pares de λ (combinações simétricas)
            comb = {x['par_lambda'][0]: (dV[x['par_lambda'][0]] + dV[x['par_lambda'][1]]) for x in linhas
                    if x['par_lambda'][0] != 7}
            ks = sorted(comb)
            mm = np.all([np.isfinite(comb[k]) for k in ks], axis=0)
            Cm = np.corrcoef(np.array([comb[k][mm][::7] for k in ks])) if len(ks) >= 2 else None
            lin = {'par': [a, b], 'lambdas_excluidos': sorted(excl), 'Imed': Imed.tolist(), 'por_lambda': linhas,
                   'p': p_exp, 'C0': c0, 'C0_ok': bool(c0_ok), 'ladrilhos': {str(k): v for k, v in res_t.items()},
                   'sigma_obs_dif_a_G': s_obs, 'sigma_pred_ruido_G': s_pred, 'sigma_pred_teorico_G': s_teo2,
                   'Q': Q, 'R_hp': s_obs / s_teo2, 'razao_menos_mais_nucleo': r_nuc, 'N_filtro_nulo': Nn,
                   'classificacao': cls,
                   'correlacao_ruido_entre_pares': {'pares': ks, 'r': None if Cm is None else Cm.tolist()},
                   'a50_ruido_G': {str(k): 1.96 * v['s_ruido'] for k, v in res_t.items()}}
            out['pares'].append(lin)
            print(f"  par t{a}–t{b} (λ excluídos {sorted(excl) or '—'}):")
            for x in linhas:
                rm = f"{x['razao_menos_mais']:.2f}" if x['razao_menos_mais'] else '—'
                print(f"     λ {x['par_lambda'][0]:2d}/{x['par_lambda'][1]:2d}  Ĩ {x['I_rel']:.2f}  σ₊ = {x['sigma_mais']:.2e}  "
                      f"σ₋/σ₊ = {rm}")
            print(f"     lei σ₊ ∝ Ĩ^p: p = {p_exp:.2f};   C0 = σ₊,0/(ε₈₃ Ĩ₀) = {c0:.2f}  {'ok' if c0_ok else 'FORA'}")
            for t in (a, b):
                r = res_t[t]
                print(f"     t{t}: s teórico {r['s_teorico']:.1f} G → s só ruído {r['s_ruido']:.1f} G  "
                      f"(a₅₀ ≈ {1.96 * r['s_ruido']:.0f} G){'  [λ faltante interpolado]' if r.get('s_ruido_com_interpolacao') else ''}")
            print(f"     σ(Δâ) observado {s_obs:.1f} G;  previsto pelo ruído {s_pred:.1f} G  →  Q = {Q:.2f};  "
                  f"pelo modelo teórico R = {s_obs / s_teo2:.2f};  filtro nulo N = {Nn:.2f};  σ₋/σ₊ no núcleo "
                  f"{r_nuc if r_nuc is None else round(r_nuc, 2)}  →  {cls}")
    cls = [p['classificacao'] for p in out['pares']]
    if cls:
        maj = max(set(cls), key=cls.count)
        out['decisao'] = maj if cls.count(maj) >= max(1, int(np.ceil(2 * len(cls) / 3))) else 'INDEFINIDO'
    return out


def figura(todos):
    pares = [(r['arquivo'][8:27], p) for r in todos for p in r['pares']]
    if not pares:
        return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    cores = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    for i, (rot, p) in enumerate(pares):
        c = cores[i % len(cores)]
        lab = f"{rot} t{p['par'][0]}–t{p['par'][1]}"
        x = [q['I_rel'] for q in p['por_lambda']]
        axs[0].loglog(x, [q['sigma_mais'] for q in p['por_lambda']], 'o-', color=c, ms=4, label=lab)
        xs = [q['I_rel'] for q in p['por_lambda'] if q['razao_menos_mais']]
        axs[1].plot(xs, [q['razao_menos_mais'] for q in p['por_lambda'] if q['razao_menos_mais']], 'o-', color=c, ms=4)
        axs[2].bar(i - 0.2, p['Q'], width=0.4, color=c, edgecolor='white')
        axs[2].bar(i + 0.2, p['R_hp'], width=0.4, color=c, alpha=0.4, edgecolor='white')
    eps = todos[0]['eps_83']
    xx = np.linspace(0.3, 1.05, 50)
    axs[0].loglog(xx, eps * xx, 'k--', lw=1, label='ε₈₃·Ĩ (p = 1)')
    axs[0].set_xlabel('Ĩ (I / I asas)'); axs[0].set_ylabel('σ₊ (ruído, em I asas)'); axs[0].legend(fontsize=6)
    axs[1].axhline(1, color='0.5', lw=1); axs[1].axhline(1.2, color='0.3', ls='--', lw=1)
    axs[1].set_xlabel('Ĩ (núcleo à esquerda)'); axs[1].set_ylabel('σ₋ / σ₊')
    axs[2].axhspan(0.85, 1.15, color='0.9', zorder=0)
    axs[2].set_xticks(range(len(pares))); axs[2].set_xticklabels([f"t{p['par'][0]}–t{p['par'][1]}" for _, p in pares], fontsize=7)
    axs[2].set_ylabel('Q (cheio) e R teórico (claro)')
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel93_ruido_por_lambda.png', dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    rel92 = {r['arquivo']: r for r in json.loads((REP / '92_sst_sobreposicao_ladrilhos.json').read_text(encoding='utf-8'))['data']}
    todos = []
    for c in sorted(a.arquivos, key=lambda x: Path(x).name):
        print('=' * 78)
        print(Path(c).name)
        r = analisar(c, rel92[Path(c).name])
        if r.get('decisao'):
            print(f"  → {r['decisao']}")
        todos.append(r)
    figura(todos)
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '93_sst_ruido_por_lambda.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos}, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório: {REP / '93_sst_ruido_por_lambda.json'}\nFigura: {FIG / 'cel93_ruido_por_lambda.png'}")


if __name__ == '__main__':
    main()
