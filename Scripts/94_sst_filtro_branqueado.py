"""
94_sst_filtro_branqueado.py — filtro casado BRANQUEADO com a lei de ruído medida (script 93), nos mosaicos CRISP2.

Script 93: o ruído de V segue σ_k = ε Ĩ_k^p com p ≈ 0,5–0,8, e não ∝ I (p = 1) como assumido nos pesos dos
mapas do script 89 (u_k = t_k/(ε Ĩ_k)²). Os pesos ótimos para o ruído real (artigo B, Eq. 4) são
    u_k = t_k / σ_k²,   σ_k = ε Ĩ_k^p
(p de cada horário = mediana dos pares válidos do 93; ε do script 83). O estimador continua sem viés
(Σ u_k t_k / D = 1); muda só o desvio:
    s(u) = √(Σ u_k² σ_k²) / Σ u_k t_k     (ruído real)
    s_antigo = s(pesos do 89),  s_branq = s(pesos novos) = (Σ t_k²/σ_k²)^(−1/2) ≤ s_antigo.

Parte A — mapas (todas as correções do 89: borda, α próprio/mestre, exclusão simétrica de quadros):
  novos mapas Outputs/Data/blos_f_branq_crisp2_<data>T<hora>_t<k>.npy; ganho previsto s_branq/s_antigo;
  calibração: inclinação de Deming de â_branq contra â_antigo (mapas do 89), ambos suavizados com 32 px,
  na região de campo fraco (|â_antigo| < 150 G).
Parte B — sobreposições (pares com alinhamento válido no 92, como no 93): ΔV_k com passa-alta, combinado com
  os pesos antigos e os novos (exclusões dos dois ladrilhos, perfil da sobreposição):
  razão medida σ(Δâ_branq)/σ(Δâ_antigo) contra a prevista s_branq/s_antigo.

Critérios (definidos antes de rodar), por horário:
  C1 (ganho real): razão medida ≤ 0,95 em ≥ 2/3 dos pares válidos (mínimo 1);
  C3 (calibração preservada): Deming entre 0,95 e 1,05 em ≥ 3 das 4 posições.
  FILTRO BRANQUEADO ADOTADO se C1 e C3; senão MANTER OS PESOS DO 89.
  Descritivo: |razão medida − prevista| (consistência da lei de ruído); a₅₀ ≈ 1,96 s_branq por pixel.

Uso:  python Scripts/94_sst_filtro_branqueado.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
      (precisa dos mapas/relatórios 89, 92 e 93 e dos mapas mestres de α do 86)
Saída: Outputs/Reports/94_sst_filtro_branqueado.json, Outputs/Figures/cel94_filtro_branqueado.png
"""
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes
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


S89 = carregar('89_sst_mapas_blos.py')
S92 = carregar('92_sst_sobreposicao_ladrilhos.py')
sigma_rob, suave_nan, amostrar = S92.sigma_rob, S92.suave_nan, S92.amostrar
W, D_BORDA, LIM_FRACO = 32, 150, 150.0


def template(Imed, lam):
    return -S89.C_ZEEMAN * S89.G_EFF * S89.LAMBDA0 ** 2 * np.gradient(Imed, lam)


def pesos_e_s(Imed, lam, eps, p, excl):
    """Pesos antigos (σ ∝ I) e branqueados (σ ∝ I^p); s de cada um sob o ruído real σ = ε Ĩ^p."""
    tk = template(Imed, lam)
    usar = np.array([l not in excl for l in range(len(Imed))])
    sig = eps * Imed ** p
    u_old = np.where(usar, tk / (eps * Imed) ** 2, 0.0)
    u_new = np.where(usar, tk / sig ** 2, 0.0)
    s = lambda u: float(np.sqrt(np.sum((u * sig) ** 2)) / np.sum(u * tk))
    return tk, u_old, u_new, s(u_old), s(u_new), float(1 / np.sqrt(np.sum(u_old * tk)))


def deming(x, y):
    sxx, syy = np.var(x), np.var(y)
    sxy = np.cov(x, y, bias=True)[0, 1]
    return float((syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy))


def parte_a(h, nome, eps, p, ruins):
    """Mapas branqueados com as correções do 89; ganho previsto e calibração contra os mapas do 89."""
    hora = nome[19:27]
    rot = f'{nome[8:18].replace("-", "")}T{hora.replace("_", "")}'
    d = h[0].data
    nt, _, nl, ny, nx = d.shape
    lam = S89.ler_grade(h, nl)
    mestre = None
    arq_m = DAT / f'alfa_mestre_crisp2_{rot}.npy'
    if arq_m.exists():
        mm = np.load(arq_m)
        mestre = np.full((ny, nx), np.nan, dtype=np.float32)
        a, b = min(ny, mm.shape[0]), min(nx, mm.shape[1])
        mestre[:a, :b] = mm[:a, :b]
    res, figs = [], []
    for t in range(nt):
        le = lambda s, l: np.asarray(d[t, s, l], dtype=np.float64)
        amostra = np.array([np.asarray(d[t, 0, l, ::8, ::8], dtype=np.float32) for l in range(nl)])
        okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
        lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(nl)]))
        I0, I14, Ic = le(0, 0), le(0, 14), le(0, lnuc)
        V0, V14 = le(3, 0), le(3, 14)
        ok = (np.isfinite(I0) & (I0 != 0) & np.isfinite(I14) & np.isfinite(Ic)
              & np.isfinite(V0) & (V0 != 0) & np.isfinite(V14) & (V14 != 0))
        interior = distance_transform_edt(binary_fill_holes(ok)) >= D_BORDA
        m = ok & interior & (I0 / np.median(I0[ok]) > 0.92) & (I0 / np.median(I0[ok]) < 1.08) \
            & (Ic / np.median(Ic[ok]) < 1.15)
        den = gaussian_filter(m.astype(np.float64), W, mode='constant')
        usa = m & (den > 0.5)
        with np.errstate(invalid='ignore', divide='ignore'):
            Gs = gaussian_filter(np.where(m, (V0 + V14) / 2, 0.0), W, mode='constant') / den
            Gi = gaussian_filter(np.where(m, (I0 + I14) / 2, 0.0), W, mode='constant') / den
        alfa = np.where(interior & ok & (den > 0.5), Gs / Gi, np.nan)
        if mestre is not None:
            um = ~np.isfinite(alfa) & np.isfinite(mestre) & interior & ok
            alfa[um] = mestre[um]
        area = interior & ok & np.isfinite(alfa)
        Iw_med = float(np.mean(((I0 + I14) / 2)[usa]))
        Imed = np.array([np.mean(le(0, l)[usa]) for l in range(nl)]) / Iw_med
        tk, u_old, u_new, s_old, s_new, s_teo89 = pesos_e_s(Imed, lam, eps, p, ruins.get(t, []))
        Dn = float(np.sum(u_new * tk))
        num = np.zeros((ny, nx))
        fin = np.ones((ny, nx), dtype=bool)
        for l in range(nl):
            if u_new[l] == 0:
                continue
            Il, Vl = le(0, l), le(3, l)
            fin &= np.isfinite(Il) & np.isfinite(Vl)
            num += u_new[l] * np.nan_to_num(Vl - np.nan_to_num(alfa) * Il) / Iw_med
        a_new = np.where(area & fin, num / Dn, np.nan).astype(np.float32)
        np.save(DAT / f'blos_f_branq_crisp2_{rot}_t{t}.npy', a_new)
        a_old = np.load(DAT / f'blos_f_crisp2_{rot}_t{t}.npy').astype(np.float64)
        fraco = np.isfinite(a_old) & np.isfinite(a_new) & (np.abs(a_old) < LIM_FRACO)
        so, sn = suave_nan(np.where(fraco, a_old, np.nan), W), suave_nan(np.where(fraco, a_new.astype(float), np.nan), W)
        mm_ = np.isfinite(so) & np.isfinite(sn)
        sub = np.zeros_like(mm_)
        sub[::4, ::4] = True
        dm = deming(so[mm_ & sub], sn[mm_ & sub])
        r = {'t': t, 'quadros_excluidos': ruins.get(t, []), 's_teorico_89_G': s_teo89, 's_antigo_ruido_real_G': s_old,
             's_branqueado_G': s_new, 'ganho_previsto': s_new / s_old, 'a50_branqueado_G': 1.96 * s_new,
             'deming_branq_vs_antigo': dm, 'calibracao_ok': bool(0.95 <= dm <= 1.05),
             'pesos_antigos_norm': (u_old / np.abs(u_old).max()).tolist(),
             'pesos_branq_norm': (u_new / np.abs(u_new).max()).tolist()}
        res.append(r)
        figs.append((a_old[::4, ::4], a_new[::4, ::4], t))
        print(f"  t={t}: s (ruído real) com pesos do 89 {s_old:.1f} G → branqueado {s_new:.1f} G  "
              f"(razão prevista {s_new / s_old:.3f};  a₅₀ ≈ {1.96 * s_new:.0f} G);  Deming branq/antigo "
              f"{dm:.3f}  {'ok' if r['calibracao_ok'] else 'FORA'}")
    return res, figs, lam


def parte_b(h, nome, eps, p, ruins, rel92, lam):
    hora = nome[19:27]
    rot = f'{nome[8:18].replace("-", "")}T{hora.replace("_", "")}'
    d = h[0].data
    nt, _, nl, ny, nx = d.shape
    afins, _ = S92.afins_wcs(h)
    As = {t: np.load(DAT / f'blos_f_crisp2_{rot}_t{t}.npy').astype(np.float64) for t in range(nt)}
    res = []
    for pr in [x for x in rel92['pares'] if x.get('sobreposicao') and x['alinhamento_valido']]:
        a, b = pr['par']
        dy, dx = pr['desloc_refino_px']
        T = S92.a_para_b(afins[a], afins[b])
        Tr = T.copy()
        Tr[:, 2] = T[:, 2] - T[:, :2] @ np.array([dx, dy], dtype=float)
        ys, xs = np.arange(0, ny, 8), np.arange(0, nx, 8)
        cob = np.isfinite(amostrar(As[b], Tr, ys, xs, 0)) & np.isfinite(As[a][::8, ::8][:len(ys), :len(xs)])
        yy, xx = np.where(cob)
        y0, y1 = max(0, ys[yy.min()] - 16), min(ny, ys[yy.max()] + 16)
        x0, x1 = max(0, xs[xx.min()] - 16), min(nx, xs[xx.max()] + 16)
        ysF, xsF = np.arange(y0, y1), np.arange(x0, x1)
        Aa, Ab = As[a][y0:y1, x0:x1], amostrar(As[b], Tr, ysF, xsF, 0)
        fraco = np.isfinite(Aa) & np.isfinite(Ab) & (np.abs(Aa) < LIM_FRACO) & (np.abs(Ab) < LIM_FRACO)
        Ia = [np.asarray(d[a, 0, l, y0:y1, x0:x1], dtype=np.float64) for l in range(nl)]
        Iw = float(np.mean(((Ia[0] + Ia[14]) / 2)[fraco]))
        Imed = np.array([np.mean(I[fraco]) for I in Ia]) / Iw
        excl = sorted(set(ruins.get(a, [])) | set(ruins.get(b, [])))
        tk, u_old, u_new, s_old, s_new, _ = pesos_e_s(Imed, lam, eps, p, excl)
        acc_o, acc_n = np.zeros(Aa.shape), np.zeros(Aa.shape)
        fin = fraco.copy()
        for l in range(nl):
            if u_old[l] == 0:
                continue
            Va = np.asarray(d[a, 3, l, y0:y1, x0:x1], dtype=np.float64)
            Vb = amostrar(np.asarray(d[b, 3, l], dtype=np.float64), Tr, ysF, xsF, 0)
            dV = (Va - Vb) / Iw
            dV = dV - suave_nan(np.where(fraco, dV, np.nan), W)
            fin &= np.isfinite(dV)
            acc_o += u_old[l] * np.nan_to_num(dV)
            acc_n += u_new[l] * np.nan_to_num(dV)
        da_o = acc_o / np.sum(u_old * tk)
        da_n = acc_n / np.sum(u_new * tk)
        r_med = sigma_rob(da_n[fin]) / sigma_rob(da_o[fin])
        r_prev = s_new / s_old
        lin = {'par': [a, b], 'lambdas_excluidos': excl, 'sigma_dif_antigo_G': sigma_rob(da_o[fin]),
               'sigma_dif_branq_G': sigma_rob(da_n[fin]), 'razao_medida': r_med, 'razao_prevista': r_prev,
               'ganho_ok': bool(r_med <= 0.95)}
        res.append(lin)
        print(f"  par t{a}–t{b}: σ(Δâ) pesos do 89 {lin['sigma_dif_antigo_G']:.1f} G → branqueado "
              f"{lin['sigma_dif_branq_G']:.1f} G;  razão medida {r_med:.3f}, prevista {r_prev:.3f}  "
              f"{'ganho ok' if lin['ganho_ok'] else 'SEM ganho ≥ 5%'}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    rel92 = {r['arquivo']: r for r in json.loads((REP / '92_sst_sobreposicao_ladrilhos.json').read_text(encoding='utf-8'))['data']}
    rel93 = {r['arquivo']: r for r in json.loads((REP / '93_sst_ruido_por_lambda.json').read_text(encoding='utf-8'))['data']}
    todos, todos_figs = [], []
    for c in sorted(a.arquivos, key=lambda x: Path(x).name):
        nome = Path(c).name
        hora = nome[19:27]
        eps, ruins = S89.EPS[hora], S89.RUINS.get(hora, {})
        ps = [x['p'] for x in rel93[nome]['pares'] if x.get('p') is not None]
        p = float(np.median(ps))
        print('=' * 78)
        print(f'{nome}\n  lei de ruído: σ = ε Ĩ^p com ε = {eps:.2e} (script 83) e p = {p:.2f} '
              f'(mediana de {len(ps)} par(es) do script 93)')
        with fits.open(c, memmap=True) as h:
            ra, figs, lam = parte_a(h, nome, eps, p, ruins)
            rb = parte_b(h, nome, eps, p, ruins, rel92[nome], lam)
        n_val = len(rb)
        n_g = sum(x['ganho_ok'] for x in rb)
        c1 = n_val > 0 and n_g >= max(1, int(np.ceil(2 * n_val / 3)))
        n_cal = sum(x['calibracao_ok'] for x in ra)
        c3 = n_cal >= 3
        dec = 'FILTRO BRANQUEADO ADOTADO' if (c1 and c3) else 'MANTER OS PESOS DO 89'
        print(f"  C1 ganho ≥ 5% nas sobreposições: {n_g} de {n_val}  {'ok' if c1 else 'FALHA'};   "
              f"C3 calibração 0,95–1,05: {n_cal} de {len(ra)}  {'ok' if c3 else 'FALHA'}  →  {dec}")
        todos.append({'arquivo': nome, 'eps': eps, 'p': p, 'posicoes': ra, 'pares': rb,
                      'C1': bool(c1), 'C3': bool(c3), 'decisao': dec})
        todos_figs.append((nome[8:27], figs, ra, rb))

    nl_ = len(todos_figs)
    fig, axs = plt.subplots(nl_, 3, figsize=(15, 4.4 * nl_), squeeze=False)
    for k, (rot, figs, ra, rb) in enumerate(todos_figs):
        lamk = np.arange(len(ra[0]['pesos_antigos_norm']))
        r0 = next((x for x in ra if not x['quadros_excluidos']), ra[0])
        axs[k, 0].plot(lamk, r0['pesos_antigos_norm'], 'o-', color='#999999', label='pesos do 89 (σ ∝ I)')
        axs[k, 0].plot(lamk, r0['pesos_branq_norm'], 'o-', color='#0072B2', label=f'branqueados (σ ∝ I^p)')
        axs[k, 0].axhline(0, color='0.7', lw=1)
        axs[k, 0].set_xlabel('índice de λ'); axs[k, 0].set_ylabel('u_k (normalizado)')
        axs[k, 0].set_title(f"{rot}  t={r0['t']}", fontsize=8); axs[k, 0].legend(fontsize=7)
        x = np.arange(len(ra))
        axs[k, 1].bar(x - 0.2, [q['s_antigo_ruido_real_G'] for q in ra], width=0.4, color='#999999', label='pesos do 89')
        axs[k, 1].bar(x + 0.2, [q['s_branqueado_G'] for q in ra], width=0.4, color='#0072B2', label='branqueado')
        axs[k, 1].set_xticks(x); axs[k, 1].set_xlabel('posição (t)'); axs[k, 1].set_ylabel('s por pixel (G)')
        axs[k, 1].legend(fontsize=7)
        if rb:
            xb = np.arange(len(rb))
            axs[k, 2].bar(xb - 0.2, [q['razao_prevista'] for q in rb], width=0.4, color='#E69F00', label='prevista')
            axs[k, 2].bar(xb + 0.2, [q['razao_medida'] for q in rb], width=0.4, color='#0072B2', label='medida')
            axs[k, 2].axhline(0.95, color='0.3', ls='--', lw=1)
            axs[k, 2].set_xticks(xb); axs[k, 2].set_xticklabels([f"t{q['par'][0]}–t{q['par'][1]}" for q in rb])
            axs[k, 2].set_ylabel('σ(Δâ) branqueado / pesos do 89'); axs[k, 2].legend(fontsize=7)
            axs[k, 2].set_ylim(0.5, 1.1)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel94_filtro_branqueado.png', dpi=110)
    plt.close(fig)
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '94_sst_filtro_branqueado.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos}, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório: {REP / '94_sst_filtro_branqueado.json'}\nFigura: {FIG / 'cel94_filtro_branqueado.png'}\n"
          f"Mapas: {DAT / 'blos_f_branq_crisp2_<data>T<hora>_t<k>.npy'}")


if __name__ == '__main__':
    main()
