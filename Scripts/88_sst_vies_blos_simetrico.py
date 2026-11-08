"""
88_sst_vies_blos_simetrico.py — viés de α(x, y) em B∥ pelo filtro casado, com exclusão SIMÉTRICA dos quadros ruins.

Script 87 (CORREÇÃO NECESSÁRIA, 4/4 nos dois horários): nas posições sem exclusões, K ≈ −270 a −350 G por unidade
de α e P95|Δâ| ≈ 2,2–2,6 G. Nas posições em que um quadro ruim foi excluído de um lado só da linha, K saltou
para −1300 a −3400 G e, em 08:01 t0 e 10:48 t2, a correção AUMENTOU a estrutura suave de â: a exclusão
assimétrica deixa a correção dependente de α·I em poucos λ perto do núcleo, onde o modelo é mais fraco (84).
A grade λ é simétrica em torno de +19 mÅ (índice k ↔ 14 − k). Aqui cada quadro ruim sai junto com o espelho.

Idêntico ao 87, exceto:
  - quadros excluídos: o ruim e o espelho (08:01 t0: 2, 4, 10, 12; 08:01 t1: 6, 8; 10:48 t2: 4, 6, 8, 10);
  - validação nova da correção.

Critérios (definidos antes de rodar), por arquivo:
  Viés: CORREÇÃO NECESSÁRIA se P95(|Δâ|) > 0,2 s em ≥ 3 das 4 posições; senão RECOMENDÁVEL.
  Validação: a correção V − α(x, y)·I_λ só é ACEITA se a estrutura suave de â (σ robusto de G_32[â])
             DIMINUI em ≥ 3 das 4 posições; senão NÃO ACEITA (não aplicar).
Limitação: o deslocamento de λ ao longo do campo (etalon inclinado) é ignorado; usa-se a grade nominal.

Uso:  python Scripts/88_sst_vies_blos_simetrico.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
Saída: Outputs/Reports/88_sst_vies_blos_simetrico.json, Outputs/Figures/cel88_vies_blos_simetrico.png
"""
import json
import argparse
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
K_MAD = 1.4826
W = 32
D_BORDA = 150
C_ZEEMAN = 4.6686e-13          # Å⁻¹ G⁻¹
G_EFF = 1.10
LAMBDA0 = 8542.09              # Å
Z50 = 1.96

# ε_V do script 83 (asas, interior) e quadros ruins (scripts 79 e 84), por horário do arquivo
EPS = {'08_01_31': 2.36e-3, '10_48_36': 2.20e-3}
RUINS_ORIG = {'08_01_31': {0: [2, 4], 1: [6]}, '10_48_36': {2: [4, 6]}}
RUINS = {h: {t: sorted(set(v) | {14 - k for k in v}) for t, v in d.items()} for h, d in RUINS_ORIG.items()}


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def ler_grade(h, nl):
    """Grade λ (Å) a partir do WCS-TAB do cabeçalho SOLARNET. Devolve (λ, diagnóstico)."""
    hd = h[0].header
    eixo = None
    for i in range(1, hd.get('NAXIS', 0) + 1):
        if 'WAVE' in str(hd.get(f'CTYPE{i}', '')).upper():
            eixo = i
    diag = {k: str(v) for k, v in hd.items() if any(k.startswith(p) for p in ('CTYPE', 'CUNIT', 'PS', 'PV', 'CRVAL', 'CDELT'))}
    for hdu in h[1:]:
        if getattr(hdu, 'columns', None) is not None:
            diag[f'tabela {hdu.name}'] = str([(c.name, c.format, c.dim) for c in hdu.columns])
    if eixo is None:
        return None, diag
    unidade = str(hd.get(f'CUNIT{eixo}', 'nm')).strip().lower()
    fator = {'nm': 10.0, 'm': 1e10, 'angstrom': 1.0, 'a': 1.0}.get(unidade, None)
    if 'TAB' in str(hd.get(f'CTYPE{eixo}', '')).upper():
        ext, col, k = hd.get(f'PS{eixo}_0'), hd.get(f'PS{eixo}_1'), hd.get(f'PV{eixo}_3', 1)
        try:
            arr = np.asarray(h[ext].data[col][0], dtype=np.float64)
        except Exception:
            return None, diag
        ncoord = len(str(col).split('+'))
        if arr.shape[-1] == ncoord and ncoord > 1:
            arr = arr[..., int(k) - 1]
        vals = np.unique(np.round(arr.ravel(), 9))
    else:
        crval, cdelt, crpix = hd.get(f'CRVAL{eixo}'), hd.get(f'CDELT{eixo}'), hd.get(f'CRPIX{eixo}', 1)
        vals = crval + (np.arange(nl) + 1 - crpix) * cdelt
    if fator is None or len(vals) != nl:
        diag['valores_lidos'] = str(vals[:40])
        return None, diag
    lam = np.sort(vals) * fator
    if not (8540 < np.median(lam) < 8545 and 0.8 < lam[-1] - lam[0] < 3.0):
        diag['valores_lidos_A'] = str(lam)
        return None, diag
    return lam, diag


def analisar(caminho):
    nome = Path(caminho).name
    hora = nome[19:27]
    eps = EPS.get(hora)
    ruins = RUINS.get(hora, {})
    out = {'arquivo': nome, 'eps_V': eps, 'posicoes': []}
    mapas = []
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt, ns, nl = d.shape[:3]
        lam, diag = ler_grade(h, nl)
        if lam is None:
            print('  NÃO consegui ler a grade λ do cabeçalho. Chaves encontradas:')
            for k, v in diag.items():
                print(f'     {k} = {v}')
            raise SystemExit('Mande estas linhas para conferência; o script parou antes de qualquer cálculo.')
        out['lambda_A'] = lam.tolist()
        print(f"  grade λ (mÅ em relação a {LAMBDA0} Å): {np.round((lam - LAMBDA0) * 1e3).astype(int).tolist()}")
        if eps is None:
            raise SystemExit(f'Horário {hora} sem ε medido no script 83.')
        for t in range(nt):
            le = lambda s, l: np.asarray(d[t, s, l], dtype=np.float64)
            I0, I14 = le(0, 0), le(0, 14)
            amostra = np.array([np.asarray(d[t, 0, l, ::8, ::8], dtype=np.float32) for l in range(nl)])
            okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
            lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(nl)]))
            Ic = le(0, lnuc)
            V0, V14 = le(3, 0), le(3, 14)
            ok = (np.isfinite(I0) & (I0 != 0) & np.isfinite(I14) & np.isfinite(Ic)
                  & np.isfinite(V0) & (V0 != 0) & np.isfinite(V14) & (V14 != 0))
            interior = distance_transform_edt(binary_fill_holes(ok)) >= D_BORDA
            asa = I0 / np.median(I0[ok])
            nuc = Ic / np.median(Ic[ok])
            m = ok & interior & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)
            den = gaussian_filter(m.astype(np.float64), W, mode='constant')
            usa = m & (den > 0.5)

            def G(X):
                with np.errstate(invalid='ignore', divide='ignore'):
                    return gaussian_filter(np.where(m, X, 0.0), W, mode='constant') / den

            Iw = (I0 + I14) / 2
            Iw_med = float(np.mean(Iw[usa]))
            alfa = G((V0 + V14) / 2) / G(Iw)
            # perfil médio e template
            usar = np.array([l not in ruins.get(t, []) for l in range(nl)])
            Imed = np.array([np.mean(le(0, l)[usa]) for l in range(nl)]) / Iw_med
            dIdl = np.gradient(Imed, lam)
            tk = -C_ZEEMAN * G_EFF * LAMBDA0 ** 2 * dIdl          # V/I_asas por G
            sk = eps * Imed
            uk = np.where(usar, tk / sk ** 2, 0.0)
            D = float(np.sum(uk * tk))
            s = 1.0 / np.sqrt(D)
            K = float(np.sum(uk * Imed) / D)
            # â antes e Δâ, pixel a pixel (acumulando um λ por vez)
            num = np.zeros_like(I0)
            numI = np.zeros_like(I0)
            for l in range(nl):
                if uk[l] == 0:
                    continue
                num += uk[l] * le(3, l) / Iw_med
                numI += uk[l] * le(0, l) / Iw_med
            a_antes = num / D
            delta = np.where(usa, alfa * numI / D, np.nan)
            a_depois = a_antes - np.nan_to_num(delta)
            dv = np.abs(delta[usa])
            p95 = float(np.percentile(dv, 95))
            sm_antes, sm_depois = G(a_antes), G(a_depois)
            sub = np.zeros_like(usa)
            sub[::4, ::4] = True
            sel = usa & sub
            pos = {'t': t, 'quadros_excluidos': ruins.get(t, []), 's_G': s, 'a50_G': Z50 * s, 'K_G_por_alfa': K,
                   'P95_abs_delta_G': p95, 'razao_P95_s': p95 / s,
                   'delta_mediana_G': float(np.median(delta[usa])),
                   'delta_p05_p95_G': [float(np.percentile(delta[usa], 5)), float(np.percentile(delta[usa], 95))],
                   'a_mediana_antes_G': float(np.median(a_antes[usa])), 'a_mediana_depois_G': float(np.median(a_depois[usa])),
                   'sigma_a_antes_G': sigma_rob(a_antes[usa]), 'sigma_a_depois_G': sigma_rob(a_depois[usa]),
                   'estrutura_suave_antes_G': sigma_rob(sm_antes[sel]),
                   'estrutura_suave_depois_G': sigma_rob(sm_depois[sel]),
                   'passa_limiar': bool(p95 / s > 0.2)}
            pos['estrutura_diminui'] = bool(pos['estrutura_suave_depois_G'] < pos['estrutura_suave_antes_G'])
            out['posicoes'].append(pos)
            mapas.append(np.where(usa, delta, np.nan)[::8, ::8])
            print(f"  t={t}: s = {s:.1f} G  (a₅₀ ≈ {Z50 * s:.1f} G)   K = {K:+.0f} G por unidade de α   "
                  f"quadros excluídos {ruins.get(t, []) or '—'}")
            print(f"        Δâ: mediana {pos['delta_mediana_G']:+.2f} G, 5–95% {pos['delta_p05_p95_G'][0]:+.2f} a "
                  f"{pos['delta_p05_p95_G'][1]:+.2f} G;  P95|Δâ| = {p95:.2f} G = {p95 / s:.3f} s  "
                  f"→ {'acima' if p95 / s > 0.2 else 'abaixo'} de 0,2 s")
            print(f"        â mediano {pos['a_mediana_antes_G']:+.1f} → {pos['a_mediana_depois_G']:+.1f} G;  σ(â) por pixel {pos['sigma_a_antes_G']:.1f} → {pos['sigma_a_depois_G']:.1f} G;   "
                  f"estrutura suave de â (32 px) {pos['estrutura_suave_antes_G']:.2f} → {pos['estrutura_suave_depois_G']:.2f} G  "
                  f"({'diminui' if pos['estrutura_diminui'] else 'NÃO diminui'})")
    n = sum(p['passa_limiar'] for p in out['posicoes'])
    out['n_posicoes_acima'] = n
    out['decisao'] = 'CORREÇÃO NECESSÁRIA' if n >= 3 else 'CORREÇÃO RECOMENDÁVEL'
    nv = sum(p['estrutura_diminui'] for p in out['posicoes'])
    out['n_posicoes_estrutura_diminui'] = nv
    out['validacao'] = 'CORREÇÃO ACEITA' if nv >= 3 else 'CORREÇÃO NÃO ACEITA'
    return out, mapas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    todos, todos_mapas = [], []
    for c in sorted(a.arquivos, key=lambda x: Path(x).name):
        print('=' * 78)
        print(Path(c).name)
        r, mapas = analisar(c)
        todos.append(r)
        todos_mapas.append(mapas)
        print(f"  posições com P95|Δâ| > 0,2 s: {r['n_posicoes_acima']} de {len(r['posicoes'])}  →  {r['decisao']}")
        print(f"  posições em que a estrutura suave de â diminui: {r['n_posicoes_estrutura_diminui']} de "
              f"{len(r['posicoes'])}  →  {r['validacao']}")

    nlin = len(todos)
    fig, axs = plt.subplots(nlin, 5, figsize=(16, 3.3 * nlin), squeeze=False,
                            gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.3]})
    vm = np.nanmax([np.nanpercentile(np.abs(mp), 99) for ms in todos_mapas for mp in ms])
    for k, (r, ms) in enumerate(zip(todos, todos_mapas)):
        for t, mp in enumerate(ms):
            im = axs[k, t].imshow(mp, origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
            axs[k, t].set_title(f"{r['arquivo'][8:27]}  t={t}", fontsize=8)
            axs[k, t].set_xticks([]); axs[k, t].set_yticks([])
        fig.colorbar(im, ax=axs[k, 3], shrink=0.8, label='Δâ = B∥ f falso (G)')
        lam = np.array(r['lambda_A'])
        x = (lam - LAMBDA0) * 1e3
        ax = axs[k, 4]
        p = r['posicoes'][0]
        ax.bar(range(len(r['posicoes'])), [q['razao_P95_s'] for q in r['posicoes']], color='#0072B2', edgecolor='white')
        ax.axhline(0.2, color='0.3', ls='--', lw=1)
        ax.set_xticks(range(len(r['posicoes']))); ax.set_xlabel('posição (t)')
        ax.set_ylabel('P95|Δâ| / s'); ax.set_title(f"→ {r['decisao']} / {r['validacao']}", fontsize=7)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel88_vies_blos_simetrico.png', dpi=110)
    plt.close(fig)

    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '88_sst_vies_blos_simetrico.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos}, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório: {REP / '88_sst_vies_blos_simetrico.json'}\nFigura: {FIG / 'cel88_vies_blos_simetrico.png'}")


if __name__ == '__main__':
    main()
