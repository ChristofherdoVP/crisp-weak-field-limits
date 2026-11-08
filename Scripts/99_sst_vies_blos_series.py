"""
99_sst_vies_blos_series.py — quanto a componente simétrica de V vicia B∥ (filtro casado) nas séries CRISP 2024-10-08?

Scripts 84 e 98: no CRISP 2024 o modelo α(x, y)·I do CRISP2 não foi confirmado como está (09:49: falha
explicada pelo sinal solar, mas o par 3/11 tem ~1,7× α·I também no campo fraco; 10:10: não explicado).
O que importa para os mapas é o viés em â. O filtro casado tem pesos u_k antissimétricos em torno do núcleo,
então uma componente exatamente simétrica não entra em â; o viés vem das assimetrias (perfil, grade com
centro a +25 mÅ, pesos ∝ 1/I²). No CRISP2 (87/88) foi P95 ≈ 2–3 G, acima do limiar.

Mesma cadeia do 88 (máscara, G w = 32 px, α(x, y) de λ0/λ14, perfil médio, t_k = −C g λ₀² dĨ/dλ,
σ_k = ε Ĩ_k, u_k = t_k/σ_k², s = (Σ u_k t_k)^−½), com:
  - ε lido do relatório do script 83 do mesmo arquivo (83_sst_ruido_v_sem_borda_<etiqueta>.json);
  - nenhum quadro excluído (as séries não têm quadros ruins identificados; os candidatos do 80 nas asas
    extremas não foram testados — limitação registrada);
  - dois modelos da componente simétrica, ambos medidos, nenhum validado:
      M1: α(x, y)·I_k em todos os λ                      Δâ₁ = α Σ u_k Ĩ_k / D
      M2: M1 com fator f em λ3 e λ11                     Δâ₂ = α Σ u_k c_k Ĩ_k / D,  c₃ = c₁₁ = f
          f = mediana do b do par 3/11 no campo fraco (relatório do 98 do mesmo arquivo).

Critério (definido antes de rodar), por arquivo e por modelo:
  VIÉS RELEVANTE se P95(|Δâ|) > 0,2 s em ≥ 75% das varreduras (mesmo limiar do 87/88).
  Decisão: nenhum modelo relevante → VIÉS DESPREZÍVEL: mapas de B∥ das séries SEM correção, e o maior
           P95|Δâ| entre os modelos entra como incerteza sistemática;
           algum relevante → CORREÇÃO NECESSÁRIA (modelo …): exige validar o modelo antes de produzir mapas.
Descritivo: K = Σ u_k Ĩ_k / D (G por unidade de α), estrutura suave de â (σ de G₃₂[â]) antes e depois de cada
correção. Controle: nos mosaicos CRISP2, as posições sem quadros ruins (08:01 t2, t3; 10:48 t0, t1, t3) devem
reproduzir K e P95 de M1 do script 88.
Limitação: deslocamento de λ ao longo do campo ignorado (grade nominal), como no 88.

Uso:  python Scripts/99_sst_vies_blos_series.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
Saída: Outputs/Reports/99_sst_vies_blos_series_<AAAAMMDDThhmmss>.json (+ _resumo.json),
       Outputs/Figures/cel99_vies_blos_<…>.png
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

import _saidas_sst as SA

PREFIXO = '99_sst_vies_blos_series'
K_MAD = 1.4826
W = 32
D_BORDA = 150
C_ZEEMAN = 4.6686e-13          # Å⁻¹ G⁻¹
G_EFF = 1.10
LAMBDA0 = 8542.09              # Å
Z50 = 1.96
LIMIAR = 0.2


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def ler_grade(h, nl):
    """Grade λ (Å) do WCS-TAB (idêntica ao 88)."""
    hd = h[0].header
    eixo = None
    for i in range(1, hd.get('NAXIS', 0) + 1):
        if 'WAVE' in str(hd.get(f'CTYPE{i}', '')).upper():
            eixo = i
    if eixo is None:
        return None
    unidade = str(hd.get(f'CUNIT{eixo}', 'nm')).strip().lower()
    fator = {'nm': 10.0, 'm': 1e10, 'angstrom': 1.0, 'a': 1.0}.get(unidade, None)
    if 'TAB' in str(hd.get(f'CTYPE{eixo}', '')).upper():
        ext, col, k = hd.get(f'PS{eixo}_0'), hd.get(f'PS{eixo}_1'), hd.get(f'PV{eixo}_3', 1)
        try:
            arr = np.asarray(h[ext].data[col][0], dtype=np.float64)
        except Exception:
            return None
        ncoord = len(str(col).split('+'))
        if arr.shape[-1] == ncoord and ncoord > 1:
            arr = arr[..., int(k) - 1]
        vals = np.unique(np.round(arr.ravel(), 9))
    else:
        crval, cdelt, crpix = hd.get(f'CRVAL{eixo}'), hd.get(f'CDELT{eixo}'), hd.get(f'CRPIX{eixo}', 1)
        vals = crval + (np.arange(nl) + 1 - crpix) * cdelt
    if fator is None or len(vals) != nl:
        return None
    lam = np.sort(vals) * fator
    if not (8540 < np.median(lam) < 8545 and 0.8 < lam[-1] - lam[0] < 3.0):
        return None
    return lam


def ler_relatorio(prefixo, tag):
    p = SA.REP / f'{prefixo}_{tag}.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))['data'][0]


def analisar(caminho):
    nome = Path(caminho).name
    tag = SA.etiqueta(nome)
    r83 = ler_relatorio('83_sst_ruido_v_sem_borda', tag)
    if r83 is None or not r83.get('eps_V'):
        raise SystemExit(f'{nome}: sem ε válido do script 83 (83_sst_ruido_v_sem_borda_{tag}.json ausente ou REPROVADO).')
    eps = float(r83['eps_V']['mediano'])
    r98 = ler_relatorio('98_sst_diagnostico_simetrica', tag)
    if r98 is None:
        raise SystemExit(f'{nome}: sem relatório do script 98 (98_sst_diagnostico_simetrica_{tag}.json). Rode o 98 antes.')
    bs = [v['fraca']['pares'][2]['b'] for v in r98['varreduras']]
    assert all(v['fraca']['pares'][2]['par'] == [3, 11] for v in r98['varreduras'])
    f311 = float(np.median(bs))
    out = {'arquivo': nome, 'eps_V': eps, 'f_par_3_11': f311,
           'f_par_3_11_p25_p75': [float(np.percentile(bs, 25)), float(np.percentile(bs, 75))], 'varreduras': []}
    print(f"  ε (script 83) = {eps:.2e};  f do par 3/11 no campo fraco (script 98) = {f311:.2f} "
          f"(quartis {out['f_par_3_11_p25_p75'][0]:.2f}–{out['f_par_3_11_p25_p75'][1]:.2f})")
    mapas = []
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt, ns, nl = d.shape[:3]
        lam = ler_grade(h, nl)
        if lam is None:
            raise SystemExit(f'{nome}: não consegui ler a grade λ do WCS-TAB.')
        out['lambda_A'] = lam.tolist()
        print(f"  grade λ (mÅ em relação a {LAMBDA0} Å): {np.round((lam - LAMBDA0) * 1e3).astype(int).tolist()}")
        ck2 = np.ones(nl)
        ck2[3] = ck2[11] = f311
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
            Imed = np.array([np.mean(le(0, l)[usa]) for l in range(nl)]) / Iw_med
            dIdl = np.gradient(Imed, lam)
            tk = -C_ZEEMAN * G_EFF * LAMBDA0 ** 2 * dIdl
            sk = eps * Imed
            uk = tk / sk ** 2
            D = float(np.sum(uk * tk))
            s = 1.0 / np.sqrt(D)
            K1 = float(np.sum(uk * Imed) / D)
            K2 = float(np.sum(uk * ck2 * Imed) / D)
            num = np.zeros_like(I0)
            nI1 = np.zeros_like(I0)
            nI2 = np.zeros_like(I0)
            for l in range(nl):
                Il = le(0, l) / Iw_med
                num += uk[l] * le(3, l) / Iw_med
                nI1 += uk[l] * Il
                nI2 += uk[l] * ck2[l] * Il
            a = num / D
            sub = np.zeros_like(usa)
            sub[::4, ::4] = True
            sel = usa & sub
            v = {'t': t, 's_G': s, 'a50_G': Z50 * s, 'K1_G_por_alfa': K1, 'K2_G_por_alfa': K2,
                 'estrutura_suave_antes_G': sigma_rob(G(a)[sel])}
            for nomeM, nI in (('M1', nI1), ('M2', nI2)):
                delta = np.where(usa, alfa * nI / D, np.nan)
                dv = np.abs(delta[usa])
                p95 = float(np.percentile(dv, 95))
                v[nomeM] = {'P95_abs_delta_G': p95, 'razao_P95_s': p95 / s,
                            'delta_mediana_G': float(np.median(delta[usa])),
                            'delta_p05_p95_G': [float(np.percentile(delta[usa], 5)),
                                                float(np.percentile(delta[usa], 95))],
                            'estrutura_suave_depois_G': sigma_rob(G(a - np.nan_to_num(delta))[sel]),
                            'acima': bool(p95 / s > LIMIAR)}
                if nomeM == 'M1':
                    mapas.append(delta[::8, ::8])
            out['varreduras'].append(v)
            print(f"  t={t}: s = {s:.1f} G (a₅₀ ≈ {Z50 * s:.1f} G)   K₁ = {K1:+.0f}  K₂ = {K2:+.0f} G/α   "
                  f"estrutura suave de â {v['estrutura_suave_antes_G']:.2f} G")
            for nomeM in ('M1', 'M2'):
                q = v[nomeM]
                print(f"        {nomeM}: Δâ 5–95% {q['delta_p05_p95_G'][0]:+.2f} a {q['delta_p05_p95_G'][1]:+.2f} G;  "
                      f"P95|Δâ| = {q['P95_abs_delta_G']:.2f} G = {q['razao_P95_s']:.3f} s "
                      f"→ {'ACIMA' if q['acima'] else 'abaixo'} de {LIMIAR} s;  estrutura após correção "
                      f"{q['estrutura_suave_depois_G']:.2f} G")
    n = len(out['varreduras'])
    rel = []
    for nomeM in ('M1', 'M2'):
        na = sum(v[nomeM]['acima'] for v in out['varreduras'])
        out[f'n_acima_{nomeM}'] = na
        out[f'P95_max_G_{nomeM}'] = max(v[nomeM]['P95_abs_delta_G'] for v in out['varreduras'])
        if SA.passa_fracao(na, n):
            rel.append(nomeM)
    out['modelos_relevantes'] = rel
    out['n_posicoes_ok'] = max(out['n_acima_M1'], out['n_acima_M2'])
    if rel:
        out['decisao'] = f"CORREÇÃO NECESSÁRIA (modelo {', '.join(rel)})"
    else:
        out['decisao'] = 'VIÉS DESPREZÍVEL: mapas sem correção'
        out['incerteza_sistematica_G'] = max(out['P95_max_G_M1'], out['P95_max_G_M2'])
    return out, mapas


def figura(r, mapas, caminho):
    vs = r['varreduras']
    n = len(vs)
    idx = list(range(n)) if n <= 4 else [int(round(x)) for x in np.linspace(0, n - 1, 4)]
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.4), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.4]})
    vm = np.nanmax([np.nanpercentile(np.abs(mp), 99) for mp in mapas])
    for j, t in enumerate(idx):
        im = axs[j].imshow(mapas[t], origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
        axs[j].set_title(f"{r['arquivo'][8:27]}  t={t}  (M1)", fontsize=8)
        axs[j].set_xticks([]); axs[j].set_yticks([])
    fig.colorbar(im, ax=axs[3], shrink=0.8, label='Δâ = B∥·f falso (G)')
    x = np.arange(n)
    axs[4].bar(x - 0.2, [v['M1']['razao_P95_s'] for v in vs], width=0.4, color='#0072B2', label='M1: α·I')
    axs[4].bar(x + 0.2, [v['M2']['razao_P95_s'] for v in vs], width=0.4, color='#D55E00',
               label=f"M2: f = {r['f_par_3_11']:.2f} em λ3, λ11")
    axs[4].axhline(LIMIAR, color='0.3', ls='--', lw=1)
    axs[4].set_xlabel('varredura (t)'); axs[4].set_ylabel('P95|Δâ| / s')
    axs[4].legend(fontsize=7)
    axs[4].set_title(r['decisao'], fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    for c in a.arquivos:
        print('=' * 78)
        print(Path(c).name)
        r, mapas = analisar(c)
        n = len(r['varreduras'])
        print(f"  varreduras com P95|Δâ| > {LIMIAR} s:  M1 {r['n_acima_M1']}/{n}   M2 {r['n_acima_M2']}/{n}"
              f"  →  {r['decisao']}")
        if 'incerteza_sistematica_G' in r:
            print(f"  incerteza sistemática (maior P95|Δâ|): {r['incerteza_sistematica_G']:.2f} G")
        rel = SA.gravar(PREFIXO, r)                          # relatório antes da figura
        cf = SA.figura('cel99_vies_blos', r['etiqueta'])
        figura(r, mapas, cf)
        print(f"  Relatório: {rel}\n  Figura: {cf}")
    SA.resumo(PREFIXO, campos_fora=('varreduras', 'lambda_A'))


if __name__ == '__main__':
    main()
