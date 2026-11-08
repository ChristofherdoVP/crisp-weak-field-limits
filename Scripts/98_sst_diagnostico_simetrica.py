"""
98_sst_diagnostico_simetrica.py — por que o modelo α(x, y)·I do script 84 falha nas séries CRISP 2024-10-08?

Script 84 (2026-10-26): CRISP2 (mosaicos 2025-08-19) 4/4 e 4/4 → modelo confirmado; CRISP 2024-10-08
09:49 10/14 (71%) e 10:10 0/10 → NÃO CONFIRMADO pela regra de 75%. Nas 09:49 o par 3/11 falha em todas
as varreduras com b ≈ 1,6–1,9 e r ≈ 0,7; nas 10:10 os pares internos têm r = 0,1–0,5 e o script 80
mostra V₀ e V₁₄ anticorrelacionados (σ₊/σ_ind = 0,82): há sinal solar antissimétrico até nas asas.

Máscara, suavização (G, w = 32 px, normalizada sobre a máscara) e α(x, y) = G[(V₀+V₁₄)/2]/G[(I₀+I₁₄)/2]
idênticos ao 84. Pixels amostrados a cada 8 px.

TESTE A — componente dependente de λ (cada λ separado, k = 1…13):
    G[V_k] = c_k · α·G[I_k] + const   → c_k e r_k
  varredura sinaliza k se c_k ∉ [0,8; 1,2] e r_k ≥ 0,7, com o espelho 14−k em c ∈ [0,8; 1,2].
  Série: α DEPENDENTE DE λ em k se k é sinalizado em ≥ 75% das varreduras.
  (Se positivo: modelo α_k = c_k·α(x, y), correção a validar em passo seguinte.)

TESTE B — vazamento do sinal solar:
  indicador de campo F = |G₁₆[(V₉ − V₅)/2]| / ⟨I⟩ (antissimétrico perto do núcleo; G₁₆ normalizada na máscara);
  máscara de campo fraco = pixels da máscara do 84 com F ≤ percentil 33 de F;
  G, α e o teste de pares do 84 refeitos só nessa máscara (mesmos limiares: 0,8 ≤ b ≤ 1,2 e r ≥ 0,7;
  ≥ 4 de 6 pares e um de 5/9, 6/8; pixels com peso da máscara no núcleo gaussiano > 0,15, porque a máscara
  tem ~1/3 dos pixels — no 84 o limiar 0,5 valia para a máscara inteira).
  Série: FALHA DO 84 EXPLICADA PELO SINAL SOLAR se o 84 falha na máscara completa (< 75%) e passa
  (≥ 75% das varreduras) na de campo fraco. Se já passa na completa: NÃO SE APLICA.
  (Selecionar por (V₉ − V₅) não enviesa o par 5/9 simétrico (V₅ + V₉): ruídos de S e D independentes.)

Decisão por arquivo: lista dos testes positivos; nenhum → NÃO EXPLICADO (componente simétrica sem correção,
vira incerteza sistemática; viés em B∥ estimado à parte, como no 87).
Controle: mosaicos CRISP2 — esperado c_k ≈ 1 em todos os λ (k sinalizado só em varreduras isoladas).
Descritivo: teste A também na máscara de campo fraco.

Uso:  python Scripts/98_sst_diagnostico_simetrica.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
Saída: Outputs/Reports/98_sst_diagnostico_simetrica_<AAAAMMDDThhmmss>.json (+ _resumo.json),
       Outputs/Figures/cel98_ck_<…>.png e cel98_mascara_fraca_<…>.png
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saidas_sst as SA

PREFIXO = '98_sst_diagnostico_simetrica'
W = 32
W_F = 16
D_BORDA = 150
AMOSTRA = 8
PERC_FRACO = 33
DEN_FRACO = 0.15
PARES = [(k, 14 - k) for k in range(1, 7)]
LAMS = list(range(1, 14))


def ajuste(P, M):
    b, c = np.polyfit(P, M, 1)
    return float(b), float(c), float(np.corrcoef(P, M)[0, 1])


def suavizador(m, w):
    den = gaussian_filter(m.astype(np.float64), w, mode='constant')

    def G(X):
        with np.errstate(invalid='ignore', divide='ignore'):
            return gaussian_filter(np.where(m, X, 0.0), w, mode='constant') / den
    return G, den


def testes_na_mascara(le, m, den_min, sub):
    """α, teste A (c_k por λ) e teste de pares do 84 numa máscara m."""
    G, den = suavizador(m, W)
    usa = m & (den > den_min)
    sel = usa & sub
    GI = {k: G(le(0, k)) for k in range(15)}
    GV = {k: G(le(3, k)) for k in range(15)}
    alfa = (GV[0] + GV[14]) / 2 / ((GI[0] + GI[14]) / 2)
    ok = sel & np.isfinite(alfa)
    for k in range(15):
        ok &= np.isfinite(GI[k]) & np.isfinite(GV[k])
    res = {'n_amostra': int(ok.sum()), 'alfa_mediana': float(np.median(alfa[ok])),
           'alfa_p05_p95': [float(np.percentile(alfa[ok], 5)), float(np.percentile(alfa[ok], 95))]}
    ck = {}
    for k in LAMS:
        c, _, r = ajuste((alfa * GI[k])[ok], GV[k][ok])
        ck[k] = {'c': c, 'r': r}
    res['ck'] = ck
    sinal = []
    for k in LAMS:
        e = 14 - k
        if e == k:
            continue
        if (not 0.8 <= ck[k]['c'] <= 1.2) and ck[k]['r'] >= 0.7 and 0.8 <= ck[e]['c'] <= 1.2:
            sinal.append(k)
    res['lambdas_sinalizados'] = sinal
    pares = []
    for (ka, kb) in PARES:
        P = (alfa * (GI[ka] + GI[kb]) / 2)[ok]
        M = ((GV[ka] + GV[kb]) / 2)[ok]
        b, c, r = ajuste(P, M)
        pares.append({'par': [ka, kb], 'b': b, 'r': r,
                      'I_rel_asas': float(np.mean((GI[ka] + GI[kb])[ok]) / np.mean((GI[0] + GI[14])[ok])),
                      'passa': bool(0.8 <= b <= 1.2 and r >= 0.7)})
    res['pares'] = pares
    res['n_pares_ok'] = sum(p['passa'] for p in pares)
    res['nucleo_ok'] = bool(any(p['passa'] for p in pares if p['par'][0] in (5, 6)))
    res['passa_84'] = bool(res['n_pares_ok'] >= 4 and res['nucleo_ok'])
    return res, usa


def analisar(caminho):
    out = {'arquivo': Path(caminho).name, 'varreduras': []}
    mapa = None
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data                                      # (t, s, λ, y, x)
        nt = d.shape[0]
        for t in range(nt):
            le = lambda s, l: np.asarray(d[t, s, l], dtype=np.float64)
            amostra = np.array([np.asarray(d[t, 0, l, ::8, ::8], dtype=np.float32) for l in range(15)])
            okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
            lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(15)]))
            I0, I14, Ic = le(0, 0), le(0, 14), le(0, lnuc)
            V0, V14 = le(3, 0), le(3, 14)
            ok = (np.isfinite(I0) & (I0 != 0) & np.isfinite(I14) & np.isfinite(Ic)
                  & np.isfinite(V0) & (V0 != 0) & np.isfinite(V14) & (V14 != 0))
            interior = distance_transform_edt(binary_fill_holes(ok)) >= D_BORDA
            asa = I0 / np.median(I0[ok])
            nuc = Ic / np.median(Ic[ok])
            m = ok & interior & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)
            for k in range(15):
                m &= np.isfinite(le(0, k)) & np.isfinite(le(3, k))
            sub = np.zeros_like(m)
            sub[::AMOSTRA, ::AMOSTRA] = True
            completa, usa = testes_na_mascara(le, m, 0.5, sub)
            # indicador de campo e máscara de campo fraco
            G16, den16 = suavizador(m, W_F)
            F = np.abs(G16((le(3, 9) - le(3, 5)) / 2)) / float(np.mean(((I0 + I14) / 2)[usa]))
            lim = float(np.percentile(F[usa], PERC_FRACO))
            mf = usa & (F <= lim)
            fraca, usa_f = testes_na_mascara(le, mf, DEN_FRACO, sub)
            v = {'t': t, 'lambda_nucleo_idx': lnuc, 'F_limiar': lim,
                 'F_mediano_completa': float(np.median(F[usa])), 'F_mediano_fraca': float(np.median(F[mf])),
                 'completa': completa, 'fraca': fraca}
            out['varreduras'].append(v)
            if mapa is None or t == nt // 2:
                mapa = (t, np.where(usa, F, np.nan)[::8, ::8], usa_f[::8, ::8])
            ca = ' '.join(f"{completa['ck'][k]['c']:4.2f}" for k in LAMS)
            print(f"  t={t}: c_k (k=1…13) {ca}")
            ra = ' '.join(f"{completa['ck'][k]['r']:4.2f}" for k in LAMS)
            print(f"        r_k           {ra}")
            print(f"        λ sinalizados: {completa['lambdas_sinalizados'] or '—'}   "
                  f"84 na máscara completa: {completa['n_pares_ok']}/6 {'passa' if completa['passa_84'] else 'FALHA'}   "
                  f"campo fraco (F ≤ {lim:.1e}, {fraca['n_amostra']} amostras): {fraca['n_pares_ok']}/6 "
                  f"{'passa' if fraca['passa_84'] else 'FALHA'}")
            print(f"        pares no campo fraco: " + '  '.join(
                f"{p['par'][0]}/{p['par'][1]} b={p['b']:.2f} r={p['r']:.2f}" for p in fraca['pares']))
    n = len(out['varreduras'])
    # teste A
    cont = {k: sum(k in v['completa']['lambdas_sinalizados'] for v in out['varreduras']) for k in LAMS}
    depend = [k for k in LAMS if SA.passa_fracao(cont[k], n)]
    out['testeA'] = {'contagem_sinalizados': cont, 'lambdas_dependentes': depend,
                     'ck_mediano': {k: float(np.median([v['completa']['ck'][k]['c'] for v in out['varreduras']]))
                                    for k in LAMS},
                     'rk_mediano': {k: float(np.median([v['completa']['ck'][k]['r'] for v in out['varreduras']]))
                                    for k in LAMS},
                     'ck_mediano_campo_fraco': {k: float(np.median([v['fraca']['ck'][k]['c'] for v in out['varreduras']]))
                                                for k in LAMS},
                     'decisao': (f'α DEPENDENTE DE λ (k = {depend})' if depend else 'SEM DEPENDÊNCIA DE λ')}
    # teste B
    n_c = sum(v['completa']['passa_84'] for v in out['varreduras'])
    n_f = sum(v['fraca']['passa_84'] for v in out['varreduras'])
    if SA.passa_fracao(n_c, n):
        dB = 'NÃO SE APLICA (84 passa na máscara completa)'
    elif SA.passa_fracao(n_f, n):
        dB = 'FALHA DO 84 EXPLICADA PELO SINAL SOLAR'
    else:
        dB = 'NÃO EXPLICADA PELO SINAL SOLAR'
    out['testeB'] = {'n_passa_completa': n_c, 'n_passa_fraca': n_f, 'decisao': dB}
    pos = []
    if depend:
        pos.append(out['testeA']['decisao'])
    if dB.startswith('FALHA DO 84 EXPLICADA'):
        pos.append(dB)
    if SA.passa_fracao(n_c, n):
        out['decisao'] = 'MODELO α(x,y) JÁ CONFIRMADO' + (f'; {out["testeA"]["decisao"]}' if depend else '')
    else:
        out['decisao'] = '; '.join(pos) if pos else 'NÃO EXPLICADO'
    out['n_posicoes_ok'] = n_c
    return out, mapa


def figuras(r, mapa, caminhos):
    vs = r['varreduras']
    n = len(vs)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for v in vs:
        cor = plt.cm.viridis(v['t'] / max(n - 1, 1))
        axs[0].plot(LAMS, [v['completa']['ck'][k]['c'] for k in LAMS], '-o', ms=3, lw=1, color=cor)
        axs[1].plot(LAMS, [v['completa']['ck'][k]['r'] for k in LAMS], '-o', ms=3, lw=1, color=cor)
        axs[2].plot(LAMS, [v['fraca']['ck'][k]['c'] for k in LAMS], '-o', ms=3, lw=1, color=cor)
    for ax in (axs[0], axs[2]):
        ax.axhspan(0.8, 1.2, color='0.9', zorder=0)
        ax.axhline(1, color='0.5', lw=1)
        ax.set_ylabel('c_k  (G[V_k] / α·G[I_k])')
    axs[1].axhline(0.7, color='0.3', ls='--', lw=1)
    axs[1].set_ylabel('r_k')
    axs[0].set_title(f"máscara do 84 — {r['testeA']['decisao']}", fontsize=8)
    axs[1].set_title(f"cor = varredura (0 → {n - 1})", fontsize=8)
    axs[2].set_title(f"campo fraco (terço inferior de F) — 84: {r['testeB']['n_passa_fraca']}/{n}", fontsize=8)
    for ax in axs:
        ax.set_xlabel('índice de λ (7 ≈ núcleo)')
        ax.set_xticks(LAMS)
    fig.suptitle(f"{r['arquivo'][8:27]}: {r['decisao']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(caminhos[0], dpi=120)
    plt.close(fig)

    t, Fm, mf = mapa
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    im = axs[0].imshow(Fm * 1e3, origin='lower', cmap='magma', vmin=0, vmax=np.nanpercentile(Fm, 99) * 1e3)
    fig.colorbar(im, ax=axs[0], shrink=0.8, label='F = |G₁₆[(V₉−V₅)/2]| / ⟨I⟩  (×10⁻³)')
    axs[1].imshow(np.where(np.isfinite(Fm), 0.3, np.nan) + mf * 0.7, origin='lower', cmap='Greys', vmin=0, vmax=1)
    axs[0].set_title(f't={t}: indicador de campo (máscara do 84)', fontsize=8)
    axs[1].set_title('preto: máscara de campo fraco; cinza: resto da máscara', fontsize=8)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(caminhos[1], dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    for c in a.arquivos:
        print('=' * 78)
        print(Path(c).name)
        r, mapa = analisar(c)
        A, B = r['testeA'], r['testeB']
        print(f"  c_k mediano: " + ' '.join(f"{k}:{A['ck_mediano'][k]:.2f}" for k in LAMS))
        print(f"  c_k mediano (campo fraco): " + ' '.join(f"{k}:{A['ck_mediano_campo_fraco'][k]:.2f}" for k in LAMS))
        print(f"  teste A: varreduras que sinalizam cada λ: "
              + ' '.join(f"{k}:{A['contagem_sinalizados'][k]}" for k in LAMS if A['contagem_sinalizados'][k])
              + f"  →  {A['decisao']}")
        print(f"  teste B: 84 passa em {B['n_passa_completa']}/{len(r['varreduras'])} (completa) e "
              f"{B['n_passa_fraca']}/{len(r['varreduras'])} (campo fraco)  →  {B['decisao']}")
        print(f"  DECISÃO: {r['decisao']}")
        rel = SA.gravar(PREFIXO, r)                          # relatório antes das figuras
        cfs = [SA.figura(n, r['etiqueta']) for n in ('cel98_ck', 'cel98_mascara_fraca')]
        figuras(r, mapa, cfs)
        print(f"  Relatório: {rel}\n  Figuras: {cfs[0]}\n           {cfs[1]}")
    SA.resumo(PREFIXO, campos_fora=('varreduras',))


if __name__ == '__main__':
    main()
