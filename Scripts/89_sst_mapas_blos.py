"""
89_sst_mapas_blos.py — mapas de B∥·f (filtro casado de campo fraco) dos mosaicos CRISP2, com todas as correções.

Produto (não é teste; não há critério de decisão). Correções validadas nos scripts 83–88:
  1. borda: exclui pixels a < 150 px da borda do campo válido (83);
  2. crosstalk I→V: V_λ ← V_λ − α(x, y)·I_λ, com o α da própria posição onde ele existe (Sol calmo e
     vizinhança, peso da suavização > 0,5) e o mapa mestre do horário nos demais pixels (84, 86);
     pixels sem nenhum dos dois ficam sem valor;
  3. quadros ruins excluídos junto com o espelho 14 − k (88): 08:01 t0 {2, 4, 10, 12}; 08:01 t1 {6, 8};
     10:48 t2 {4, 6, 8, 10}.
Filtro (artigo B, Eqs. 1–4): â = Σ u_k V_k / Σ u_k t_k, u_k = t_k/σ_k², t_k = −C g_eff λ₀² dĨ/dλ do perfil
médio de Sol calmo da posição, σ_k = ε Ĩ_k (ε do script 83). s = (Σ u_k t_k)^(−1/2) é o desvio TEÓRICO por pixel.

Ressalvas que acompanham os mapas:
  - 08:01 t0: correção de α não validada (a estrutura suave de â aumentou no script 88);
  - s teórico assume ruído ∝ I em todos os λ, não validado perto do núcleo; σ(â) medido no Sol calmo é ~2 s;
  - template de Sol calmo em toda a posição: em rede/plage o perfil difere e o campo fraco pode não valer;
  - deslocamento de λ ao longo do campo (etalon inclinado) ignorado.

Uso:  python Scripts/89_sst_mapas_blos.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
Saída: Outputs/Data/blos_f_crisp2_<data>T<hora>_t<k>.npy (float32, G, NaN fora da área válida),
       Outputs/Reports/89_sst_mapas_blos.json, Outputs/Figures/cel89_mapas_blos.png
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
DAT = ROOT / 'Outputs' / 'Data'
K_MAD = 1.4826
W = 32
D_BORDA = 150
C_ZEEMAN = 4.6686e-13
G_EFF = 1.10
LAMBDA0 = 8542.09
EPS = {'08_01_31': 2.36e-3, '10_48_36': 2.20e-3}
RUINS_ORIG = {'08_01_31': {0: [2, 4], 1: [6]}, '10_48_36': {2: [4, 6]}}
RUINS = {h: {t: sorted(set(v) | {14 - k for k in v}) for t, v in d.items()} for h, d in RUINS_ORIG.items()}
NAO_VALIDADA = {('08_01_31', 0)}


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def ler_grade(h, nl):
    hd = h[0].header
    eixo = next((i for i in range(1, hd.get('NAXIS', 0) + 1) if 'WAVE' in str(hd.get(f'CTYPE{i}', '')).upper()), None)
    if eixo is None:
        return None
    fator = {'nm': 10.0, 'm': 1e10, 'angstrom': 1.0, 'a': 1.0}.get(str(hd.get(f'CUNIT{eixo}', 'nm')).strip().lower())
    if 'TAB' in str(hd.get(f'CTYPE{eixo}', '')).upper():
        ext, col, k = hd.get(f'PS{eixo}_0'), hd.get(f'PS{eixo}_1'), hd.get(f'PV{eixo}_3', 1)
        arr = np.asarray(h[ext].data[col][0], dtype=np.float64)
        n = len(str(col).split('+'))
        if arr.shape[-1] == n and n > 1:
            arr = arr[..., int(k) - 1]
        vals = np.unique(np.round(arr.ravel(), 9))
    else:
        vals = hd[f'CRVAL{eixo}'] + (np.arange(nl) + 1 - hd.get(f'CRPIX{eixo}', 1)) * hd[f'CDELT{eixo}']
    if fator is None or len(vals) != nl:
        return None
    lam = np.sort(vals) * fator
    return lam if (8540 < np.median(lam) < 8545) else None


def processar(caminho):
    nome = Path(caminho).name
    hora = nome[19:27]
    eps, ruins = EPS[hora], RUINS.get(hora, {})
    rotulo = f'{nome[8:18].replace("-", "")}T{hora.replace("_", "")}'
    mestre_arq = DAT / f'alfa_mestre_crisp2_{rotulo}.npy'
    mestre = np.load(mestre_arq) if mestre_arq.exists() else None
    out = {'arquivo': nome, 'eps_V': eps, 'mapa_mestre_alfa': str(mestre_arq) if mestre is not None else None,
           'posicoes': []}
    figs = []
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt, ns, nl, ny, nx = d.shape
        lam = ler_grade(h, nl)
        if lam is None:
            raise SystemExit('Grade λ não lida do cabeçalho (ver script 87).')
        if mestre is not None:
            mm = np.full((ny, nx), np.nan, dtype=np.float32)
            a, b = min(ny, mestre.shape[0]), min(nx, mestre.shape[1])
            mm[:a, :b] = mestre[:a, :b]
            mestre = mm
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
            asa = I0 / np.median(I0[ok])
            nuc = Ic / np.median(Ic[ok])
            m = ok & interior & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)
            den = gaussian_filter(m.astype(np.float64), W, mode='constant')
            usa = m & (den > 0.5)
            with np.errstate(invalid='ignore', divide='ignore'):
                Gs = gaussian_filter(np.where(m, (V0 + V14) / 2, 0.0), W, mode='constant') / den
                Gi = gaussian_filter(np.where(m, (I0 + I14) / 2, 0.0), W, mode='constant') / den
            alfa_proprio = np.where(interior & ok & (den > 0.5), Gs / Gi, np.nan)
            alfa = alfa_proprio.copy()
            fonte = np.where(np.isfinite(alfa_proprio), 1, 0)
            if mestre is not None:
                usa_m = ~np.isfinite(alfa) & np.isfinite(mestre) & interior & ok
                alfa[usa_m] = mestre[usa_m]
                fonte[usa_m] = 2
            area = interior & ok & np.isfinite(alfa)
            Iw_med = float(np.mean(((I0 + I14) / 2)[usa]))
            usar = np.array([l not in ruins.get(t, []) for l in range(nl)])
            Imed = np.array([np.mean(le(0, l)[usa]) for l in range(nl)]) / Iw_med
            tk = -C_ZEEMAN * G_EFF * LAMBDA0 ** 2 * np.gradient(Imed, lam)
            uk = np.where(usar, tk / (eps * Imed) ** 2, 0.0)
            D = float(np.sum(uk * tk))
            s = 1.0 / np.sqrt(D)
            num = np.zeros((ny, nx))
            fin = np.ones((ny, nx), dtype=bool)
            for l in range(nl):
                if uk[l] == 0:
                    continue
                Il, Vl = le(0, l), le(3, l)
                fin &= np.isfinite(Il) & np.isfinite(Vl)
                num += uk[l] * np.nan_to_num(Vl - np.nan_to_num(alfa) * Il) / Iw_med
            ahat = np.where(area & fin, num / D, np.nan).astype(np.float32)
            DAT.mkdir(parents=True, exist_ok=True)
            arq = DAT / f'blos_f_crisp2_{rotulo}_t{t}.npy'
            np.save(arq, ahat)
            calmo_ok = usa & np.isfinite(ahat)
            z = ahat[calmo_ok] / s
            pos = {'t': t, 'arquivo_mapa': str(arq), 'quadros_excluidos': ruins.get(t, []),
                   'correcao_alfa_validada': (hora, t) not in NAO_VALIDADA,
                   's_teorico_G': s, 'sigma_a_calmo_G': sigma_rob(ahat[calmo_ok]),
                   'frac_area_valida': float((area & fin).sum() / max(ok.sum(), 1)),
                   'frac_alfa_proprio': float((fonte[area] == 1).mean()), 'frac_alfa_mestre': float((fonte[area] == 2).mean()),
                   'P01_P99_G': [float(np.nanpercentile(ahat, 1)), float(np.nanpercentile(ahat, 99))],
                   'frac_calmo_acima_1p96s_teorico': float(np.mean(np.abs(z) > 1.96))}
            out['posicoes'].append(pos)
            figs.append((((I0 + I14) / 2 / Iw_med)[::4, ::4], ahat[::4, ::4], pos, nome[8:27]))
            print(f"  t={t}: s teórico {s:.1f} G, σ(â) Sol calmo {pos['sigma_a_calmo_G']:.1f} G;  área válida "
                  f"{pos['frac_area_valida']:.0%} do campo (α próprio {pos['frac_alfa_proprio']:.0%}, mestre "
                  f"{pos['frac_alfa_mestre']:.0%});  â P1–P99: {pos['P01_P99_G'][0]:+.0f} a {pos['P01_P99_G'][1]:+.0f} G"
                  f"{'   [correção de α NÃO validada]' if not pos['correcao_alfa_validada'] else ''}")
    return out, figs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    todos, figs = [], []
    for c in sorted(a.arquivos, key=lambda x: Path(x).name):
        print('=' * 78)
        print(Path(c).name)
        r, f = processar(c)
        if r['mapa_mestre_alfa'] is None:
            print('  aviso: mapa mestre de α não encontrado (rode o script 86); pixels de rede ficam sem valor')
        todos.append(r)
        figs.extend(f)

    n = len(figs)
    ncol = 4
    nlin = 2 * int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nlin, ncol, figsize=(3.4 * ncol, 3.3 * nlin), squeeze=False)
    vm = float(np.ceil(6 * np.median([f[2]['s_teorico_G'] for f in figs]) / 10) * 10)   # ±6 s: rede satura
    for i, (Iw, ah, pos, rot) in enumerate(figs):
        li, co = 2 * (i // ncol), i % ncol
        axs[li, co].imshow(Iw, origin='lower', cmap='gray', vmin=0.8, vmax=1.2)
        axs[li, co].set_title(f'{rot}  t={pos["t"]}  I asas', fontsize=7)
        im = axs[li + 1, co].imshow(ah, origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
        tit = f'B∥·f  (s teórico {pos["s_teorico_G"]:.0f} G)'
        if not pos['correcao_alfa_validada']:
            tit += '\ncorreção de α NÃO validada'
        axs[li + 1, co].set_title(tit, fontsize=7, color='k' if pos['correcao_alfa_validada'] else '#D55E00')
        for ax in (axs[li, co], axs[li + 1, co]):
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.5, extend='both', label=f'B∥·f (G), escala ±{vm:.0f} G (rede saturada)')
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel89_mapas_blos.png', dpi=110)
    plt.close(fig)

    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '89_sst_mapas_blos.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos}, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório: {REP / '89_sst_mapas_blos.json'}\nFigura: {FIG / 'cel89_mapas_blos.png'}\n"
          f"Mapas: {DAT / 'blos_f_crisp2_<data>T<hora>_t<k>.npy'}")


if __name__ == '__main__':
    main()
