"""
103_sst_falso_alarme.py — o limiar de 5% de falso alarme do artigo A vale no ruído real dos mapas de B∥·f?

Artigo A: detecção com limiar fixado a 5% de falso alarme, |â| > z_c·s com z_c = 1,96 (ruído gaussiano), validado
só no sintético. Script 100: o desvio por pixel dos mapas das séries CRISP 2024-10-08 é o s teórico (R ≈ 1).
Falta saber se as CAUDAS são gaussianas: caudas pesadas aumentam o falso alarme e mudam os a₅₀.

Ruído puro, sem sinal solar, dos mapas salvos pelo 100 (Outputs/Data/blos_f_crisp_<…>_t<k>.npy):
  (i)  uma varredura: Δâ = â(t+1) − â(t), passa-alta 32 px (como no 100), z = Δâ / √(s_t² + s_t+1²);
  (ii) média temporal: maior N do 101, A − B (pares × ímpares de 2N varreduras), z = (A − B)/(√2 σ_N),
       σ_N medido no 101 (ganho √N validado).
  Máscara "Sol calmo" (I, ≥ 150 px da borda) da varredura 0, como nos scripts 79–102.
  Frações |z| > 1,960 (esperado 5%), > 2,576 (1%), > 3,291 (0,1%); quantil empírico de |z| a 95%.
  Com √2 no denominador, z tem a variância de um mapa só: a fração estima o falso alarme de um mapa.

Critério (definido antes de rodar), por série:
  FALSO ALARME VALIDADO se a fração |z| > 1,96 está entre 4% e 6% em ≥ 75% dos pares de varreduras (i)
  E no caso (ii);  → o limiar z_c = 1,96 com o s teórico dá 5% nos dados.
  Senão: NÃO VALIDADO → o z_c que dá 5% (quantil empírico) é reportado para corrigir os a₅₀.
Descritivo: as mesmas frações com σ robusto empírico no lugar de s (só a forma da distribuição).
CRISP2: sem varreduras repetidas; lá o desvio real é 1,6–1,8 × o modelo (92/93), e os a₅₀ usam o σ empírico.

Uso:  python Scripts/103_sst_falso_alarme.py Data/RAW_SST/nb_8542_2024-10-08*.fits
Saída: Outputs/Reports/103_sst_falso_alarme_<…>.json (+ _resumo.json), Outputs/Figures/cel103_falso_alarme_<…>.png
"""
import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saidas_sst as SA

PREFIXO = '103_sst_falso_alarme'
DAT = SA.ROOT / 'Outputs' / 'Data'
K_MAD = 1.4826
W = 32
D_BORDA = 150
Z = (1.960, 2.576, 3.291)
FA_MIN, FA_MAX = 0.04, 0.06


def mascara_calmo(caminho):
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nl = d.shape[2]
        amostra = np.array([np.asarray(d[0, 0, l, ::8, ::8], dtype=np.float32) for l in range(nl)])
        okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
        lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(nl)]))
        I0, I14, Ic = (np.asarray(d[0, 0, l], dtype=np.float64) for l in (0, 14, lnuc))
        V0 = np.asarray(d[0, 3, 0], dtype=np.float64)
    ok = np.isfinite(I0) & (I0 != 0) & np.isfinite(I14) & np.isfinite(Ic) & np.isfinite(V0) & (V0 != 0)
    interior = distance_transform_edt(binary_fill_holes(ok)) >= D_BORDA
    asa = I0 / np.median(I0[ok])
    nuc = Ic / np.median(Ic[ok])
    return ok & interior & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)


def passa_alta(X, m):
    mm = m & np.isfinite(X)
    den = gaussian_filter(mm.astype(np.float64), W, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        lo = gaussian_filter(np.where(mm, X, 0.0), W, mode='constant') / den
    usa = mm & (den > 0.5)
    return (X - lo)[usa]


def estatisticas(z):
    z = z[np.isfinite(z)]
    a = np.abs(z)
    s_rob = K_MAD * np.median(np.abs(z - np.median(z)))
    r = {'n': int(z.size), 'mediana': float(np.median(z)), 'sigma_rob': float(s_rob),
         'frac': {f'{zc:.3f}': float(np.mean(a > zc)) for zc in Z},
         'frac_sigma_empirico': {f'{zc:.3f}': float(np.mean(np.abs(z - np.median(z)) / s_rob > zc)) for zc in Z},
         'zc_95_empirico': float(np.percentile(a, 95))}
    r['passa'] = bool(FA_MIN <= r['frac']['1.960'] <= FA_MAX)
    return r


def analisar(caminho):
    nome = Path(caminho).name
    tag = SA.etiqueta(nome)
    r100 = json.loads((SA.REP / f'100_sst_mapas_blos_series_{tag}.json').read_text(encoding='utf-8'))['data'][0]
    r101 = json.loads((SA.REP / f'101_sst_media_temporal_{tag}.json').read_text(encoding='utf-8'))['data'][0]
    s = {v['t']: v['s_teorico_G'] for v in r100['varreduras']}
    nt = len(s)
    mapas = [np.load(DAT / f'blos_f_crisp_{tag}_t{t}.npy').astype(np.float64) for t in range(nt)]
    m = mascara_calmo(caminho)
    out = {'arquivo': nome, 'pares': []}
    hist = []
    for t in range(nt - 1):
        z = passa_alta(mapas[t + 1] - mapas[t], m) / np.sqrt(s[t] ** 2 + s[t + 1] ** 2)
        e = estatisticas(z)
        e['t'] = [t, t + 1]
        out['pares'].append(e)
        hist.append(z)
        print(f"  par {t}→{t + 1}: |z| > 1,96: {e['frac']['1.960']:.2%}   > 2,58: {e['frac']['2.576']:.2%}   "
              f"> 3,29: {e['frac']['3.291']:.3%}   σ_rob(z) {e['sigma_rob']:.3f}   z₉₅ empírico {e['zc_95_empirico']:.2f}"
              f"  → {'passa' if e['passa'] else 'FALHA'}")
    N = int(r101['N_max'])
    sN = float(r101['por_N'][-1]['sigma_media_N_G'])
    bloco = mapas[:2 * N]
    A = np.mean(bloco[0::2], axis=0)
    B = np.mean(bloco[1::2], axis=0)
    zm = passa_alta(A - B, m) / (np.sqrt(2) * sN)
    em = estatisticas(zm)
    em['N'] = N
    out['media'] = em
    print(f"  média de N = {N}: |z| > 1,96: {em['frac']['1.960']:.2%}   > 2,58: {em['frac']['2.576']:.2%}   "
          f"> 3,29: {em['frac']['3.291']:.3%}   z₉₅ empírico {em['zc_95_empirico']:.2f}  → {'passa' if em['passa'] else 'FALHA'}")
    n_ok = sum(p['passa'] for p in out['pares'])
    out['n_posicoes_ok'] = n_ok
    ok = SA.passa_fracao(n_ok, len(out['pares'])) and em['passa']
    out['decisao'] = 'FALSO ALARME VALIDADO' if ok else 'FALSO ALARME NÃO VALIDADO'
    out['frac_1p96_mediana'] = float(np.median([p['frac']['1.960'] for p in out['pares']]))
    out['zc_95_empirico_mediano'] = float(np.median([p['zc_95_empirico'] for p in out['pares']]))
    return out, np.concatenate(hist), zm


def figura(r, z1, zm, caminho):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    x = np.linspace(-6, 6, 241)
    c = 0.5 * (x[1:] + x[:-1])
    for ax, z, tit in ((axs[0], z1, 'uma varredura (diferenças consecutivas)'),
                       (axs[1], zm, f"média de N = {r['media']['N']} (pares − ímpares)")):
        h, _ = np.histogram(z[np.isfinite(z)], bins=x, density=True)
        ax.semilogy(c, h, color='#0072B2', lw=1.5, label='dados')
        ax.semilogy(c, norm.pdf(c), 'k--', lw=1, label='gaussiana N(0, 1)')
        for zc in (-1.96, 1.96):
            ax.axvline(zc, color='0.5', lw=1, ls=':')
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel('z = ruído / s'); ax.set_ylabel('densidade')
        ax.set_title(tit, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle(f"{r['arquivo'][8:27]}: |z| > 1,96 em {r['frac_1p96_mediana']:.2%} (mediana dos pares), "
                 f"{r['media']['frac']['1.960']:.2%} (média)  →  {r['decisao']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    for c in a.arquivos:
        print('=' * 78)
        print(Path(c).name)
        r, z1, zm = analisar(c)
        print(f"  pares com 4% ≤ fração ≤ 6%: {r['n_posicoes_ok']} de {len(r['pares'])};  média: "
              f"{'passa' if r['media']['passa'] else 'FALHA'}  →  {r['decisao']}")
        print(f"  z que dá 5% nos dados (mediana dos pares): {r['zc_95_empirico_mediano']:.2f}  (gaussiano: 1,96)")
        rel = SA.gravar(PREFIXO, r)
        cf = SA.figura('cel103_falso_alarme', r['etiqueta'])
        figura(r, z1, zm, cf)
        print(f"  Relatório: {rel}\n  Figura: {cf}")
    SA.resumo(PREFIXO, campos_fora=('pares', 'media'))


if __name__ == '__main__':
    main()
