"""
101_sst_media_temporal.py — ganho da média temporal nos mapas de B∥·f das séries CRISP 2024-10-08.

Script 100: mapas por varredura com ruído validado (R mediano 1,01 e 0,98; s = 16,5 e 24,3 G).
Artigo A (briefing, achado 17, cópia antiga de 2024-07-08): média de 10 varreduras deu ganho 3,11 (ideal 3,16);
agrupamento 2×2 só 1,11 (ruído correlacionado em 2–3 px). Planejamento do CRISP no modelo sintético:
média de 11 varreduras → B·f₅₀ = 19,2 G (teoria 17,7 G). Aqui o ganho é medido nos mapas novos.

Mapas lidos de Outputs/Data/blos_f_crisp_<AAAAMMDDThhmmss>_t<k>.npy (script 100). Máscara "Sol calmo" (I, ≥ 150 px
da borda) da varredura 0, como nos scripts 79–100. Sem realinhamento (100: deslocamento inteiro 0 em todos os pares).

Para cada N = 1 … ⌊nt/2⌋, em blocos de 2N varreduras consecutivas sem sobreposição:
  A = média das varreduras pares do bloco, B = média das ímpares (N cada; mesma época média, a evolução solar
  se cancela em primeira ordem);  σ_N = σ_rob(passa-alta₃₂(A − B)) / √2 = desvio de UMA média de N varreduras
  (média sobre os blocos);  ganho medido G(N) = σ_1 / σ_N;  ideal G = √N.
Descritivo: agrupamento 2×2 (média de 4 px) em N = 1: ganho G_bin (ideal 2 para ruído independente entre pixels).

Critério (definido antes de rodar), por série:
  MÉDIA TEMPORAL VALIDADA se G(N)/√N ≥ 0,9 para o maior N disponível (7 às 09:49, 5 às 10:10)
  → a₅₀ de um mapa médio de N varreduras = 1,96 s/√N.
  Senão: NÃO VALIDADA (ruído correlacionado entre varreduras ou evolução) → reportar o ganho medido.
Limitação: diferenças entre varreduras não enxergam ruído FIXO (igual em todas as varreduras), que também não cai com
a média; a parte instrumental fixa conhecida é a componente simétrica (α), limitada pelo script 99 (≤ 0,9–1,2 G).
Descritivo: a₅₀ com a média de todas as varreduras, extrapolação para N = 11 (planejamento do briefing,
19,2 G no sintético; comparação qualitativa: perfil, grade e máscara diferentes).

Uso:  python Scripts/101_sst_media_temporal.py Data/RAW_SST/nb_8542_2024-10-08*.fits
Saída: Outputs/Reports/101_sst_media_temporal_<…>.json (+ _resumo.json), Outputs/Figures/cel101_media_temporal_<…>.png,
       Outputs/Data/blos_f_crisp_<…>_media.npy (média de todas as varreduras)
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

PREFIXO = '101_sst_media_temporal'
DAT = SA.ROOT / 'Outputs' / 'Data'
K_MAD = 1.4826
W = 32
D_BORDA = 150
Z50 = 1.96
LIMIAR = 0.9
N_PLANEJ = 11


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


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
    return ok & interior & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15), ((I0 + I14) / 2)


def sigma_hp(X, m):
    """σ robusto do passa-alta (subtrai G₃₂ normalizada na máscara), longe da borda da máscara."""
    mm = m & np.isfinite(X)
    den = gaussian_filter(mm.astype(np.float64), W, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        lo = gaussian_filter(np.where(mm, X, 0.0), W, mode='constant') / den
    usa = mm & (den > 0.5)
    return sigma_rob((X - lo)[usa]), int(usa.sum())


def bin2(X):
    ny, nx = (X.shape[0] // 2) * 2, (X.shape[1] // 2) * 2
    Y = X[:ny, :nx]
    return 0.25 * (Y[0::2, 0::2] + Y[1::2, 0::2] + Y[0::2, 1::2] + Y[1::2, 1::2])


def analisar(caminho):
    nome = Path(caminho).name
    tag = SA.etiqueta(nome)
    p100 = SA.REP / f'100_sst_mapas_blos_series_{tag}.json'
    if not p100.exists():
        raise SystemExit(f'{nome}: falta {p100.name} (rode o 100).')
    r100 = json.loads(p100.read_text(encoding='utf-8'))['data'][0]
    s_teo = float(r100['s_teorico_mediano_G'])
    if any(p['desloc_px'] != [0, 0] for p in r100['pares']):
        raise SystemExit(f'{nome}: o 100 achou deslocamento inteiro entre varreduras; este script não realinha.')
    mapas = []
    t = 0
    while (DAT / f'blos_f_crisp_{tag}_t{t}.npy').exists():
        mapas.append(np.load(DAT / f'blos_f_crisp_{tag}_t{t}.npy').astype(np.float64))
        t += 1
    nt = len(mapas)
    if nt < 4:
        raise SystemExit(f'{nome}: só {nt} mapas em Outputs/Data (rode o 100).')
    m, Iw = mascara_calmo(caminho)
    print(f"  {nt} mapas;  s teórico (100) = {s_teo:.1f} G;  máscara Sol calmo {m.mean():.0%} do quadro")
    out = {'arquivo': nome, 'n_varreduras': nt, 's_teorico_G': s_teo, 'por_N': []}
    sig1 = None
    for N in range(1, nt // 2 + 1):
        vals, npx = [], []
        for b0 in range(0, nt - 2 * N + 1, 2 * N):
            bloco = mapas[b0:b0 + 2 * N]
            A = np.mean(bloco[0::2], axis=0)
            B = np.mean(bloco[1::2], axis=0)
            sd, n = sigma_hp(A - B, m)
            vals.append(sd / np.sqrt(2))          # desvio de UMA média de N varreduras
            npx.append(n)
        sN = float(np.mean(vals))
        if N == 1:
            sig1 = sN
        G = sig1 / sN
        lin = {'N': N, 'n_blocos': len(vals), 'sigma_media_N_G': sN, 'ganho': G, 'ganho_ideal': float(np.sqrt(N)),
               'razao_ganho_ideal': G / np.sqrt(N), 'a50_G': Z50 * sN}
        out['por_N'].append(lin)
        print(f"  N = {N}: {len(vals)} bloco(s);  σ da média de N = {sN:5.2f} G;  ganho {G:.2f} (ideal {np.sqrt(N):.2f}, "
              f"razão {G / np.sqrt(N):.3f});  a₅₀ = {Z50 * sN:.1f} G")
    # agrupamento 2×2 em N = 1
    vb = []
    mb = bin2(m.astype(float)) > 0.99
    for t0 in range(0, nt - 1, 2):
        sd, _ = sigma_hp(bin2(mapas[t0 + 1] - mapas[t0]), mb)
        vb.append(sd / np.sqrt(2))
    s_bin = float(np.mean(vb))
    out['binning_2x2'] = {'sigma_G': s_bin, 'ganho': sig1 / s_bin, 'ganho_ideal': 2.0}
    print(f"  agrupamento 2×2 (N = 1): σ = {s_bin:.2f} G → ganho {sig1 / s_bin:.2f} (ideal 2 se o ruído fosse "
          f"independente entre pixels)")
    ult = out['por_N'][-1]
    out['N_max'] = ult['N']
    out['razao_N_max'] = ult['razao_ganho_ideal']
    out['decisao'] = ('MÉDIA TEMPORAL VALIDADA' if ult['razao_ganho_ideal'] >= LIMIAR
                      else 'MÉDIA TEMPORAL NÃO VALIDADA')
    out['sigma_1_medido_G'] = sig1
    out['sigma_1_sobre_s_teorico'] = sig1 / s_teo
    out['a50_media_todas_G'] = Z50 * sig1 / np.sqrt(nt)
    out['a50_N11_extrapolado_G'] = Z50 * sig1 / np.sqrt(N_PLANEJ)
    media = np.mean(mapas, axis=0).astype(np.float32)
    arq = DAT / f'blos_f_crisp_{tag}_media.npy'
    np.save(arq, media)
    out['arquivo_media'] = str(arq)
    return out, mapas[0], media, Iw, m


def figura(r, a0, am, Iw, m, caminho):
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8), gridspec_kw={'width_ratios': [1, 1, 1.15, 1.15]})
    vm = float(np.ceil(4 * r['s_teorico_G'] / 10) * 10)
    for ax, X, tit in ((axs[0], a0, 'uma varredura'), (axs[1], am, f"média de {r['n_varreduras']} varreduras")):
        im = ax.imshow(X[::4, ::4], origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
        ax.set_title(f"{r['arquivo'][8:27]}: {tit}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs[1], shrink=0.8, extend='both', label=f'B∥·f (G), ±{vm:.0f} G')
    N = np.array([q['N'] for q in r['por_N']])
    G = np.array([q['ganho'] for q in r['por_N']])
    Nc = np.linspace(1, max(N.max(), N_PLANEJ), 100)
    axs[2].plot(Nc, np.sqrt(Nc), color='0.5', lw=1, label='√N (ideal)')
    axs[2].plot(Nc, LIMIAR * np.sqrt(Nc), color='0.5', lw=1, ls='--', label=f'{LIMIAR}·√N (limiar)')
    axs[2].plot(N, G, 'o', color='#0072B2', ms=6, label='medido')
    axs[2].set_xlabel('varreduras na média (N)'); axs[2].set_ylabel('ganho σ₁/σ_N')
    axs[2].set_title(r['decisao'], fontsize=8); axs[2].legend(fontsize=7)
    a50 = np.array([q['a50_G'] for q in r['por_N']])
    axs[3].plot(N, a50, 'o', color='#0072B2', ms=6, label='medido (1,96 σ_N)')
    axs[3].plot(Nc, Z50 * r['sigma_1_medido_G'] / np.sqrt(Nc), color='0.5', lw=1, label='1,96 σ₁/√N')
    axs[3].axvline(N_PLANEJ, color='#D55E00', lw=1, ls=':')
    axs[3].text(N_PLANEJ, axs[3].get_ylim()[1] * 0.95, f" N = {N_PLANEJ}: {r['a50_N11_extrapolado_G']:.1f} G",
                fontsize=7, color='#D55E00', va='top')
    axs[3].set_xlabel('varreduras na média (N)'); axs[3].set_ylabel('a₅₀ por pixel (G)')
    axs[3].legend(fontsize=7)
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
        r, a0, am, Iw, m = analisar(c)
        print(f"  σ₁ medido / s teórico = {r['sigma_1_sobre_s_teorico']:.3f};  razão ganho/√N em N = {r['N_max']}: "
              f"{r['razao_N_max']:.3f}  →  {r['decisao']}")
        print(f"  a₅₀ por pixel: 1 varredura {Z50 * r['sigma_1_medido_G']:.1f} G;  média de {r['n_varreduras']}: "
              f"{r['a50_media_todas_G']:.1f} G;  extrapolado para N = {N_PLANEJ}: {r['a50_N11_extrapolado_G']:.1f} G "
              f"(planejamento sintético do briefing: 19,2 G)")
        rel = SA.gravar(PREFIXO, r)                          # relatório antes da figura
        cf = SA.figura('cel101_media_temporal', r['etiqueta'])
        figura(r, a0, am, Iw, m, cf)
        print(f"  Relatório: {rel}\n  Figura: {cf}\n  Mapa médio: {r['arquivo_media']}")
    SA.resumo(PREFIXO, campos_fora=('por_N',))


if __name__ == '__main__':
    main()
