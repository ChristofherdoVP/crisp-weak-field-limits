"""
85_sst_alfa_mestre.py — o crosstalk α(x, y) é fixo no detector? Mapa mestre e validação cruzada 08:01 ↔ 10:48.

Script 84: a componente de V simétrica nas asas é α(x, y)·I_λ (modelo confirmado nos dois horários, 4/4).
Os mapas de α têm o mesmo padrão leste–oeste nas 8 posições, que olham para céus diferentes: o α deve
estar preso às coordenadas do detector. Se estiver, um único mapa (mediana das posições) corrige qualquer
cubo da campanha, inclusive nos pixels de rede, que a máscara de Sol calmo exclui.

Para cada arquivo e posição (máscara idêntica à do 84: Sol calmo, ≥ 150 px da borda, suavização w = 32 px):
  α_t(x, y) = G[(V₀ + V₁₄)/2] / G[(I₀ + I₁₄)/2]      (NaN fora da máscara)

Critério (definido antes de rodar), com exatamente dois arquivos (A = 08:01, B = 10:48):
  C1: correlação pixel a pixel entre α_A,t e α_B,t (mesma posição t, pixels válidos nos dois) r ≥ 0,8
      em ≥ 3 das 4 posições.
  C2: validação cruzada. Mapa mestre de A = mediana de α_A,0…3; aplicado às 4 posições de B:
        razão = σ(G[S − α_mestre·I_w]) / σ(G[S])   (amplitude da componente simétrica suave, nas asas)
      razão ≤ 0,5 em ≥ 3 das 4 posições; e o mesmo com o mestre de B aplicado a A.
  MAPA MESTRE VÁLIDO se C1 e C2 (nos dois sentidos). Senão: NÃO VÁLIDO → corrigir cada posição com o próprio α.
Descritivo: matriz de correlação 8 × 8 de todos os α; razão com o próprio α (referência do 84).

Saída extra (só se VÁLIDO): mapa mestre de todas as 8 posições, buracos preenchidos por suavização
normalizada (w = 64 px), em Outputs/Data/alfa_mestre_crisp2_20250819.npy (float32, forma comum dos dois cubos, NaN fora
do campo).

Os dois cubos podem diferir em 1 px na forma (2520 × 2520 e 2519 × 2520): ambos são recortados para a forma
comum a partir do pixel (0, 0).

Uso:  python Scripts/85_sst_alfa_mestre.py <arquivo 08:01>.fits <arquivo 10:48>.fits
Saída: Outputs/Reports/85_sst_alfa_mestre.json, Outputs/Figures/cel85_alfa_mestre.png
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
W_PREENCHE = 64
D_BORDA = 150


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def ler_posicao(d, t, ny, nx):
    """Lê o necessário de uma posição (recortada para ny × nx) e devolve máscara, G(·), S, I_w e o campo válido."""
    nl = d.shape[2]
    le = lambda s, l: np.asarray(d[t, s, l, :ny, :nx], dtype=np.float64)
    amostra = np.array([np.asarray(d[t, 0, l, :ny:8, :nx:8], dtype=np.float32) for l in range(nl)])
    okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
    lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(nl)]))
    I0, I14, Ic = le(0, 0), le(0, 14), le(0, lnuc)
    V0, V14 = le(3, 0), le(3, 14)
    ok = (np.isfinite(I0) & (I0 != 0) & np.isfinite(I14) & np.isfinite(Ic)
          & np.isfinite(V0) & (V0 != 0) & np.isfinite(V14) & (V14 != 0))
    campo = binary_fill_holes(ok)
    interior = distance_transform_edt(campo) >= D_BORDA
    asa = I0 / np.median(I0[ok])
    nuc = Ic / np.median(Ic[ok])
    m = ok & interior & (asa > 0.92) & (asa < 1.08) & (nuc < 1.15)
    den = gaussian_filter(m.astype(np.float64), W, mode='constant')
    usa = m & (den > 0.5)

    def G(X):
        with np.errstate(invalid='ignore', divide='ignore'):
            return gaussian_filter(np.where(m, X, 0.0), W, mode='constant') / den

    S = (V0 + V14) / 2
    Iw = (I0 + I14) / 2
    return {'usa': usa, 'G': G, 'S': S, 'Iw': Iw, 'campo': campo}


def alfa_de(p):
    a = p['G'](p['S']) / p['G'](p['Iw'])
    return np.where(p['usa'], a, np.nan).astype(np.float32)


def razao_correcao(p, alfa_m):
    ok = p['usa'] & np.isfinite(alfa_m)
    antes = p['G'](p['S'])
    corr = p['S'] - np.nan_to_num(alfa_m.astype(np.float64)) * p['Iw']
    depois = p['G'](corr)
    sub = np.zeros_like(ok)
    sub[::4, ::4] = True
    s = ok & sub
    return sigma_rob(depois[s]) / sigma_rob(antes[s]), float(s.mean() / max((p['usa'] & sub).mean(), 1e-12))


def corr_mapas(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    ok[1::4, :] = False; ok[2::4, :] = False; ok[3::4, :] = False
    ok[:, 1::4] = False; ok[:, 2::4] = False; ok[:, 3::4] = False
    if ok.sum() < 1000:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def mediana(mapas):
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmedian(np.stack(mapas), axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs=2)
    a = ap.parse_args()
    arqs = sorted(a.arquivos, key=lambda c: Path(c).name)          # 08:01 antes de 10:48
    nomes = [Path(c).name[8:27] for c in arqs]
    rot = ['A', 'B']
    formas = []
    for c in arqs:
        with fits.open(c, memmap=True) as h:
            formas.append(h[0].data.shape[-2:])
    ny, nx = min(f[0] for f in formas), min(f[1] for f in formas)
    if len(set(formas)) > 1:
        print(f'Formas diferentes {formas}: os dois cubos são recortados para {ny} × {nx} a partir do pixel (0, 0); '
              f'a diferença de ≤ {max(abs(formas[0][0] - formas[1][0]), abs(formas[0][1] - formas[1][1]))} px é '
              f'desprezível para um α suavizado com w = {W} px.')
    alfas = {}
    for k, c in enumerate(arqs):
        print(f'{rot[k]} = {Path(c).name}')
        with fits.open(c, memmap=True) as h:
            d = h[0].data
            for t in range(d.shape[0]):
                p = ler_posicao(d, t, ny, nx)
                alfas[(k, t)] = alfa_de(p)
                print(f'   t={t}: α calculado ({np.isfinite(alfas[(k, t)]).mean():.0%} do quadro)')
    nt = 1 + max(t for (_, t) in alfas)

    # C1: mesma posição, dois horários
    print('=' * 78)
    print('C1: correlação α(08:01) × α(10:48), mesma posição')
    c1 = []
    for t in range(nt):
        r = corr_mapas(alfas[(0, t)], alfas[(1, t)])
        c1.append({'t': t, 'r': r, 'passa': bool(r is not None and r >= 0.8)})
        print(f'   t={t}: r = {r:.3f}  {"passa" if c1[-1]["passa"] else "FALHA"}')
    n_c1 = sum(x['passa'] for x in c1)

    # matriz 8 × 8 (descritivo)
    chaves = [(k, t) for k in range(2) for t in range(nt)]
    M = np.full((len(chaves), len(chaves)), np.nan)
    for i, ki in enumerate(chaves):
        for j, kj in enumerate(chaves):
            r = corr_mapas(alfas[ki], alfas[kj])
            M[i, j] = np.nan if r is None else r

    # C2: validação cruzada
    mestre = {k: mediana([alfas[(k, t)] for t in range(nt)]) for k in range(2)}
    c2 = {}
    print('=' * 78)
    print('C2: razão de amplitude da componente simétrica após a correção (≤ 0,5 passa)')
    for k, c in enumerate(arqs):
        outro = 1 - k
        c2[rot[k]] = []
        with fits.open(c, memmap=True) as h:
            d = h[0].data
            for t in range(nt):
                p = ler_posicao(d, t, ny, nx)
                rx, cob = razao_correcao(p, mestre[outro])
                rp, _ = razao_correcao(p, alfas[(k, t)])
                c2[rot[k]].append({'t': t, 'razao_mestre_do_outro': rx, 'cobertura': cob,
                                   'razao_proprio_alfa': rp, 'passa': bool(rx <= 0.5)})
                print(f'   mestre {rot[outro]} → {rot[k]} t={t}: razão {rx:.3f}  (com o próprio α: {rp:.3f}; '
                      f'cobertura {cob:.0%})  {"passa" if rx <= 0.5 else "FALHA"}')
    n_c2 = {r: sum(x['passa'] for x in v) for r, v in c2.items()}
    valido = n_c1 >= 3 and all(n >= 3 for n in n_c2.values())
    decisao = 'MAPA MESTRE VÁLIDO' if valido else 'NÃO VÁLIDO'
    print('=' * 78)
    print(f'C1: {n_c1} de {nt}   C2 (A→B): {n_c2["B"]} de {nt}   C2 (B→A): {n_c2["A"]} de {nt}   →  {decisao}')

    # mapa mestre final (só se válido)
    caminho_mestre = None
    todos = mediana([alfas[x] for x in chaves])
    campo = np.isfinite(todos)
    w = gaussian_filter(campo.astype(np.float64), W_PREENCHE, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        preenchido = gaussian_filter(np.where(campo, todos, 0.0), W_PREENCHE, mode='constant') / w
    final = np.where(campo, todos, np.where(w > 0.2, preenchido, np.nan)).astype(np.float32)
    if valido:
        DAT.mkdir(parents=True, exist_ok=True)
        caminho_mestre = DAT / 'alfa_mestre_crisp2_20250819.npy'
        np.save(caminho_mestre, final)
        print(f'Mapa mestre salvo: {caminho_mestre}')
        print(f'   α mestre: mediana {np.nanmedian(final):+.2e}, 5–95%: {np.nanpercentile(final, 5):+.2e} a '
              f'{np.nanpercentile(final, 95):+.2e}')

    # figura
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    rot8 = [f'{rot[k]} t{t}' for (k, t) in chaves]
    im = axs[0].imshow(M, vmin=0, vmax=1, cmap='viridis')
    axs[0].set_xticks(range(len(chaves))); axs[0].set_xticklabels(rot8, rotation=90, fontsize=7)
    axs[0].set_yticks(range(len(chaves))); axs[0].set_yticklabels(rot8, fontsize=7)
    for i in range(len(chaves)):
        for j in range(len(chaves)):
            if np.isfinite(M[i, j]):
                axs[0].text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=5,
                            color='white' if M[i, j] < 0.6 else 'black')
    fig.colorbar(im, ax=axs[0], shrink=0.8, label='r entre mapas de α')
    axs[0].set_title(f'A = {nomes[0]}, B = {nomes[1]}', fontsize=8)
    x = np.arange(nt)
    for k, (r_, cor) in enumerate(zip(['B', 'A'], ['#0072B2', '#D55E00'])):
        v = c2[r_]
        axs[1].bar(x + (k - 0.5) * 0.38, [q['razao_mestre_do_outro'] for q in v], width=0.38, color=cor,
                   edgecolor='white', label=f'mestre do outro → {r_}')
        axs[1].plot(x + (k - 0.5) * 0.38, [q['razao_proprio_alfa'] for q in v], '_', color='k', ms=14, mew=2)
    axs[1].axhline(0.5, color='0.3', ls='--', lw=1)
    axs[1].set_xticks(x); axs[1].set_xlabel('posição (t)')
    axs[1].set_ylabel('amplitude depois / antes  (traço: próprio α)'); axs[1].legend(fontsize=7)
    vm = np.nanpercentile(np.abs(final), 99)
    im2 = axs[2].imshow(final[::8, ::8] * 1e3, origin='lower', cmap='RdBu_r', vmin=-vm * 1e3, vmax=vm * 1e3)
    axs[2].set_xticks([]); axs[2].set_yticks([])
    axs[2].set_title(f'α mestre (8 posições)  →  {decisao}', fontsize=8)
    fig.colorbar(im2, ax=axs[2], shrink=0.8, label='α  (×10⁻³)')
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel85_alfa_mestre.png', dpi=120)
    plt.close(fig)

    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '85_sst_alfa_mestre.json', 'w', encoding='utf-8') as f:
        json.dump({'arquivos': {'A': Path(arqs[0]).name, 'B': Path(arqs[1]).name},
                   'C1': c1, 'C2': c2, 'n_C1': n_c1, 'n_C2': n_c2, 'decisao': decisao,
                   'matriz_correlacao': {'ordem': rot8, 'r': [[None if not np.isfinite(v) else float(v) for v in l] for l in M]},
                   'mapa_mestre': None if caminho_mestre is None else str(caminho_mestre)},
                  f, ensure_ascii=False, indent=2)
    print(f"\nRelatório: {REP / '85_sst_alfa_mestre.json'}\nFigura: {FIG / 'cel85_alfa_mestre.png'}")


if __name__ == '__main__':
    main()
