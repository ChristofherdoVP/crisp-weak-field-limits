"""
90_sst_reprodutibilidade_blos.py — os mapas de B∥·f (script 89) se reproduzem entre 08:01 e 10:48?

Os mesmos 4 ladrilhos do mosaico foram observados em 08:01 (r₀ 13 cm, elevação 16°) e 10:48 (r₀ 6 cm,
elevação 54°), com α(x, y) diferente em cada horário (script 85). Se as correções (borda, α, exclusão
simétrica) estão certas, o campo de grande escala da região ativa deve ser o mesmo nos dois horários.
A diferença NÃO mede ruído: o campo de pequena escala evolui em minutos.

Alinhamento (independente de B∥, para não favorecer a correlação dos mapas):
  - imagem: I das asas (λ0 + λ14)/2, suavizada com σ = 16 px (some a granulação; ficam poros e plage);
  - busca de rotação (−5° … +5°, passo 0,25°; refinamento ±0,5°, passo 0,05°) e translação por correlação
    cruzada (FFT), em resolução reduzida 4× e depois 2×; a correlação de Pearson de I na área comum
    alinhada é a qualidade do alinhamento;
  - o mapa de 10:48 é rotacionado e deslocado para a grade de 08:01 (resolução 2×, 0,1″/px).
Comparação: â suavizado com σ = 32 px (1,6″) e 8 px (0,4″), na área válida comum; região de campo fraco
|â₀₈| < 150 G e |â₁₀| < 150 G (fora dos poros, onde o campo fraco não vale).

Critério (definido antes de rodar), por ladrilho:
  alinhamento válido se r(I) ≥ 0,6;
  mapa reproduzido se alinhamento válido E r(â suavizado 32 px, campo fraco) ≥ 0,8.
  MAPAS REPRODUZÍVEIS se ≥ 3 dos 4 ladrilhos passam; senão NÃO REPRODUZÍVEIS.
Descritivo: r em 8 px e em resolução 2×; inclinação de Deming (erros iguais nos dois eixos); mediana e
σ robusto da diferença 10:48 − 08:01. O ladrilho 08:01 t0 (correção de α não validada) é marcado.

Os dois cubos diferem em 1 linha (2520 e 2519): imagens e mapas são recortados para a forma comum.

Uso:  python Scripts/90_sst_reprodutibilidade_blos.py <arquivo 08:01>.fits <arquivo 10:48>.fits
      (os mapas Outputs/Data/blos_f_crisp2_*_t<k>.npy do script 89 precisam existir)
Saída: Outputs/Reports/90_sst_reprodutibilidade_blos.json, Outputs/Figures/cel90_reprodutibilidade_blos.png
"""
import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, rotate, shift as nd_shift, zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'
DAT = ROOT / 'Outputs' / 'Data'
K_MAD = 1.4826
LIM_CAMPO_FRACO = 150.0
NAO_VALIDADA = {('080131', 0)}


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def reduz(a, f):
    """Média em blocos f × f, ignorando NaN."""
    ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:ny, :nx].reshape(ny // f, f, nx // f, f)
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(b, axis=(1, 3))


def suave_nan(a, s):
    m = np.isfinite(a)
    num = gaussian_filter(np.where(m, a, 0.0), s)
    den = gaussian_filter(m.astype(float), s)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(m & (den > 0.5), num / den, np.nan)


def transformar(a, ang, dy, dx):
    """Rotaciona (em torno do centro) e desloca, preservando NaN."""
    m = np.isfinite(a).astype(float)
    r = rotate(np.nan_to_num(a), ang, reshape=False, order=1, mode='constant', cval=0.0)
    rm = rotate(m, ang, reshape=False, order=1, mode='constant', cval=0.0)
    r = nd_shift(r, (dy, dx), order=1, mode='constant', cval=0.0)
    rm = nd_shift(rm, (dy, dx), order=1, mode='constant', cval=0.0)
    return np.where(rm > 0.99, r / np.maximum(rm, 1e-9), np.nan)


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 500:
        return None, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def melhor_deslocamento(A, B):
    """Translação (dy, dx) que leva B sobre A, por correlação cruzada FFT com bordas em zero."""
    a = np.where(np.isfinite(A), A - np.nanmean(A), 0.0)
    b = np.where(np.isfinite(B), B - np.nanmean(B), 0.0)
    ny, nx = a.shape
    F = np.fft.rfft2(a, s=(2 * ny, 2 * nx)) * np.conj(np.fft.rfft2(b, s=(2 * ny, 2 * nx)))
    c = np.fft.irfft2(F, s=(2 * ny, 2 * nx))
    c = np.fft.fftshift(c)
    candidatos = np.argsort(c.ravel())[::-1][:5]
    melhor = (-np.inf, 0, 0)
    for idx in candidatos:
        iy, ix = np.unravel_index(idx, c.shape)
        dy, dx = iy - ny, ix - nx
        r, n = pearson(A, nd_shift(np.nan_to_num(B, nan=np.nan), (dy, dx), order=0, mode='constant', cval=np.nan))
        if r is not None and r > melhor[0]:
            melhor = (r, dy, dx)
    return melhor


def alinhar(I08, I10):
    """Busca rotação e translação em 4× e refina em 2×. Devolve (ângulo, dy, dx) em pixels da grade 2×, r."""
    a4, b4 = reduz(I08, 4), reduz(I10, 4)
    melhor = (-np.inf, 0.0, 0, 0)
    for ang in np.arange(-5, 5.001, 0.25):
        r, dy, dx = melhor_deslocamento(a4, transformar(b4, ang, 0, 0))
        if r > melhor[0]:
            melhor = (r, ang, dy, dx)
    _, ang0, dy4, dx4 = melhor
    a2, b2 = reduz(I08, 2), reduz(I10, 2)
    melhor = (-np.inf, ang0, 2 * dy4, 2 * dx4)
    for ang in np.arange(ang0 - 0.5, ang0 + 0.501, 0.05):
        br = transformar(b2, ang, 0, 0)                     # rotaciona uma vez por ângulo
        for ddy in range(-2, 3):
            for ddx in range(-2, 3):
                dy, dx = 2 * dy4 + ddy, 2 * dx4 + ddx
                r, _ = pearson(a2, nd_shift(br, (dy, dx), order=0, mode='constant', cval=np.nan))
                if r is not None and r > melhor[0]:
                    melhor = (r, ang, dy, dx)
    return melhor


def deming(x, y):
    sxx, syy = np.var(x), np.var(y)
    sxy = np.cov(x, y, bias=True)[0, 1]
    return float((syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy))


def I_asas(caminho, t):
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        I = (np.asarray(d[t, 0, 0], dtype=np.float64) + np.asarray(d[t, 0, 14], dtype=np.float64)) / 2
        V = np.asarray(d[t, 3, 0], dtype=np.float64)
    I[~(np.isfinite(I) & np.isfinite(V) & (V != 0))] = np.nan
    return I / np.nanmedian(I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs=2)
    a = ap.parse_args()
    arqs = sorted(a.arquivos, key=lambda c: Path(c).name)
    rot = [f'{Path(c).name[8:18].replace("-", "")}T{Path(c).name[19:27].replace("_", "")}' for c in arqs]
    with fits.open(arqs[0], memmap=True) as h:
        nt = h[0].data.shape[0]
    res, paineis = [], []
    for t in range(nt):
        print(f'— ladrilho t={t}')
        Ia, Ib = I_asas(arqs[0], t), I_asas(arqs[1], t)
        ny0, nx0 = min(Ia.shape[0], Ib.shape[0]), min(Ia.shape[1], Ib.shape[1])   # 2520 × 2519 linhas
        I08 = suave_nan(Ia[:ny0, :nx0], 16)
        I10 = suave_nan(Ib[:ny0, :nx0], 16)
        rI, ang, dy, dx = alinhar(I08, I10)
        print(f'   alinhamento: rotação {ang:+.2f}°, deslocamento ({dy:+d}, {dx:+d}) px na grade 2× '
              f'({dy * 0.1006:+.2f}″, {dx * 0.1006:+.2f}″);  r(I) = {rI:.3f}  '
              f'{"válido" if rI >= 0.6 else "INVÁLIDO"}')
        A08 = reduz(np.load(DAT / f'blos_f_crisp2_{rot[0]}_t{t}.npy').astype(np.float64)[:ny0, :nx0], 2)
        A10 = reduz(np.load(DAT / f'blos_f_crisp2_{rot[1]}_t{t}.npy').astype(np.float64)[:ny0, :nx0], 2)
        ny, nx = min(A08.shape[0], A10.shape[0]), min(A08.shape[1], A10.shape[1])
        A08, A10 = A08[:ny, :nx], transformar(A10[:ny, :nx], ang, dy, dx)
        fraco = (np.abs(A08) < LIM_CAMPO_FRACO) & (np.abs(A10) < LIM_CAMPO_FRACO)
        lin = {'t': t, 'rotacao_graus': float(ang), 'desloc_px_2x': [int(dy), int(dx)], 'r_I': float(rI),
               'alinhamento_valido': bool(rI >= 0.6),
               'correcao_alfa_validada_0801': (rot[0][9:], t) not in NAO_VALIDADA}
        for s_px, nome in ((16, 's32'), (4, 's8'), (0, 'pleno')):        # σ em pixels da grade 2×
            a8 = suave_nan(np.where(fraco, A08, np.nan), s_px) if s_px else np.where(fraco, A08, np.nan)
            a1 = suave_nan(np.where(fraco, A10, np.nan), s_px) if s_px else np.where(fraco, A10, np.nan)
            r, n = pearson(a8, a1)
            m = np.isfinite(a8) & np.isfinite(a1)
            dif = (a1 - a8)[m]
            lin[nome] = {'r': r, 'n': n, 'deming': deming(a8[m], a1[m]) if n > 500 else None,
                         'dif_mediana_G': float(np.median(dif)), 'dif_sigma_G': sigma_rob(dif)}
            if nome == 's32':
                paineis.append((a8, a1, t, lin))
        lin['reproduzido'] = bool(lin['alinhamento_valido'] and lin['s32']['r'] is not None and lin['s32']['r'] >= 0.8)
        res.append(lin)
        print(f"   r(â, campo fraco):  32 px {lin['s32']['r']:.3f}   8 px {lin['s8']['r']:.3f}   pleno {lin['pleno']['r']:.3f};  "
              f"Deming (32 px) {lin['s32']['deming']:.2f};  diferença 10:48 − 08:01 (32 px): mediana "
              f"{lin['s32']['dif_mediana_G']:+.1f} G, σ {lin['s32']['dif_sigma_G']:.1f} G  →  "
              f"{'reproduzido' if lin['reproduzido'] else 'NÃO reproduzido'}"
              f"{'   [08:01 t0: correção de α não validada]' if not lin['correcao_alfa_validada_0801'] else ''}")
    n_ok = sum(x['reproduzido'] for x in res)
    decisao = 'MAPAS REPRODUZÍVEIS' if n_ok >= 3 else 'NÃO REPRODUZÍVEIS'
    print('=' * 78)
    print(f'ladrilhos reproduzidos: {n_ok} de {nt}  →  {decisao}')

    fig, axs = plt.subplots(3, nt, figsize=(3.6 * nt, 10.5), squeeze=False)
    for (a8, a1, t, lin) in paineis:
        vm = 40
        axs[0, t].imshow(a8[::2, ::2], origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
        axs[0, t].set_title(f't={t}  08:01  (32 px)', fontsize=8)
        axs[1, t].imshow(a1[::2, ::2], origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
        axs[1, t].set_title(f'10:48 alinhado  ({lin["rotacao_graus"]:+.2f}°)', fontsize=8)
        m = np.isfinite(a8) & np.isfinite(a1)
        axs[2, t].hexbin(a8[m], a1[m], gridsize=60, extent=(-vm, vm, -vm, vm), bins='log', cmap='Greys')
        axs[2, t].plot([-vm, vm], [-vm, vm], color='#D55E00', lw=1)
        axs[2, t].set_xlabel('â 08:01 (G)'); axs[2, t].set_ylabel('â 10:48 (G)')
        axs[2, t].set_title(f"r = {lin['s32']['r']:.2f}, Deming {lin['s32']['deming']:.2f}"
                            f"{'  [t0 não validado]' if not lin['correcao_alfa_validada_0801'] else ''}", fontsize=8)
        for ax in axs[:2, t]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'B∥·f campo fraco (|â| < {LIM_CAMPO_FRACO:.0f} G), suavizado 1,6″  →  {decisao}', fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel90_reprodutibilidade_blos.png', dpi=110)
    plt.close(fig)
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '90_sst_reprodutibilidade_blos.json', 'w', encoding='utf-8') as f:
        json.dump({'arquivos': [Path(c).name for c in arqs], 'ladrilhos': res, 'n_reproduzidos': n_ok,
                   'decisao': decisao}, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório: {REP / '90_sst_reprodutibilidade_blos.json'}\nFigura: {FIG / 'cel90_reprodutibilidade_blos.png'}")


if __name__ == '__main__':
    main()
