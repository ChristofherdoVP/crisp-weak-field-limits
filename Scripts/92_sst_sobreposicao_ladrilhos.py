"""
92_sst_sobreposicao_ladrilhos.py — ruído real do filtro casado e consistência da correção de α nas sobreposições
entre ladrilhos vizinhos do mesmo horário; mosaico de B∥·f em coordenadas solares.

Script 91: mosaico 2×2 com passo de ~81″ e campo de ~127″ → ladrilhos vizinhos se sobrepõem ~45″. Os ladrilhos
são observados em sequência (08:01: 20,6 s entre eles; 10:48: ~61 s), em posições diferentes do detector.
Na sobreposição, o mesmo pedaço do Sol foi visto duas vezes quase ao mesmo tempo:
  Δâ = â_a − â_b   (â dos mapas do script 89, já corrigidos de borda, α e quadros ruins)
  - σ(Δâ)/√(s_a² + s_b²) testa o modelo de ruído do filtro (s teórico, ruído ∝ I; script 89). É um TETO:
    inclui evolução solar em 20–60 s (até ~180 s no par t3–t0 de 10:48), seeing diferente e alinhamento;
  - a estrutura suave de Δâ (σ = 32 px) testa a correção de α: α varia com a posição no detector, então um
    erro de α apareceria como gradiente em Δâ.

Geometria:
  - pixel → (HPLN, HPLT) pelo WCS-TAB (algoritmo -TAB do FITS: ψ = CRVAL + CDELT·PC·(p − CRPIX), índice
    fracionário na tabela, interpolação bilinear; média sobre λ); ajuste afim por ladrilho;
  - verificação: escala ajustada deve ficar a < 10% de 0,0503″/px; senão, usa os cantos da tabela como os
    cantos do quadro e avisa;
  - transformação a → b pela composição dos ajustes afins; refinamento da translação por correlação cruzada
    de I das asas (passa-alta, granulação) na sobreposição; â_b é amostrado por vizinho mais próximo
    (interpolar suavizaria o ruído e subestimaria σ);
  - correções de apontamento encadeadas t0 → t1 → t2 → t3 e fechamento pelo par t3 → t0 (diagnóstico).

Critérios (definidos antes de rodar), por par vizinho (t0–t1, t1–t2, t2–t3, t3–t0), região de campo fraco
(|â_a| e |â_b| < 150 G):
  alinhamento válido se r(I passa-alta) ≥ 0,8;
  modelo de ruído: R = σ(Δâ)/√(s_a² + s_b²) entre 0,8 e 1,25;
  correção de α consistente: σ(G₃₂[Δâ]) < 0,2 σ(Δâ) E |mediana(Δâ)| < 0,2 σ(Δâ)
    (a faixa de sobreposição é estreita: um erro de α em gradiente aparece nela quase como desvio constante,
     que o σ da estrutura suave não vê — verificado em teste sintético antes de rodar nos dados).
  Por horário: MODELO DE RUÍDO VALIDADO se alinhamento válido e R na faixa em ≥ 3 dos 4 pares;
               CORREÇÃO DE α CONSISTENTE se alinhamento válido e estrutura < 0,2 em ≥ 3 dos 4 pares.

Uso:  python Scripts/92_sst_sobreposicao_ladrilhos.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
      (precisa dos mapas Outputs/Data/blos_f_crisp2_*_t<k>.npy e do relatório 89)
Saída: Outputs/Reports/92_sst_sobreposicao_ladrilhos.json,
       Outputs/Figures/cel92_sobreposicao.png e cel92_mosaico_blos.png
"""
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, map_coordinates, shift as nd_shift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'
DAT = ROOT / 'Outputs' / 'Data'
K_MAD = 1.4826
ESCALA = 0.0503            # ″/px nominal
LIM_FRACO = 150.0
PARES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def suave_nan(a, s):
    m = np.isfinite(a)
    num = gaussian_filter(np.where(m, a, 0.0), s)
    den = gaussian_filter(m.astype(float), s)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(m & (den > 0.5), num / den, np.nan)


# ───────────────────────── geometria ─────────────────────────
def afins_wcs(h):
    """Ajuste afim pixel (x, y; 0-based) → (HPLN, HPLT) por passo t. Devolve lista de matrizes 2×3 e diagnóstico."""
    hd = h[0].header
    nt, _, nl, ny, nx = h[0].data.shape
    eixo = {}
    for i in range(1, hd['NAXIS'] + 1):
        c = str(hd.get(f'CTYPE{i}', '')).upper()
        if c.startswith('HPLN'):
            eixo['x'] = i
        if c.startswith('HPLT'):
            eixo['y'] = i
    ix, iy = eixo['x'], eixo['y']
    ext, col = hd[f'PS{ix}_0'], hd[f'PS{ix}_1']
    arr = np.asarray(h[ext].data[col][0], dtype=np.float64)          # (nt, nl, K2, K1, M)
    nomes = [n.strip().upper() for n in str(col).split('+')]
    mx, my = nomes.index('HPLN'), nomes.index('HPLT')
    K2, K1 = arr.shape[-3], arr.shape[-2]

    def indice(i, K):
        nome_idx = hd.get(f'PS{i}_2')
        if nome_idx:
            try:
                return np.asarray(h[ext].data[nome_idx][0], dtype=np.float64).ravel()
            except Exception:
                pass
        return np.arange(1, K + 1, dtype=np.float64)

    ups_x, ups_y = indice(ix, K1), indice(iy, K2)

    def psi(i, px, py):
        pc = lambda a, b: hd.get(f'PC{a}_{b}', 1.0 if a == b else 0.0)
        p = {ix: px + 1.0, iy: py + 1.0}
        s = sum(pc(i, j) * (p[j] - hd.get(f'CRPIX{j}', 0.0)) for j in (ix, iy))
        return hd.get(f'CRVAL{i}', 0.0) + hd.get(f'CDELT{i}', 1.0) * s

    gy, gx = np.meshgrid(np.linspace(0, ny - 1, 6), np.linspace(0, nx - 1, 6), indexing='ij')
    gx, gy = gx.ravel(), gy.ravel()
    afins, diag = [], []
    for t in range(nt):
        C = arr[t].mean(axis=0)                                       # (K2, K1, M), média em λ
        modo = 'WCS-TAB'
        for tentativa in range(2):
            if tentativa == 0:
                fx = np.interp(psi(ix, gx, gy), ups_x, np.arange(K1), left=np.nan, right=np.nan)
                fy = np.interp(psi(iy, gx, gy), ups_y, np.arange(K2), left=np.nan, right=np.nan)
                if np.any(~np.isfinite(fx)) or np.any(~np.isfinite(fy)):   # extrapolação linear
                    fx = (psi(ix, gx, gy) - ups_x[0]) / (ups_x[-1] - ups_x[0]) * (K1 - 1)
                    fy = (psi(iy, gx, gy) - ups_y[0]) / (ups_y[-1] - ups_y[0]) * (K2 - 1)
            else:
                modo = 'cantos da tabela = cantos do quadro'
                fx = gx / (nx - 1) * (K1 - 1)
                fy = gy / (ny - 1) * (K2 - 1)
            x0, y0 = np.clip(np.floor(fx).astype(int), 0, K1 - 2), np.clip(np.floor(fy).astype(int), 0, K2 - 2)
            wx, wy = fx - x0, fy - y0
            def bil(m):
                return (C[y0, x0, m] * (1 - wx) * (1 - wy) + C[y0, x0 + 1, m] * wx * (1 - wy)
                        + C[y0 + 1, x0, m] * (1 - wx) * wy + C[y0 + 1, x0 + 1, m] * wx * wy)
            X = np.vstack([gx, gy, np.ones_like(gx)]).T
            cx, *_ = np.linalg.lstsq(X, bil(mx), rcond=None)
            cy, *_ = np.linalg.lstsq(X, bil(my), rcond=None)
            A = np.vstack([cx, cy])                                   # sky = A @ [x, y, 1]
            esc = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
            if abs(esc / ESCALA - 1) < 0.10:
                break
        ang = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        diag.append({'t': t, 'modo': modo, 'escala_arcsec_px': esc, 'angulo_eixo_x_graus': ang,
                     'centro': [float(v) for v in A @ np.array([(nx - 1) / 2, (ny - 1) / 2, 1.0])]})
        afins.append(A)
    return afins, diag


def a_para_b(Aa, Ab):
    """Matriz 2×3 que leva pixel de a em pixel de b."""
    La, la = Aa[:, :2], Aa[:, 2]
    Lb, lb = Ab[:, :2], Ab[:, 2]
    Lbi = np.linalg.inv(Lb)
    return np.hstack([Lbi @ La, (Lbi @ (la - lb))[:, None]])


def amostrar(img, T, ys, xs, ordem):
    """img de b amostrada nas posições de a (grade ys × xs), via T (a → b)."""
    YY, XX = np.meshgrid(ys, xs, indexing='ij')
    xb = T[0, 0] * XX + T[0, 1] * YY + T[0, 2]
    yb = T[1, 0] * XX + T[1, 1] * YY + T[1, 2]
    m = np.isfinite(img).astype(float)
    v = map_coordinates(np.nan_to_num(img), [yb, xb], order=ordem, mode='constant', cval=0.0)
    w = map_coordinates(m, [yb, xb], order=ordem, mode='constant', cval=0.0)
    return np.where(w > 0.99, v / np.maximum(w, 1e-9), np.nan)


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return (float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())) if m.sum() > 500 else (None, int(m.sum()))


def melhor_desloc(A, B, maxd):
    a = np.where(np.isfinite(A), A - np.nanmean(A), 0.0)
    b = np.where(np.isfinite(B), B - np.nanmean(B), 0.0)
    ny, nx = a.shape
    c = np.fft.irfft2(np.fft.rfft2(a, s=(2 * ny, 2 * nx)) * np.conj(np.fft.rfft2(b, s=(2 * ny, 2 * nx))),
                      s=(2 * ny, 2 * nx))
    c = np.fft.fftshift(c)
    cy, cx = ny, nx
    jan = c[cy - maxd:cy + maxd + 1, cx - maxd:cx + maxd + 1]
    iy, ix = np.unravel_index(np.argmax(jan), jan.shape)
    return iy - maxd, ix - maxd


# ───────────────────────── análise ─────────────────────────
def I_asas(d, t):
    I = (np.asarray(d[t, 0, 0], dtype=np.float64) + np.asarray(d[t, 0, 14], dtype=np.float64)) / 2
    V = np.asarray(d[t, 3, 0], dtype=np.float64)
    I[~(np.isfinite(I) & np.isfinite(V) & (V != 0))] = np.nan
    return I / np.nanmedian(I)


def analisar(caminho, s_teo):
    nome = Path(caminho).name
    rot = f'{nome[8:18].replace("-", "")}T{nome[19:27].replace("_", "")}'
    out = {'arquivo': nome, 'pares': []}
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt, _, _, ny, nx = d.shape
        afins, diag = afins_wcs(h)
        out['geometria'] = diag
        for g in diag:
            print(f"  t={g['t']}: centro ({g['centro'][0]:+.1f}″, {g['centro'][1]:+.1f}″), escala "
                  f"{g['escala_arcsec_px']:.4f}″/px, eixo x a {g['angulo_eixo_x_graus']:+.2f}°  [{g['modo']}]")
        Is = [I_asas(d, t) for t in range(nt)]
    As = [np.load(DAT / f'blos_f_crisp2_{rot}_t{t}.npy').astype(np.float64) for t in range(nt)]
    corr = {0: np.zeros(2)}
    paineis = []
    for (a, b) in PARES:
        T = a_para_b(afins[a], afins[b])
        # caixa da sobreposição na grade de a (amostragem esparsa para achar a região)
        ys, xs = np.arange(0, ny, 8), np.arange(0, nx, 8)
        cob = np.isfinite(amostrar(As[b], T, ys, xs, 0)) & np.isfinite(As[a][::8, ::8][:len(ys), :len(xs)])
        if cob.sum() < 200:
            print(f'  par t{a}–t{b}: sem sobreposição útil')
            out['pares'].append({'par': [a, b], 'sobreposicao': False})
            continue
        yy, xx = np.where(cob)
        y0, y1 = max(0, ys[yy.min()] - 16), min(ny, ys[yy.max()] + 16)
        x0, x1 = max(0, xs[xx.min()] - 16), min(nx, xs[xx.max()] + 16)
        ysF, xsF = np.arange(y0, y1), np.arange(x0, x1)
        Ia = Is[a][y0:y1, x0:x1]
        Ib = amostrar(Is[b], T, ysF, xsF, 1)
        hp = lambda X: X - suave_nan(X, 20)
        Ia_hp, Ib_hp = hp(Ia), hp(Ib)
        # refinamento: grosso em 2× (±60 px de busca), fino em resolução cheia (±2 px)
        dy2, dx2 = melhor_desloc(Ia_hp[::2, ::2], Ib_hp[::2, ::2], 30)
        melhor = (-np.inf, 0, 0)
        for ddy in range(-2, 3):
            for ddx in range(-2, 3):
                dyy, dxx = 2 * dy2 + ddy, 2 * dx2 + ddx
                r, _ = pearson(Ia_hp, nd_shift(Ib_hp, (dyy, dxx), order=0, mode='constant', cval=np.nan))
                if r is not None and r > melhor[0]:
                    melhor = (r, dyy, dxx)
        rI, dy, dx = melhor
        Tr = T.copy()
        Tr[:, 2] = T[:, 2] - T[:, :2] @ np.array([dx, dy], dtype=float)     # T'(p) = T(p − d)
        Ab = amostrar(As[b], Tr, ysF, xsF, 0)
        Aa = As[a][y0:y1, x0:x1]
        fraco = np.isfinite(Aa) & np.isfinite(Ab) & (np.abs(Aa) < LIM_FRACO) & (np.abs(Ab) < LIM_FRACO)
        D = np.where(fraco, Aa - Ab, np.nan)
        sD = sigma_rob(D[fraco])
        sesp = float(np.hypot(s_teo[a], s_teo[b]))
        R = sD / sesp
        Ds = suave_nan(D, 32)
        sS = sigma_rob(Ds[np.isfinite(Ds)])
        dsky = afins[a][:, :2] @ np.array([dx, dy], dtype=float)
        if b != 0:
            corr[b] = corr[a] + dsky
        else:
            out['fechamento_arcsec'] = [float(v) for v in (corr[a] + dsky)]
        lin = {'par': [a, b], 'sobreposicao': True, 'n_pixels_campo_fraco': int(fraco.sum()),
               'desloc_refino_px': [int(dy), int(dx)], 'desloc_refino_arcsec': [float(v) for v in dsky],
               'r_I': float(rI), 'alinhamento_valido': bool(rI >= 0.8),
               's_a_G': s_teo[a], 's_b_G': s_teo[b], 'sigma_dif_G': sD, 'R': R,
               'R_na_faixa': bool(0.8 <= R <= 1.25), 'estrutura_suave_G': sS, 'razao_estrutura': sS / sD,
               'mediana_dif_G': float(np.median(D[fraco]))}
        lin['estrutura_ok'] = bool(sS < 0.2 * sD and abs(lin['mediana_dif_G']) < 0.2 * sD)
        out['pares'].append(lin)
        paineis.append((a, b, D, Ds, lin))
        print(f"  par t{a}–t{b}: refino ({dy:+d}, {dx:+d}) px = ({dsky[0]:+.2f}″, {dsky[1]:+.2f}″);  r(I) = {rI:.3f}  "
              f"{'válido' if lin['alinhamento_valido'] else 'INVÁLIDO'};  {lin['n_pixels_campo_fraco']} px")
        print(f"        σ(Δâ) = {sD:.1f} G,  √(s_a² + s_b²) = {sesp:.1f} G  →  R = {R:.2f}  "
              f"{'na faixa' if lin['R_na_faixa'] else 'FORA da faixa'};   estrutura suave {sS:.2f} G = "
              f"{sS / sD:.3f} σ;   mediana Δâ {lin['mediana_dif_G']:+.1f} G = {lin['mediana_dif_G'] / sD:+.3f} σ  →  "
              f"{'α ok' if lin['estrutura_ok'] else 'α INCONSISTENTE'}")
    if 'fechamento_arcsec' in out:
        f = out['fechamento_arcsec']
        print(f"  fechamento t0 → t1 → t2 → t3 → t0: ({f[0]:+.2f}″, {f[1]:+.2f}″)  (ideal 0)")
    out['correcoes_apontamento_arcsec'] = {str(k): [float(x) for x in v] for k, v in corr.items()}
    validos = [p for p in out['pares'] if p.get('sobreposicao') and p['alinhamento_valido']]
    out['n_R_ok'] = sum(p['R_na_faixa'] for p in validos)
    out['n_estrutura_ok'] = sum(p['estrutura_ok'] for p in validos)
    out['decisao_ruido'] = 'MODELO DE RUÍDO VALIDADO' if out['n_R_ok'] >= 3 else 'MODELO DE RUÍDO NÃO VALIDADO'
    out['decisao_alfa'] = 'CORREÇÃO DE α CONSISTENTE' if out['n_estrutura_ok'] >= 3 else 'CORREÇÃO DE α NÃO CONSISTENTE'
    return out, afins, corr, As, Is, paineis


def mosaico(afins, corr, As, Is, passo_arcsec=0.2):
    ny, nx = As[0].shape
    cent = np.array([A @ np.array([(nx - 1) / 2, (ny - 1) / 2, 1.0]) for A in afins])
    meia = 0.5 * max(ny, nx) * ESCALA + 3
    lo, hi = cent.min(axis=0) - meia, cent.max(axis=0) + meia
    gx = np.arange(lo[0], hi[0], passo_arcsec)
    gy = np.arange(lo[1], hi[1], passo_arcsec)
    SX, SY = np.meshgrid(gx, gy)
    somaA, somaI, n = np.zeros(SX.shape), np.zeros(SX.shape), np.zeros(SX.shape)
    for t, A in enumerate(afins):
        c = corr.get(t, np.zeros(2))
        L, l = A[:, :2], A[:, 2]
        Li = np.linalg.inv(L)
        px = Li[0, 0] * (SX - c[0] - l[0]) + Li[0, 1] * (SY - c[1] - l[1])
        py = Li[1, 0] * (SX - c[0] - l[0]) + Li[1, 1] * (SY - c[1] - l[1])
        a = map_coordinates(np.nan_to_num(As[t], nan=np.nan), [py, px], order=0, mode='constant', cval=np.nan)
        i = map_coordinates(np.nan_to_num(Is[t], nan=np.nan), [py, px], order=1, mode='constant', cval=np.nan)
        ok = np.isfinite(a)
        somaA[ok] += a[ok]
        somaI[ok] += i[ok]
        n[ok] += 1
    with np.errstate(invalid='ignore'):
        return gx, gy, np.where(n > 0, somaA / n, np.nan), np.where(n > 0, somaI / n, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    rel89 = json.loads((REP / '89_sst_mapas_blos.json').read_text(encoding='utf-8'))
    s89 = {r['arquivo']: [p['s_teorico_G'] for p in r['posicoes']] for r in rel89['data']}
    todos, mosaicos, todos_paineis = [], [], []
    for c in sorted(a.arquivos, key=lambda x: Path(x).name):
        print('=' * 78)
        print(Path(c).name)
        r, afins, corr, As, Is, paineis = analisar(c, s89[Path(c).name])
        print(f"  pares com R em 0,8–1,25: {r['n_R_ok']} de 4  →  {r['decisao_ruido']}")
        print(f"  pares com estrutura e mediana < 0,2 σ: {r['n_estrutura_ok']} de 4  →  {r['decisao_alfa']}")
        todos.append(r)
        todos_paineis.append((Path(c).name[8:27], paineis))
        mosaicos.append((Path(c).name[8:27], mosaico(afins, corr, As, Is)))
        del As, Is

    nl = len(todos_paineis)
    fig, axs = plt.subplots(2 * nl, 4, figsize=(14, 6.2 * nl), squeeze=False)
    for k, (rot, pain) in enumerate(todos_paineis):
        for j, (a, b, D, Ds, lin) in enumerate(pain):
            vm = 3 * lin['sigma_dif_G']
            axs[2 * k, j].imshow(D[::2, ::2], origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
            axs[2 * k, j].set_title(f"{rot}  t{a}−t{b}: Δâ  (R = {lin['R']:.2f})", fontsize=7)
            axs[2 * k + 1, j].imshow(Ds[::2, ::2], origin='lower', cmap='RdBu_r', vmin=-vm / 5, vmax=vm / 5)
            axs[2 * k + 1, j].set_title(f"Δâ suavizado 1,6″  ({lin['razao_estrutura']:.3f} σ)", fontsize=7)
            for ax in (axs[2 * k, j], axs[2 * k + 1, j]):
                ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel92_sobreposicao.png', dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(nl, 2, figsize=(13, 6.2 * nl), squeeze=False)
    for k, (rot, (gx, gy, MA, MI)) in enumerate(mosaicos):
        ext = (gx[0], gx[-1], gy[0], gy[-1])
        axs[k, 0].imshow(MI, origin='lower', cmap='gray', vmin=0.7, vmax=1.3, extent=ext)
        axs[k, 0].set_title(f'{rot}  I asas (mosaico)', fontsize=9)
        im = axs[k, 1].imshow(MA, origin='lower', cmap='RdBu_r', vmin=-60, vmax=60, extent=ext)
        axs[k, 1].set_title(f'{rot}  B∥·f (G), ±60 G', fontsize=9)
        for ax in axs[k]:
            ax.set_xlabel('HPLN (″)'); ax.set_ylabel('HPLT (″)')
        fig.colorbar(im, ax=axs[k, 1], shrink=0.8, extend='both')
    fig.tight_layout()
    fig.savefig(FIG / 'cel92_mosaico_blos.png', dpi=100)
    plt.close(fig)

    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '92_sst_sobreposicao_ladrilhos.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos}, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório: {REP / '92_sst_sobreposicao_ladrilhos.json'}\n"
          f"Figuras: {FIG / 'cel92_sobreposicao.png'}\n         {FIG / 'cel92_mosaico_blos.png'}")


if __name__ == '__main__':
    main()
