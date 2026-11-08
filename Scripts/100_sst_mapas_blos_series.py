"""
100_sst_mapas_blos_series.py — mapas de B∥·f das séries CRISP 2024-10-08 e validação do ruído dos mapas
por diferença entre varreduras consecutivas.

Produto: mapa de â (filtro casado de campo fraco, artigo B, Eqs. 1–4) por varredura, como no script 89, com:
  - borda: exclui pixels a < 150 px da borda do campo válido (83);
  - SEM correção de crosstalk α: script 99 → viés desprezível nas duas séries (P95|Δâ| ≤ 0,054 s;
    sistemática ≤ 0,9 G às 09:49 e ≤ 1,2 G às 10:10);
  - sem quadros excluídos (nenhum identificado nas séries; candidatos nas asas do script 80 não testados);
  - template t_k = −C g_eff λ₀² dĨ/dλ do perfil médio de Sol calmo da varredura; σ_k = ε Ĩ_k com ε lido do
    relatório do script 83 do mesmo arquivo; u_k = t_k/σ_k²; s = (Σ u_k t_k)^(−½) = desvio teórico por pixel.

Validação do ruído (critério definido antes de rodar), por par de varreduras consecutivas (t, t+1):
  - alinhamento como no script 96 (correlação cruzada de I das asas passa-alta, região central 1024 px;
    deslocamento inteiro aplicado sem interpolar); pré-requisito: mediana do |resíduo subpixel| < 0,3 px;
  - máscara: "Sol calmo" das duas varreduras (I, ≥ 150 px da borda);
  - Δâ = â(t+1) − â(t), passa-alta (subtrai G₃₂ normalizada na máscara) para tirar evolução lenta e seeing;
  - R = σ_rob(Δâ) / √(s_t² + s_{t+1}²).
  RUÍDO DOS MAPAS VALIDADO se 0,8 ≤ R ≤ 1,25 em ≥ 75% dos pares → s teórico vale como desvio por pixel;
  senão NÃO VALIDADO → o desvio por pixel a usar é o empírico σ_rob(Δâ)/√2 (reportado).
  (Evolução solar em ~28 s nas escalas < 32 px soma em Δâ: tende a aumentar R, principalmente nas 10:10.)
Descritivo: s, a₅₀ = 1,96 s, σ(â) no Sol calmo, P1–P99 de â, fração do Sol calmo com |â| > 1,96 s.
Ressalvas: template de Sol calmo em toda a varredura (rede/plage têm outro perfil); deslocamento de λ no
campo ignorado; nos poros o campo fraco provavelmente subestima o campo.

Uso:  python Scripts/100_sst_mapas_blos_series.py Data/RAW_SST/nb_8542_2024-10-08*.fits
Saída: Outputs/Data/blos_f_crisp_<AAAAMMDDThhmmss>_t<k>.npy (float32, G, NaN fora da área válida),
       Outputs/Reports/100_sst_mapas_blos_series_<…>.json (+ _resumo.json), Outputs/Figures/cel100_mapas_blos_<…>.png
"""
import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes, shift as nd_shift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saidas_sst as SA

PREFIXO = '100_sst_mapas_blos_series'
DAT = SA.ROOT / 'Outputs' / 'Data'
K_MAD = 1.4826
W = 32
D_BORDA = 150
C_ZEEMAN = 4.6686e-13
G_EFF = 1.10
LAMBDA0 = 8542.09
Z50 = 1.96
R_MIN, R_MAX = 0.8, 1.25


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


def suave_mask(X, m, w):
    den = gaussian_filter(m.astype(np.float64), w, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        return gaussian_filter(np.where(m, X, 0.0), w, mode='constant') / den


def desloc_subpixel(A, B):
    """Deslocamento (dy, dx) que leva B sobre A (idêntico ao script 96)."""
    if not (np.isfinite(A).any() and np.isfinite(B).any()):
        raise SystemExit('Região central sem dados válidos para o alinhamento.')
    a = np.where(np.isfinite(A), A - np.nanmean(A), 0.0)
    b = np.where(np.isfinite(B), B - np.nanmean(B), 0.0)
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b)))))
    iy, ix = np.unravel_index(np.argmax(c), c.shape)

    def par(m1, m0, p1):
        den = m1 - 2 * m0 + p1
        return 0.0 if den == 0 else 0.5 * (m1 - p1) / den
    fy = par(c[iy - 1, ix], c[iy, ix], c[iy + 1, ix])
    fx = par(c[iy, ix - 1], c[iy, ix], c[iy, ix + 1])
    return iy - c.shape[0] // 2, ix - c.shape[1] // 2, fy, fx


def processar(caminho):
    nome = Path(caminho).name
    tag = SA.etiqueta(nome)
    p83 = SA.REP / f'83_sst_ruido_v_sem_borda_{tag}.json'
    if not p83.exists():
        raise SystemExit(f'{nome}: falta {p83.name} (rode o 83).')
    r83 = json.loads(p83.read_text(encoding='utf-8'))['data'][0]
    if not r83.get('eps_V'):
        raise SystemExit(f'{nome}: o 83 reprovou este arquivo; sem ε.')
    eps = float(r83['eps_V']['mediano'])
    out = {'arquivo': nome, 'eps_V': eps, 'varreduras': [], 'pares': []}
    figs = []
    print(f"  ε (script 83) = {eps:.2e}")
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt, ns, nl, ny, nx = d.shape
        lam = ler_grade(h, nl)
        if lam is None:
            raise SystemExit(f'{nome}: grade λ não lida do WCS-TAB.')
        cy, cx = ny // 2, nx // 2
        hy, hx = min(512, cy), min(512, cx)
        rc = (slice(cy - hy, cy + hy), slice(cx - hx, cx + hx))
        ant = None
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
            Iw = (I0 + I14) / 2
            Iw_med = float(np.mean(Iw[m]))
            Imed = np.array([np.mean(le(0, l)[m]) for l in range(nl)]) / Iw_med
            tk = -C_ZEEMAN * G_EFF * LAMBDA0 ** 2 * np.gradient(Imed, lam)
            uk = tk / (eps * Imed) ** 2
            D = float(np.sum(uk * tk))
            s = 1.0 / np.sqrt(D)
            num = np.zeros((ny, nx))
            fin = np.ones((ny, nx), dtype=bool)
            for l in range(nl):
                Vl = le(3, l)
                fin &= np.isfinite(Vl)
                num += uk[l] * np.nan_to_num(Vl) / Iw_med
            area = interior & ok & fin
            ahat = np.where(area, num / D, np.nan)
            DAT.mkdir(parents=True, exist_ok=True)
            arq = DAT / f'blos_f_crisp_{tag}_t{t}.npy'
            np.save(arq, ahat.astype(np.float32))
            calmo = m & np.isfinite(ahat)
            v = {'t': t, 'arquivo_mapa': str(arq), 's_teorico_G': s, 'a50_G': Z50 * s,
                 'sigma_a_calmo_G': sigma_rob(ahat[calmo]),
                 'frac_area_valida': float(area.sum() / max(ok.sum(), 1)),
                 'P01_P99_G': [float(np.nanpercentile(ahat, 1)), float(np.nanpercentile(ahat, 99))],
                 'frac_calmo_acima_a50': float(np.mean(np.abs(ahat[calmo]) > Z50 * s))}
            out['varreduras'].append(v)
            print(f"  t={t}: s = {s:.1f} G (a₅₀ ≈ {Z50 * s:.1f} G);  σ(â) Sol calmo {v['sigma_a_calmo_G']:.1f} G;  "
                  f"área válida {v['frac_area_valida']:.0%};  â P1–P99 {v['P01_P99_G'][0]:+.0f} a {v['P01_P99_G'][1]:+.0f} G")
            if ant is not None:
                hp = lambda X: X - suave_mask(np.nan_to_num(X), np.isfinite(X), 20)
                dy, dx, fy, fx = desloc_subpixel(hp(ant['Iw'][rc]), hp(Iw[rc]))
                desl = (lambda X: nd_shift(X, (dy, dx), order=0, mode='constant', cval=np.nan)) if (dy or dx) else (lambda X: X)
                a1 = desl(ahat)
                mb = desl(m.astype(float)) > 0.5
                mm = ant['m'] & mb & np.isfinite(a1) & np.isfinite(ant['a'])
                da = np.where(mm, a1 - ant['a'], np.nan)
                da_hp = np.where(mm, da - suave_mask(np.nan_to_num(da), mm, W), np.nan)
                den = gaussian_filter(mm.astype(np.float64), W, mode='constant')
                usa = mm & (den > 0.5)
                sd = sigma_rob(da_hp[usa])
                R = sd / np.sqrt(ant['s'] ** 2 + s ** 2)
                p = {'t': [t - 1, t], 'desloc_px': [int(dy), int(dx)], 'residuo_subpixel_px': float(np.hypot(fy, fx)),
                     'n_pixels': int(usa.sum()), 'sigma_delta_a_G': sd, 'sigma_empirico_por_mapa_G': sd / np.sqrt(2),
                     'R': float(R), 'passa': bool(R_MIN <= R <= R_MAX)}
                out['pares'].append(p)
                print(f"        par {t - 1}→{t}: desloc. ({dy:+d}, {dx:+d}) px, resíduo {p['residuo_subpixel_px']:.2f} px;  "
                      f"σ(Δâ) = {sd:.1f} G  →  R = {R:.2f}  ({'passa' if p['passa'] else 'FALHA'});  "
                      f"σ empírico por mapa {sd / np.sqrt(2):.1f} G")
            if t in ([0, nt // 3, 2 * nt // 3, nt - 1] if nt > 4 else range(nt)):
                figs.append((t, (Iw / Iw_med)[::4, ::4], ahat[::4, ::4], s))
            ant = {'Iw': Iw, 'm': m, 'a': ahat, 's': s}
    pares = out['pares']
    n = len(pares)
    res_med = float(np.median([p['residuo_subpixel_px'] for p in pares])) if n else None
    n_ok = sum(p['passa'] for p in pares)
    out['residuo_subpixel_mediano_px'] = res_med
    out['alinhamento_valido'] = bool(res_med is not None and res_med < 0.3)
    out['n_posicoes_ok'] = n_ok
    out['R_mediano'] = float(np.median([p['R'] for p in pares])) if n else None
    out['sigma_empirico_mediano_G'] = float(np.median([p['sigma_empirico_por_mapa_G'] for p in pares])) if n else None
    out['s_teorico_mediano_G'] = float(np.median([v['s_teorico_G'] for v in out['varreduras']]))
    if not out['alinhamento_valido']:
        out['decisao'] = 'VALIDAÇÃO IMPOSSÍVEL (alinhamento)'
    elif SA.passa_fracao(n_ok, n):
        out['decisao'] = 'RUÍDO DOS MAPAS VALIDADO (s teórico)'
    else:
        out['decisao'] = 'RUÍDO DOS MAPAS NÃO VALIDADO (usar σ empírico)'
    return out, figs


def figura(r, figs, caminho):
    n = len(figs)
    fig, axs = plt.subplots(2, n + 1, figsize=(3.4 * (n + 1), 6.4), squeeze=False,
                            gridspec_kw={'width_ratios': [1] * n + [1.3]})
    s_med = r['s_teorico_mediano_G']
    vm = float(np.ceil(6 * s_med / 10) * 10)
    for j, (t, Iw, ah, s) in enumerate(figs):
        axs[0, j].imshow(Iw, origin='lower', cmap='gray', vmin=0.8, vmax=1.2)
        axs[0, j].set_title(f"{r['arquivo'][8:27]}  t={t}  I asas", fontsize=7)
        im = axs[1, j].imshow(ah, origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
        axs[1, j].set_title(f'B∥·f  (s = {s:.0f} G)', fontsize=7)
        for ax in (axs[0, j], axs[1, j]):
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs[1, n - 1], shrink=0.8, extend='both', label=f'B∥·f (G), ±{vm:.0f} G')
    ax = axs[0, n]
    x = np.arange(len(r['pares']))
    ax.bar(x, [p['R'] for p in r['pares']], color=['#0072B2' if p['passa'] else '#D55E00' for p in r['pares']],
           edgecolor='white')
    ax.axhspan(R_MIN, R_MAX, color='0.9', zorder=0)
    ax.set_xlabel('par de varreduras (t → t+1)'); ax.set_ylabel('R = σ(Δâ) / √(s_t² + s_t+1²)')
    ax.set_title(r['decisao'], fontsize=8)
    ax = axs[1, n]
    ax.plot([v['t'] for v in r['varreduras']], [v['s_teorico_G'] for v in r['varreduras']], 'o-', color='#0072B2',
            label='s teórico')
    ax.plot([p['t'][1] for p in r['pares']], [p['sigma_empirico_por_mapa_G'] for p in r['pares']], 's--',
            color='#D55E00', label='σ(Δâ)/√2 (empírico)')
    ax.set_xlabel('varredura (t)'); ax.set_ylabel('desvio por pixel (G)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(caminho, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    for c in a.arquivos:
        print('=' * 78)
        print(Path(c).name)
        r, figs = processar(c)
        print(f"  alinhamento: resíduo subpixel mediano {r['residuo_subpixel_mediano_px']:.2f} px  "
              f"({'válido' if r['alinhamento_valido'] else 'INVÁLIDO'})")
        print(f"  pares com 0,8 ≤ R ≤ 1,25: {r['n_posicoes_ok']} de {len(r['pares'])}  (R mediano {r['R_mediano']:.2f})"
              f"  →  {r['decisao']}")
        print(f"  s teórico mediano {r['s_teorico_mediano_G']:.1f} G;  σ empírico por mapa (mediana) "
              f"{r['sigma_empirico_mediano_G']:.1f} G")
        rel = SA.gravar(PREFIXO, r)                          # relatório antes da figura
        cf = SA.figura('cel100_mapas_blos', r['etiqueta'])
        figura(r, figs, cf)
        print(f"  Relatório: {rel}\n  Figura: {cf}\n  Mapas: {DAT / ('blos_f_crisp_' + r['etiqueta'] + '_t<k>.npy')}")
    SA.resumo(PREFIXO, campos_fora=('varreduras', 'pares'))


if __name__ == '__main__':
    main()
