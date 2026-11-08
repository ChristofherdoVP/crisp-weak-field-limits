"""
96_sst_ruido_series_temporais.py — ruído de Stokes V em cada λ por diferença entre varreduras consecutivas
(séries temporais CRISP 2024-10-08) e teste do modelo de ruído do artigo B.

Modelo do artigo B (Src/physics.py, 'CRISP/SST'): σ_V = ε · I local em todos os λ, ε = 3,2×10⁻³ (medido em
CRISP 2024-07-08). Script 83: nas asas extremas (λ0/λ14) os dados de 2024-10-08 dão ε = 3,23–3,29×10⁻³.
Script 95: NSUMEXP = 40 quadros em λ0, λ1, λ13, λ14 e 64 nos demais → se o ruído vai com 1/√N, os λ internos
teriam ~0,79× o ruído relativo das asas extremas, o que o modelo do B não inclui.

Para cada série e cada par de varreduras consecutivas (t, t+1), 27,7 s de intervalo, mesma posição no detector:
  - alinhamento: correlação cruzada de I das asas (passa-alta) numa região central de 1024 × 1024 px;
    deslocamento inteiro aplicado se ≠ 0 (sem interpolar); deslocamento subpixel reportado;
  - máscara: campo válido nas duas, ≥ 150 px da borda, "Sol calmo" pela intensidade (scripts 79–94);
  - ΔV_k = (V_k(t+1) − V_k(t)) / ⟨I asas⟩, passa-alta (subtrai a suavização gaussiana de 32 px);
  - pares espelhados (k, 14 − k): σ₊,k = σ(ΔV_k + ΔV_{14−k})/2 (sinal de campo fraco se cancela → ruído por
    varredura), σ₋,k = σ(ΔV_k − ΔV_{14−k})/2; núcleo (k = 7): σ(ΔV_7)/√2;
  - ruído relativo à intensidade local: ε_k = σ_k / Ĩ_k (Ĩ_k = ⟨I_k⟩/⟨I asas⟩ no Sol calmo).
Valores da série = mediana sobre os pares de varreduras.

Critérios (definidos antes de rodar), por série:
  MÉTODO VÁLIDO se a mediana do |deslocamento subpixel residual| < 0,3 px E a mediana de σ₋/σ₊ nos pares das
    asas (0/14, 1/13, 2/12) está entre 0,9 e 1,1;
  MODELO DO B CONFIRMADO se ε_k está dentro de ±15% de 3,2×10⁻³ em ≥ 12 dos 15 λ.
  Decisão final: MODELO DO B CONFIRMADO só se as duas séries forem válidas e confirmarem.
Descritivo: ajuste σ_k = ε₀ Ĩ_k^p √(64/N_k) (com N de NSUMEXP) e, para comparar, sem o fator de N;
ε das asas por par de varreduras (variação no tempo).

Uso:  python Scripts/96_sst_ruido_series_temporais.py Data/RAW_SST/nb_8542_2024*.fits
Saída: Outputs/Reports/96_sst_ruido_series_temporais.json, Outputs/Figures/cel96_ruido_series.png
"""
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt, binary_fill_holes, shift as nd_shift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'
EPS_B = 3.2e-3
D_BORDA, W_HP = 150, 32


def carregar(nome):
    esp = importlib.util.spec_from_file_location(nome.replace('.py', ''), ROOT / 'Scripts' / nome)
    m = importlib.util.module_from_spec(esp)
    esp.loader.exec_module(m)
    return m


S92 = carregar('92_sst_sobreposicao_ladrilhos.py')
sigma_rob, suave_nan = S92.sigma_rob, S92.suave_nan


def nsumexp(h, nl):
    try:
        hd = h['VAR-EXT-NSUMEXP']
        col = 'VALUES' if 'VALUES' in hd.columns.names else hd.columns.names[0]   # como no script 95
        v = np.asarray(hd.data[col][0], dtype=float)
        ax = list(v.shape).index(nl)
        return np.nanmean(np.moveaxis(v, ax, 0).reshape(nl, -1), axis=1)
    except Exception:
        return None


def desloc_subpixel(A, B):
    """Deslocamento (dy, dx) que leva B sobre A: pico da correlação cruzada FFT + ajuste parabólico."""
    a = np.where(np.isfinite(A), A - np.nanmean(A), 0.0)
    b = np.where(np.isfinite(B), B - np.nanmean(B), 0.0)
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b)))))
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    def par(m1, m0, p1):
        den = m1 - 2 * m0 + p1
        return 0.0 if den == 0 else 0.5 * (m1 - p1) / den
    fy = par(c[iy - 1, ix], c[iy, ix], c[iy + 1, ix])
    fx = par(c[iy, ix - 1], c[iy, ix], c[iy, ix + 1])
    cy, cx = c.shape[0] // 2, c.shape[1] // 2
    return iy - cy, ix - cx, fy, fx


def mascara(I0, I14, Ic, V0):
    ok = np.isfinite(I0) & (I0 != 0) & np.isfinite(I14) & np.isfinite(Ic) & np.isfinite(V0) & (V0 != 0)
    interior = distance_transform_edt(binary_fill_holes(ok)) >= D_BORDA
    calmo = ok & interior & (I0 / np.median(I0[ok]) > 0.92) & (I0 / np.median(I0[ok]) < 1.08) \
        & (Ic / np.median(Ic[ok]) < 1.15)
    return calmo


def analisar(caminho):
    nome = Path(caminho).name
    out = {'arquivo': nome, 'pares_varreduras': []}
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt, _, nl, ny, nx = d.shape
        N = nsumexp(h, nl)
        out['NSUMEXP'] = None if N is None else N.tolist()
        amostra = np.array([np.asarray(d[0, 0, l, ::8, ::8], dtype=np.float32) for l in range(nl)])
        okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
        lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(nl)]))
        cy, cx = ny // 2, nx // 2
        rc = (slice(cy - 512, cy + 512), slice(cx - 512, cx + 512))
        ant = None
        for t in range(nt):
            I = [np.asarray(d[t, 0, l], dtype=np.float32) for l in range(nl)]
            V = [np.asarray(d[t, 3, l], dtype=np.float32) for l in range(nl)]
            m = mascara(I[0], I[14], I[lnuc], V[0])
            Iw = (I[0] + I[14]) / 2
            atual = {'I': I, 'V': V, 'm': m, 'Iw': Iw}
            if ant is not None:
                A = ant['Iw'][rc].astype(float)
                B = Iw[rc].astype(float)
                hp = lambda X: X - suave_nan(X, 20)
                dy, dx, fy, fx = desloc_subpixel(hp(A), hp(B))
                desl = lambda X: nd_shift(X, (dy, dx), order=0, mode='constant', cval=np.nan) if (dy or dx) else X
                mb = desl(m.astype(float)) > 0.5
                mm = ant['m'] & mb
                Iwm = float(np.mean(ant['Iw'][mm]))
                Imed = np.array([np.mean(ant['I'][l][mm]) for l in range(nl)]) / Iwm
                dV = []
                for l in range(nl):
                    x = np.where(mm, (desl(V[l]).astype(float) - ant['V'][l].astype(float)) / Iwm, np.nan)
                    dV.append(x - suave_nan(x, W_HP))
                sub = np.zeros_like(mm)
                sub[::2, ::2] = True
                sel = mm & sub
                sig = np.full(nl, np.nan)
                sm = np.full(nl, np.nan)
                for k in range(8):
                    j = 14 - k
                    ok = sel & np.isfinite(dV[k]) & np.isfinite(dV[j])
                    if j != k:
                        s_p = sigma_rob((dV[k] + dV[j])[ok]) / 2
                        s_m = sigma_rob((dV[k] - dV[j])[ok]) / 2
                        sig[k] = sig[j] = s_p
                        sm[k] = sm[j] = s_m
                    else:
                        sig[k] = sigma_rob(dV[k][ok]) / np.sqrt(2)
                lin = {'t': [t - 1, t], 'desloc_int_px': [int(dy), int(dx)], 'desloc_subpx': [float(fy), float(fx)],
                       'Imed': Imed.tolist(), 'sigma': sig.tolist(), 'sigma_menos': sm.tolist(),
                       'razao_menos_mais_asas': float(np.nanmedian([sm[k] / sig[k] for k in (0, 1, 2)])),
                       'eps_asas': float(sig[0] / Imed[0])}
                out['pares_varreduras'].append(lin)
                print(f"  varreduras {t - 1}→{t}: desloc. ({dy:+d}, {dx:+d}) px, resíduo ({fy:+.2f}, {fx:+.2f});  "
                      f"σ₋/σ₊ asas {lin['razao_menos_mais_asas']:.3f};  ε asas {lin['eps_asas']:.2e}")
            ant = atual
    P = out['pares_varreduras']
    Imed = np.median([p['Imed'] for p in P], axis=0)
    sig = np.median([p['sigma'] for p in P], axis=0)
    eps_k = sig / Imed
    razao = eps_k / EPS_B
    dentro = np.abs(razao - 1) <= 0.15
    res = float(np.median([np.hypot(*p['desloc_subpx']) for p in P]))
    rmm = float(np.median([p['razao_menos_mais_asas'] for p in P]))
    valido = res < 0.3 and 0.9 <= rmm <= 1.1
    confirma = int(dentro.sum()) >= 12
    out.update({'Imed': Imed.tolist(), 'sigma_k': sig.tolist(), 'eps_k': eps_k.tolist(), 'razao_eps_B': razao.tolist(),
                'n_lambda_dentro_15pc': int(dentro.sum()), 'residuo_subpx_mediano': res,
                'razao_menos_mais_asas_mediana': rmm, 'metodo_valido': bool(valido),
                'confirma_B': bool(valido and confirma)})
    # ajustes descritivos
    lnI = np.log(Imed)
    p_sem = np.polyfit(lnI, np.log(sig), 1)
    out['ajuste_sem_N'] = {'p': float(p_sem[0]), 'eps0': float(np.exp(p_sem[1]))}
    if N is not None:
        y = np.log(sig) - 0.5 * np.log(64.0 / N)
        p_com = np.polyfit(lnI, y, 1)
        resid = y - np.polyval(p_com, lnI)
        out['ajuste_com_N'] = {'p': float(p_com[0]), 'eps0_N64': float(np.exp(p_com[1])), 'rms_resid': float(np.std(resid))}
        out['ajuste_sem_N']['rms_resid'] = float(np.std(np.log(sig) - np.polyval(p_sem, lnI)))
    print('  ' + '-' * 74)
    print('   λ    Ĩ_k    N    σ_k (×10⁻³)   ε_k = σ_k/Ĩ_k (×10⁻³)   ε_k / 3,2e-3')
    for l in range(len(sig)):
        nn = '—' if N is None else f'{N[l]:.0f}'
        print(f"  {l:2d}  {Imed[l]:.3f}  {nn:>3}     {sig[l] * 1e3:.3f}            {eps_k[l] * 1e3:.3f}              "
              f"{razao[l]:.2f} {'✓' if dentro[l] else '✗'}")
    print(f"  resíduo subpixel mediano {res:.2f} px;  σ₋/σ₊ asas (mediana) {rmm:.3f}  →  método "
          f"{'VÁLIDO' if valido else 'INVÁLIDO'}")
    print(f"  λ com ε_k a ±15% de 3,2e-3: {int(dentro.sum())} de {len(sig)}  →  "
          f"{'confirma o modelo do B' if (valido and confirma) else 'NÃO confirma o modelo do B'}")
    a = out['ajuste_sem_N']
    print(f"  ajuste sem N: σ = {a['eps0']:.2e} Ĩ^{a['p']:.2f}" + (f"  (rms {a['rms_resid']:.3f})" if 'rms_resid' in a else ''))
    if 'ajuste_com_N' in out:
        b = out['ajuste_com_N']
        print(f"  ajuste com N: σ = {b['eps0_N64']:.2e} Ĩ^{b['p']:.2f} √(64/N)  (rms {b['rms_resid']:.3f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    todos = []
    for c in sorted(a.arquivos, key=lambda x: Path(x).name):
        print('=' * 78)
        print(Path(c).name)
        todos.append(analisar(c))
    ok = all(r['confirma_B'] for r in todos)
    dec = 'MODELO DO B CONFIRMADO' if ok else 'MODELO DO B NÃO CONFIRMADO'
    print('=' * 78)
    print(f'Decisão: {dec}')
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '96_sst_ruido_series_temporais.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos, 'decisao': dec, 'eps_B': EPS_B}, f, ensure_ascii=False, indent=2)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    cores = ['#0072B2', '#D55E00', '#009E73']
    for i, r in enumerate(todos):
        c = cores[i % 3]
        lab = r['arquivo'][8:27]
        x = np.arange(len(r['eps_k']))
        axs[0].plot(x, np.array(r['eps_k']) * 1e3, 'o-', color=c, label=lab)
        axs[1].loglog(r['Imed'], np.array(r['sigma_k']) * 1e3, 'o', color=c, label=lab)
        axs[2].plot([p['eps_asas'] * 1e3 for p in r['pares_varreduras']], 'o-', color=c, label=lab)
    axs[0].axhline(EPS_B * 1e3, color='0.3', lw=1, label='artigo B')
    axs[0].axhspan(EPS_B * 0.85e3, EPS_B * 1.15e3, color='0.9', zorder=0)
    axs[0].set_xlabel('índice de λ'); axs[0].set_ylabel('ε_k = σ_k / Ĩ_k  (×10⁻³)'); axs[0].legend(fontsize=7)
    axs[1].set_xlabel('Ĩ_k'); axs[1].set_ylabel('σ_k (×10⁻³, em I asas)')
    axs[2].axhline(EPS_B * 1e3, color='0.3', lw=1)
    axs[2].set_xlabel('par de varreduras'); axs[2].set_ylabel('ε nas asas extremas (×10⁻³)')
    fig.suptitle(dec, fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel96_ruido_series.png', dpi=120)
    plt.close(fig)
    print(f"\nRelatório: {REP / '96_sst_ruido_series_temporais.json'}\nFigura: {FIG / 'cel96_ruido_series.png'}")


if __name__ == '__main__':
    main()
