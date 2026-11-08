"""
84_sst_crosstalk_alfa_xy.py — o gradiente de V simétrico no campo é crosstalk I→V variando no campo, α(x, y)?

Script 83: no interior do campo, S = (V₀ + V₁₄)/2 tem um gradiente leste–oeste de ~±2×10⁻³ I, igual nas
8 posições e nos dois horários. O script 81 só testava α constante contra a granulação (r ≈ 0).
Modelo testado aqui:  V_λ(x, y) ⊃ α(x, y) · I_λ(x, y)   (mesmo α em todos os λ).

Para cada posição do mosaico, no Sol calmo e a ≥ 150 px da borda (máscara do script 83):
  - suavização gaussiana normalizada só sobre a máscara, w = 32 px (G);
  - α(x, y) = G[(V₀ + V₁₄)/2] / G[(I₀ + I₁₄)/2]   (medido só nas asas extremas);
  - para cada par simétrico (k, 14 − k), k = 1…6:
        medido   M_k = G[(V_k + V_{14−k})/2]
        previsto P_k = α · G[(I_k + I_{14−k})/2]
    ajuste M_k = b P_k + c em pixels amostrados a cada 8 px → inclinação b e correlação r;
  - descritivo: amplitude de M_k antes e depois da correção V − α I (σ robusto, em unidades de ⟨I_k⟩).
O α vem só de λ0/λ14; os pares 1–6 são exposições independentes, então o teste não é circular.

Critério (definido antes de rodar):
  par k passa se 0,8 ≤ b ≤ 1,2 e r ≥ 0,7
  posição passa se ≥ 4 dos 6 pares passam E pelo menos um dos pares perto do núcleo (5/9 ou 6/8) passa
    (nos pares 1–3 I ≈ I das asas, então um desvio que não escala com I também passaria; só os pares
     com I bem menor que nas asas distinguem os dois modelos — verificado em teste sintético)
  MODELO α(x, y) CONFIRMADO no arquivo se ≥ 3 das 4 posições passam → correção V − α(x, y) I vale em todos os λ.
  Caso contrário: NÃO CONFIRMADO (a componente simétrica não escala com I; correção não deve ser aplicada).

Uso:  python Scripts/84_sst_crosstalk_alfa_xy.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
Saída: Outputs/Reports/84_sst_crosstalk_alfa_xy.json,
       Outputs/Figures/cel84_modelo_alfa.png e cel84_mapas_alfa.png

Saídas por arquivo (2026-10-26, _saidas_sst.py): relatório <prefixo>_<AAAAMMDDThhmmss>.json e figuras
<nome>_<AAAAMMDDThhmmss>.png, mais <prefixo>_resumo.json com todos os arquivos já analisados.
Critério por arquivo em proporção: "≥ 3 das 4 posições" ≡ ≥ 75% das posições/varreduras (idêntico nos
mosaicos; fixado antes de reexecutar nas séries do CRISP, que têm 14 e 10 varreduras).
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

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'
K_MAD = 1.4826
W = 32
D_BORDA = 150
AMOSTRA = 8
PARES = [(k, 14 - k) for k in range(1, 7)]


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def analisar(caminho):
    out = {'arquivo': Path(caminho).name, 'posicoes': []}
    mapas = []
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data                                      # (t, s, λ, y, x)
        nt, ns, nl = d.shape[:3]
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
            mf = m.astype(np.float64)
            den = gaussian_filter(mf, W, mode='constant')
            usa = m & (den > 0.5)

            def G(X):
                with np.errstate(invalid='ignore', divide='ignore'):
                    return gaussian_filter(np.where(m, X, 0.0), W, mode='constant') / den

            alfa = G((V0 + V14) / 2) / G((I0 + I14) / 2)
            sub = np.zeros_like(usa)
            sub[::AMOSTRA, ::AMOSTRA] = True
            sel = usa & sub
            pos = {'t': t, 'lambda_nucleo_idx': lnuc, 'n_amostra': int(sel.sum()),
                   'alfa_mediana': float(np.median(alfa[usa])),
                   'alfa_p05_p95': [float(np.percentile(alfa[usa], 5)), float(np.percentile(alfa[usa], 95))],
                   'pares': []}
            for (ka, kb) in PARES:
                Ia, Ib, Va, Vb = le(0, ka), le(0, kb), le(3, ka), le(3, kb)
                okp = np.isfinite(Ia) & np.isfinite(Ib) & np.isfinite(Va) & np.isfinite(Vb)
                s2 = sel & okp
                Ip = G((Ia + Ib) / 2)
                M = G((Va + Vb) / 2)
                P = alfa * Ip
                Mc = G((Va - alfa * Ia + Vb - alfa * Ib) / 2)
                b, c = np.polyfit(P[s2], M[s2], 1)
                r = float(np.corrcoef(P[s2], M[s2])[0, 1])
                mI = float(np.mean(Ip[s2]))
                passa = bool(0.8 <= b <= 1.2 and r >= 0.7)
                pos['pares'].append({'par': [ka, kb], 'I_rel_asas': mI / float(np.mean(G((I0 + I14) / 2)[s2])),
                                     'b': float(b), 'c_sobre_I': float(c / mI), 'r': r,
                                     'amp_antes': sigma_rob(M[s2]) / mI, 'amp_depois': sigma_rob(Mc[s2]) / mI,
                                     'passa': passa})
            pos['n_pares_ok'] = sum(p['passa'] for p in pos['pares'])
            nucleo_ok = any(p['passa'] for p in pos['pares'] if p['par'][0] in (5, 6))
            pos['nucleo_ok'] = bool(nucleo_ok)
            pos['passa'] = bool(pos['n_pares_ok'] >= 4 and nucleo_ok)
            out['posicoes'].append(pos)
            mapas.append(np.where(usa, alfa, np.nan)[::8, ::8])
            print(f"  t={t}: α mediano {pos['alfa_mediana']:+.2e}  (5–95%: {pos['alfa_p05_p95'][0]:+.2e} a "
                  f"{pos['alfa_p05_p95'][1]:+.2e})")
            for p in pos['pares']:
                print(f"        par {p['par'][0]:2d}/{p['par'][1]:2d}  I/I_asas {p['I_rel_asas']:.2f}  b = {p['b']:.3f}  "
                      f"r = {p['r']:.3f}  amplitude {p['amp_antes']:.2e} → {p['amp_depois']:.2e}  "
                      f"{'passa' if p['passa'] else 'FALHA'}")
            print(f"        pares que passam: {pos['n_pares_ok']} de 6, perto do núcleo: {'sim' if pos['nucleo_ok'] else 'não'}  →  posição {'passa' if pos['passa'] else 'FALHA'}")
    n_ok = sum(p['passa'] for p in out['posicoes'])
    out['n_posicoes_ok'] = n_ok
    out['decisao'] = 'MODELO α(x,y) CONFIRMADO' if SA.passa_fracao(n_ok, len(out['posicoes'])) else 'NÃO CONFIRMADO'
    return out, mapas


def figuras(todos, todos_mapas, caminhos):
    cores = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for k, r in enumerate(todos):
        ls = '-' if k == 0 else '--'
        n = len(r['posicoes'])
        for p in r['posicoes']:
            x = [q['I_rel_asas'] for q in p['pares']]
            cor = cores[p['t']] if n <= 4 else plt.cm.viridis(p['t'] / max(n - 1, 1))   # séries: cor = tempo
            mk = 'o' if p['passa'] else 'x'
            axs[0].plot(x, [q['b'] for q in p['pares']], ls=ls, marker=mk, ms=4, color=cor,
                        label=f"{r['arquivo'][8:27]} t={p['t']}" + ('' if p['passa'] else ' (falha)'))
            axs[1].plot(x, [q['r'] for q in p['pares']], ls=ls, marker=mk, ms=4, color=cor)
    axs[0].set_title(f"{todos[0]['arquivo'][8:27]}: {todos[0]['n_posicoes_ok']}/{len(todos[0]['posicoes'])} "
                     f"→ {todos[0]['decisao']}", fontsize=9)
    axs[0].axhspan(0.8, 1.2, color='0.9', zorder=0); axs[0].axhline(1, color='0.5', lw=1)
    axs[1].axhline(0.7, color='0.3', ls='--', lw=1)
    for ax in axs:
        ax.set_xlabel('⟨I⟩ do par / ⟨I⟩ das asas  (núcleo à esquerda)')
    axs[0].set_ylabel('inclinação b (medido / previsto)'); axs[1].set_ylabel('correlação r')
    axs[0].legend(fontsize=6, ncol=2 if len(todos[0]['posicoes']) > 7 else 1)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminhos[0], dpi=120)
    plt.close(fig)

    nlin = len(todos)
    fig, axs = plt.subplots(nlin, 4, figsize=(12, 3.2 * nlin), squeeze=False)
    vm = np.nanmax([np.nanpercentile(np.abs(mp), 99) for ms in todos_mapas for mp in ms])
    for k, (r, ms) in enumerate(zip(todos, todos_mapas)):
        idx = list(range(len(ms))) if len(ms) <= 4 else [int(round(x)) for x in np.linspace(0, len(ms) - 1, 4)]
        for j, t in enumerate(idx):                     # séries temporais: 4 varreduras igualmente espaçadas
            im = axs[k, j].imshow(ms[t] * 1e3, origin='lower', cmap='RdBu_r', vmin=-vm * 1e3, vmax=vm * 1e3)
            axs[k, j].set_title(f"{r['arquivo'][8:27]}  t={t}", fontsize=8)
            axs[k, j].set_xticks([]); axs[k, j].set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label='α(x, y)  (×10⁻³)')
    fig.savefig(caminhos[1], dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    todos, todos_mapas = [], []
    for c in a.arquivos:
        print('=' * 78)
        print(Path(c).name)
        r, mapas = analisar(c)
        todos.append(r)
        todos_mapas.append(mapas)
        print(f"  posições/varreduras que passam: {r['n_posicoes_ok']} de {len(r['posicoes'])} "
              f"(mínimo {SA.FRAC_MIN:.0%})  →  {r['decisao']}")
        rel = SA.gravar('84_sst_crosstalk_alfa_xy', r)           # relatório antes das figuras
        cfs = [SA.figura(n, r['etiqueta']) for n in ('cel84_modelo_alfa', 'cel84_mapas_alfa')]
        figuras([r], [mapas], cfs)
        print(f"  Relatório: {rel}\n  Figuras: {cfs[0]}\n           {cfs[1]}")
    SA.resumo('84_sst_crosstalk_alfa_xy')


if __name__ == '__main__':
    main()
