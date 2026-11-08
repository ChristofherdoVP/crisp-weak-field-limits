"""
86_sst_alfa_mestre_por_horario.py — mapa mestre de α(x, y) por horário, com validação deixando uma posição de fora.

Script 85: α não é o mesmo entre 08:01 e 10:48 (r = 0,54–0,90; NÃO VÁLIDO), mas dentro de cada horário
as 4 posições do mosaico, que olham céus diferentes, têm r = 0,99–1,00. Então α é instrumental e fixo
no detector durante uma observação. Um mapa por horário (mediana das 4 posições) cobre também os pixels de
rede, que a máscara de Sol calmo exclui em cada posição.

Para cada arquivo, separadamente (máscara idêntica aos scripts 84/85: Sol calmo, ≥ 150 px da borda, w = 32 px):
  α_t = G[(V₀ + V₁₄)/2] / G[(I₀ + I₁₄)/2]
  validação deixando uma de fora: mestre_(−t) = mediana de α nas outras 3 posições, aplicado à posição t:
        razão = σ(G[S − mestre_(−t)·I_w]) / σ(G[S])
  referência (descritivo): a mesma razão com o próprio α_t; correlação entre mestre_(−t) e α_t.

Critério (definido antes de rodar), por arquivo:
  razão ≤ 0,2 em ≥ 3 das 4 posições → MAPA MESTRE DO HORÁRIO VÁLIDO; senão NÃO VÁLIDO.
Se VÁLIDO, salva o mestre das 4 posições, com buracos preenchidos por suavização normalizada (w = 64 px):
  Outputs/Data/alfa_mestre_crisp2_<data>T<hora>.npy  (float32, forma do cubo, NaN fora do campo)

Uso:  python Scripts/86_sst_alfa_mestre_por_horario.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
Saída: Outputs/Reports/86_sst_alfa_mestre_por_horario.json, Outputs/Figures/cel86_alfa_mestre_por_horario.png
"""
import json
import argparse
import warnings
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
LIMIAR = 0.2


def sigma_rob(x):
    return float(K_MAD * np.median(np.abs(x - np.median(x))))


def ler_posicao(d, t):
    nl = d.shape[2]
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

    def G(X):
        with np.errstate(invalid='ignore', divide='ignore'):
            return gaussian_filter(np.where(m, X, 0.0), W, mode='constant') / den

    return {'usa': usa, 'G': G, 'S': (V0 + V14) / 2, 'Iw': (I0 + I14) / 2}


def alfa_de(p):
    return np.where(p['usa'], p['G'](p['S']) / p['G'](p['Iw']), np.nan).astype(np.float32)


def razao(p, alfa_m):
    ok = p['usa'] & np.isfinite(alfa_m)
    sub = np.zeros_like(ok)
    sub[::4, ::4] = True
    s = ok & sub
    antes = p['G'](p['S'])
    depois = p['G'](p['S'] - np.nan_to_num(alfa_m.astype(np.float64)) * p['Iw'])
    return sigma_rob(depois[s]) / sigma_rob(antes[s]), float(s.sum() / max((p['usa'] & sub).sum(), 1))


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    sub = np.zeros_like(ok)
    sub[::4, ::4] = True
    ok &= sub
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 1000 else None


def mediana(mapas):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(np.stack(mapas), axis=0).astype(np.float32)


def preencher(mapa):
    campo = np.isfinite(mapa)
    w = gaussian_filter(campo.astype(np.float64), W_PREENCHE, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        suave = gaussian_filter(np.where(campo, mapa, 0.0), W_PREENCHE, mode='constant') / w
    return np.where(campo, mapa, np.where(w > 0.2, suave, np.nan)).astype(np.float32)


def analisar(caminho):
    nome = Path(caminho).name
    out = {'arquivo': nome, 'posicoes': []}
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt = d.shape[0]
        alfas = []
        for t in range(nt):
            alfas.append(alfa_de(ler_posicao(d, t)))
        for t in range(nt):
            p = ler_posicao(d, t)
            mestre_menos_t = mediana([alfas[j] for j in range(nt) if j != t])
            rx, cob = razao(p, mestre_menos_t)
            rp, _ = razao(p, alfas[t])
            r = corr(mestre_menos_t, alfas[t])
            passa = bool(rx <= LIMIAR)
            out['posicoes'].append({'t': t, 'razao_mestre_sem_t': rx, 'razao_proprio_alfa': rp,
                                    'r_mestre_vs_proprio': r, 'cobertura': cob, 'passa': passa})
            print(f'  t={t}: razão com mestre das outras 3 = {rx:.3f}   (próprio α: {rp:.3f};  r = {r:.3f};  '
                  f'cobertura {cob:.0%})  {"passa" if passa else "FALHA"}')
    n_ok = sum(p['passa'] for p in out['posicoes'])
    out['n_posicoes_ok'] = n_ok
    out['decisao'] = 'MAPA MESTRE DO HORÁRIO VÁLIDO' if n_ok >= 3 else 'NÃO VÁLIDO'
    todos = mediana(alfas)
    final = preencher(todos)
    out['cobertura_mestre_sem_preencher'] = float(np.isfinite(todos).mean())
    out['cobertura_mestre_preenchido'] = float(np.isfinite(final).mean())
    out['alfa_mestre_mediana'] = float(np.nanmedian(final))
    out['alfa_mestre_p05_p95'] = [float(np.nanpercentile(final, 5)), float(np.nanpercentile(final, 95))]
    out['mapa_mestre'] = None
    if out['decisao'].endswith('VÁLIDO') and not out['decisao'].startswith('NÃO'):
        DAT.mkdir(parents=True, exist_ok=True)
        c = DAT / f'alfa_mestre_crisp2_{nome[8:18].replace("-", "")}T{nome[19:27].replace("_", "")}.npy'
        np.save(c, final)
        out['mapa_mestre'] = str(c)
    return out, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    todos, mapas = [], []
    for c in sorted(a.arquivos, key=lambda x: Path(x).name):
        print('=' * 78)
        print(Path(c).name)
        r, final = analisar(c)
        todos.append(r)
        mapas.append(final)
        print(f"  posições com razão ≤ {LIMIAR}: {r['n_posicoes_ok']} de {len(r['posicoes'])}  →  {r['decisao']}")
        print(f"  mestre: cobertura {r['cobertura_mestre_sem_preencher']:.0%} do quadro (preenchido: "
              f"{r['cobertura_mestre_preenchido']:.0%});  α mediana {r['alfa_mestre_mediana']:+.2e}, 5–95%: "
              f"{r['alfa_mestre_p05_p95'][0]:+.2e} a {r['alfa_mestre_p05_p95'][1]:+.2e}")
        if r['mapa_mestre']:
            print(f"  salvo: {r['mapa_mestre']}")

    n = len(todos)
    fig, axs = plt.subplots(n, 2, figsize=(10, 4 * n), squeeze=False)
    vm = max(np.nanpercentile(np.abs(m), 99) for m in mapas)
    for k, (r, m) in enumerate(zip(todos, mapas)):
        x = np.arange(len(r['posicoes']))
        axs[k, 0].bar(x - 0.2, [p['razao_mestre_sem_t'] for p in r['posicoes']], width=0.4, color='#0072B2',
                      edgecolor='white', label='mestre das outras 3 posições')
        axs[k, 0].bar(x + 0.2, [p['razao_proprio_alfa'] for p in r['posicoes']], width=0.4, color='#999999',
                      edgecolor='white', label='próprio α (referência)')
        axs[k, 0].axhline(LIMIAR, color='0.3', ls='--', lw=1)
        axs[k, 0].set_xticks(x); axs[k, 0].set_xlabel('posição (t)')
        axs[k, 0].set_ylabel('amplitude depois / antes'); axs[k, 0].legend(fontsize=7)
        axs[k, 0].set_title(f"{r['arquivo'][8:27]}  →  {r['decisao']}", fontsize=8)
        im = axs[k, 1].imshow(m[::8, ::8] * 1e3, origin='lower', cmap='RdBu_r', vmin=-vm * 1e3, vmax=vm * 1e3)
        axs[k, 1].set_xticks([]); axs[k, 1].set_yticks([])
        axs[k, 1].set_title(f"α mestre {r['arquivo'][8:27]} (preenchido)", fontsize=8)
        fig.colorbar(im, ax=axs[k, 1], shrink=0.8, label='α  (×10⁻³)')
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'cel86_alfa_mestre_por_horario.png', dpi=120)
    plt.close(fig)

    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '86_sst_alfa_mestre_por_horario.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos}, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório: {REP / '86_sst_alfa_mestre_por_horario.json'}\n"
          f"Figura: {FIG / 'cel86_alfa_mestre_por_horario.png'}")


if __name__ == '__main__':
    main()
