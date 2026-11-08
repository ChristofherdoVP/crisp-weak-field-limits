"""
110_sst_limite_campo_fraco_v3.py — o 102 refeito com o motor v3 (Src/physics_v3: núcleo do atlas, INSTRUMENTAL_V3).

Por que: a parte 1 do 102 (validade do filtro de campo fraco e limiares da máscara de confiabilidade) usou o motor v2,
cujo núcleo é largo demais (105/106). Os limiares (I asas 0,91–1,08 e I núcleo ≤ 1,19 × calmo) e os ganhos em plage
e penumbra podem mudar com o núcleo correto.

Mudanças em relação ao 102 (o resto é idêntico, inclusive os critérios e as grades reais):
  - perfis de Src/physics_v3 (get_stokes com V_CORE_KMS_V3 e degradar_stokes v3);
  - instrumento de cada grade = INSTRUMENTAL_V3 ('CRISP/SST' para 2024-10-08: 10,5 pm, espúria 6%, σ = εI;
    'CRISP2/SST' para 2025-08-19: 11,9 pm, sem espúria, σ = ε Ĩ_w (I/Ĩ_w)^0,71), pesos u = t/σ² com esse σ;
  - 1b usa o ganho RELATIVO g/g₀ (g₀ = ganho do perfil de Sol calmo, B = 200 G): com o v3 o estimador tem g₀ ≈ 1,04–1,05
    (108) e os mapas são divididos por g₀; o que a máscara precisa garantir é |g/g₀ − 1| ≤ 0,10. (No 102, g₀ ≈ 1,008 e
    a diferença era irrelevante.) Decidido antes de rodar.
  - máscaras salvas como Outputs/Data/confiavel_v3_<etiqueta>_<…>.npy (as do 102 ficam intactas).
Critérios (os do 102, definidos antes): 1a FAIXA ÚTIL se B_lim ≥ 5 × a₅₀ de uma varredura.
Descritivo novo: limiares v3 × v2 (relatório do 102); se todos mudarem menos de 0,02 em I relativo, as máscaras do 102
continuam válidas e a parte 2 não precisa ser refeita.

Uso:  python Scripts/110_sst_limite_campo_fraco_v3.py                       (só a parte 1)
      python Scripts/110_sst_limite_campo_fraco_v3.py Data/RAW_SST/<arquivo>.fits [mais arquivos]   (parte 2)
Saída: Outputs/Reports/110_sst_limite_campo_fraco_v3.json, Outputs/Figures/cel110_linearidade_modelo.png,
       Outputs/Figures/cel110_confiavel_<…>.png
"""
import sys
import copy
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saidas_sst as SA
import Src.physics_v3 as V3
from Src.physics_v3 import INSTRUMENTAL_V3, LAM, V_CORE_KMS_V3, get_stokes, degradar_stokes, sigma_por_ponto

DAT = ROOT / 'Outputs' / 'Data'
C_ZEEMAN = 4.6686e-13
G_EFF = 1.10
LAMBDA0 = 8542.09
LIMIAR = 0.10
FATOR_UTIL = 5.0
B_GRADE = np.unique(np.concatenate([np.arange(10, 200, 10), np.arange(200, 1000, 25), np.arange(1000, 3001, 50)]))
GRADES_MA = {
    'CRISP_20241008': (np.array([-650, -500, -350, -275, -200, -125, -50, 25, 100, 175, 250, 325, 400, 550, 700.]), 25.0),
    'CRISP2_20250819': (np.array([-656, -506, -356, -281, -206, -131, -56, 19, 94, 169, 244, 319, 394, 544, 694.]), 19.0),
}
A50_UMA = {'CRISP_20241008': [34.0, 48.4], 'CRISP2_20250819': [16.0, 27.0]}   # 101 e 94 (G)


INSTR_GRADE = {'CRISP_20241008': 'CRISP/SST', 'CRISP2_20250819': 'CRISP2/SST'}


def preparar(nome, offs_ma, centro_ma):
    inst = copy.deepcopy(INSTRUMENTAL_V3[INSTR_GRADE[nome]])
    inst['grade_dl_pm'] = (offs_ma - centro_ma) / 10.0
    rng = np.random.default_rng(0)
    deg = lambda B, eta=0.0, kap=1.0: degradar_stokes(
        get_stokes(float(B), filling=1.0, v_km_s=V_CORE_KMS_V3, preench_nucleo=eta, contraste_cont=kap),
        LAM, inst, rng_local=rng, fator_ruido=0.0)
    d0 = deg(0.0)
    I = d0['I'].astype(float)
    lam = d0['lam_obs'] * 10.0 if np.median(d0['lam_obs']) < 2000 else d0['lam_obs']
    tk = -C_ZEEMAN * G_EFF * LAMBDA0 ** 2 * np.gradient(I, lam)
    uk = tk / sigma_por_ponto(inst, I) ** 2
    D = float(np.sum(uk * tk))
    knuc = int(np.argmin(I))
    return deg, uk, D, I, knuc


def faixa_contigua(x, ok, x0):
    """Maior intervalo contíguo de x com ok, contendo o ponto mais próximo de x0."""
    i0 = int(np.argmin(np.abs(x - x0)))
    if not ok[i0]:
        return None
    a = i0
    while a > 0 and ok[a - 1]:
        a -= 1
    b = i0
    while b < len(x) - 1 and ok[b + 1]:
        b += 1
    return a, b


def linearidade(nome, offs_ma, centro_ma):
    deg, uk, D, I, knuc = preparar(nome, offs_ma, centro_ma)
    g = np.array([float(np.sum(uk * deg(B)['V']) / D / B) for B in B_GRADE])
    g_rel = g / g[0]
    fora = np.where(np.abs(g_rel - 1) > LIMIAR)[0]
    B_lim = float(B_GRADE[fora[0]]) if len(fora) else float('inf')
    util = bool(B_lim >= FATOR_UTIL * max(A50_UMA[nome]))
    # 1b: template de Sol calmo em perfis diferentes (B = 200 G)
    obs = lambda d: (float((d['I'][0] + d['I'][14]) / (I[0] + I[14])), float(d['I'][knuc] / I[knuc]))
    etas = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    g_eta, nuc_eta = [], []
    for e in etas:
        d = deg(200.0, eta=e)
        g_eta.append(float(np.sum(uk * d['V']) / D / 200.0))
        nuc_eta.append(obs(d)[1])
    kaps = np.round(np.arange(0.3, 1.3001, 0.01), 2)
    g_kap, asa_kap = [], []
    for k in kaps:
        d = deg(200.0, kap=k)
        g_kap.append(float(np.sum(uk * d['V']) / D / 200.0))
        asa_kap.append(obs(d)[0])
    g_eta, nuc_eta, g_kap, asa_kap = map(np.array, (g_eta, nuc_eta, g_kap, asa_kap))
    g0 = float(np.sum(uk * deg(200.0)['V']) / D / 200.0)          # Sol calmo, B = 200 G
    g_eta_abs, g_kap_abs = g_eta.copy(), g_kap.copy()
    g_eta, g_kap = g_eta / g0, g_kap / g0                          # ganho relativo (mapas divididos por g₀)
    fe = faixa_contigua(etas, np.abs(g_eta - 1) <= LIMIAR, 0.0)
    fk = faixa_contigua(kaps, np.abs(g_kap - 1) <= LIMIAR, 1.0)
    nuc_max = float(nuc_eta[fe[1]])
    asa_min, asa_max = float(asa_kap[fk[0]]), float(asa_kap[fk[1]])
    print(f"  {nome}: ganho em 10 G = {g[0]:.3f};  g/g(10 G) em 500/1000/2000/3000 G = "
          + ' / '.join(f"{g_rel[np.argmin(np.abs(B_GRADE - b))]:.3f}" for b in (500, 1000, 2000, 3000))
          + f";  B_lim = {'> 3000' if not np.isfinite(B_lim) else f'{B_lim:.0f}'} G  →  "
            f"{'FAIXA ÚTIL' if util else 'FAIXA INSUFICIENTE'} (5 × a₅₀ máx. = {FATOR_UTIL * max(A50_UMA[nome]):.0f} G)")
    print(f"     g₀ (Sol calmo, 200 G) = {g0:.3f};  template de Sol calmo: |g/g₀ − 1| ≤ {LIMIAR} para η ≤ {etas[fe[1]]:.2f} (I núcleo ≤ {nuc_max:.2f} × calmo) "
          f"e κ em {kaps[fk[0]]:.2f}–{kaps[fk[1]]:.2f} (I asas {asa_min:.2f}–{asa_max:.2f} × calmo);  "
          f"g/g₀ em η = 0,25/0,5: {g_eta[25]:.2f}/{g_eta[50]:.2f};  g/g₀ em κ = 0,7/0,3: "
          f"{g_kap[np.argmin(np.abs(kaps - 0.7))]:.2f}/{g_kap[0]:.2f}")
    return {'grade_mA': offs_ma.tolist(), 'centro_mA': centro_ma, 'ganho_10G': float(g[0]), 'g0_200G': g0,
            'instrumento': INSTR_GRADE[nome], 'g_eta_abs': g_eta_abs.tolist(), 'g_kappa_abs': g_kap_abs.tolist(),
            'B_G': B_GRADE.tolist(), 'g_rel': g_rel.tolist(), 'B_lim_G': B_lim, 'a50_uma_varredura_G': A50_UMA[nome],
            'decisao': 'FAIXA ÚTIL' if util else 'FAIXA INSUFICIENTE',
            'eta': etas.tolist(), 'g_eta': g_eta.tolist(), 'nuc_rel_eta': nuc_eta.tolist(),
            'kappa': kaps.tolist(), 'g_kappa': g_kap.tolist(), 'asa_rel_kappa': asa_kap.tolist(),
            'eta_max': float(etas[fe[1]]), 'kappa_faixa': [float(kaps[fk[0]]), float(kaps[fk[1]])],
            'nuc_max': nuc_max, 'asa_min': asa_min, 'asa_max': asa_max}


def confiavel(caminho, L):
    nome = Path(caminho).name
    tag = SA.etiqueta(nome)
    serie = (DAT / f'blos_f_crisp_{tag}_media.npy').exists()
    g0 = L['g0_200G']
    B_lim = L['B_lim_G']
    res = {'arquivo': nome, 'etiqueta': tag, 'B_lim_G': B_lim, 'asa_faixa': [L['asa_min'], L['asa_max']],
           'nuc_max': L['nuc_max'], 'mapas': []}
    figs = []
    with fits.open(caminho, memmap=True) as h:
        d = h[0].data
        nt, ns, nl = d.shape[:3]
        idx = [0] if serie else list(range(nt))
        for t in idx:
            arq = DAT / (f'blos_f_crisp_{tag}_media.npy' if serie else f'blos_f_crisp2_{tag}_t{t}.npy')
            if not arq.exists():
                print(f"  {nome}: mapa {arq.name} não encontrado (rode o 101 ou o 89); pulando")
                continue
            a = np.load(arq).astype(np.float64) / g0                 # B∥·f corrigido pelo ganho do estimador v3
            amostra = np.array([np.asarray(d[t, 0, l, ::8, ::8], dtype=np.float32) for l in range(nl)])
            okA = np.all(np.isfinite(amostra), axis=0) & (amostra[0] != 0)
            lnuc = int(np.argmin([np.median(amostra[l][okA]) for l in range(nl)]))
            I0, I14, Ic = (np.asarray(d[t, 0, l], dtype=np.float64) for l in (0, 14, lnuc))
            ny, nx = min(a.shape[0], I0.shape[0]), min(a.shape[1], I0.shape[1])
            a, I0, I14, Ic = a[:ny, :nx], I0[:ny, :nx], I14[:ny, :nx], Ic[:ny, :nx]
            Iw = (I0 + I14) / 2
            val = np.isfinite(a) & np.isfinite(Iw) & (Iw != 0) & np.isfinite(Ic)
            asa = Iw / np.median(Iw[val])
            nuc = Ic / np.median(Ic[val])
            escuro = asa < L['asa_min']
            claro_asa = asa > L['asa_max']
            plage = nuc > L['nuc_max']
            forte = np.abs(a) > B_lim
            conf = val & ~escuro & ~claro_asa & ~plage & ~forte
            vs = val.sum()
            fr = {'t': 'media' if serie else t, 'mapa': str(arq),
                  'frac_escuro': float((val & escuro).sum() / vs),
                  'frac_asa_clara': float((val & claro_asa & ~escuro).sum() / vs),
                  'frac_nucleo_brilhante': float((val & plage & ~escuro & ~claro_asa).sum() / vs),
                  'frac_acima_Blim': float((val & forte & ~escuro & ~claro_asa & ~plage).sum() / vs),
                  'frac_confiavel': float(conf.sum() / vs),
                  'P01_P99_confiavel_G': [float(np.percentile(a[conf], 1)), float(np.percentile(a[conf], 99))]
                  if conf.any() else None}
            res['mapas'].append(fr)
            sufixo = 'media' if serie else f't{t}'
            np.save(DAT / f'confiavel_v3_{tag}_{sufixo}.npy', conf)
            print(f"  {nome[8:27]} {sufixo}: confiável {fr['frac_confiavel']:.0%} da área válida  (escuro "
                  f"{fr['frac_escuro']:.0%}, asa clara {fr['frac_asa_clara']:.0%}, núcleo brilhante "
                  f"{fr['frac_nucleo_brilhante']:.0%}, |â| > B_lim {fr['frac_acima_Blim']:.1%});  â P1–P99 no "
                  f"confiável: {fr['P01_P99_confiavel_G'][0]:+.0f} a {fr['P01_P99_confiavel_G'][1]:+.0f} G")
            figs.append((sufixo, asa[::4, ::4], a[::4, ::4], (val & ~conf)[::4, ::4], val[::4, ::4]))
    return res, figs


def fig_linearidade(lin, caminho):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    cores = {'CRISP_20241008': '#0072B2', 'CRISP2_20250819': '#D55E00'}
    for nome, r in lin.items():
        c = cores[nome]
        axs[0].plot(r['B_G'], r['g_rel'], color=c, lw=1.5, label=nome)
        axs[1].plot(r['nuc_rel_eta'], r['g_eta'], color=c, lw=1.5, label=f"{nome}: núcleo ≤ {r['nuc_max']:.2f}")
        axs[1].axvline(r['nuc_max'], color=c, lw=1, ls=':')
        axs[2].plot(r['asa_rel_kappa'], r['g_kappa'], color=c, lw=1.5,
                    label=f"{nome}: asas {r['asa_min']:.2f}–{r['asa_max']:.2f}")
        for v in (r['asa_min'], r['asa_max']):
            axs[2].axvline(v, color=c, lw=1, ls=':')
    for ax in axs:
        ax.axhspan(1 - LIMIAR, 1 + LIMIAR, color='0.9', zorder=0)
        ax.legend(fontsize=7)
    axs[0].set_xlabel('B (G), f = 1, θ = 0, perfil de Sol calmo'); axs[0].set_ylabel('â/B normalizado em 10 G')
    axs[0].set_title('1a: saturação Zeeman', fontsize=9)
    axs[1].set_xlabel('I núcleo / I núcleo Sol calmo (η de 0 a 1)'); axs[1].set_ylabel('g/g₀, B = 200 G')
    axs[1].set_title('1b: núcleo preenchido (plage), template de Sol calmo', fontsize=9)
    axs[2].set_xlabel('I asas / I asas Sol calmo (κ de 0,3 a 1,3)'); axs[2].set_ylabel('g/g₀, B = 200 G')
    axs[2].set_title('1b: contraste do contínuo (penumbra/umbra/fácula)', fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def fig_confiavel(res, figs, caminho):
    n = len(figs)
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 6.8), squeeze=False)
    vm = 200.0
    for j, (suf, asa, a, nconf, val) in enumerate(figs):
        axs[0, j].imshow(np.where(val, asa, np.nan), origin='lower', cmap='gray', vmin=0.6, vmax=1.3)
        axs[0, j].set_title(f"{res['arquivo'][8:27]} {suf}: I asas", fontsize=7)
        im = axs[1, j].imshow(a, origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm)
        axs[1, j].imshow(np.where(nconf, 1.0, np.nan), origin='lower', cmap='Greys', vmin=0, vmax=1.6, alpha=0.75)
        fr = res['mapas'][j]['frac_confiavel']
        axs[1, j].set_title(f'B∥·f; cinza = não confiável ({1 - fr:.0%})', fontsize=7)
        for ax in axs[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs[1, :], shrink=0.8, extend='both', label=f'B∥·f (G), ±{vm:.0f} G')
    fig.savefig(caminho, dpi=100)
    plt.close(fig)


def comparar_v2(lin, out):
    """Limiares v3 × v2 (relatório do 102). Mudança < 0,02 em todos → máscaras do 102 continuam válidas."""
    p = SA.REP / '102_sst_limite_campo_fraco.json'
    if not p.exists():
        print('  (relatório do 102 não encontrado: comparação com o v2 não feita)')
        return
    v2 = json.loads(p.read_text(encoding='utf-8'))['linearidade']
    comp, max_d = {}, 0.0
    print('\n  Limiares da máscara, v2 (102) → v3:')
    for nome, r in lin.items():
        c = {}
        for k in ('asa_min', 'asa_max', 'nuc_max'):
            c[k] = [v2[nome][k], r[k]]
            max_d = max(max_d, abs(r[k] - v2[nome][k]))
        comp[nome] = c
        print(f"    {nome}: asas {v2[nome]['asa_min']:.3f}–{v2[nome]['asa_max']:.3f} → {r['asa_min']:.3f}–{r['asa_max']:.3f};  "
              f"núcleo ≤ {v2[nome]['nuc_max']:.3f} → {r['nuc_max']:.3f};  g(η = 0,25/0,5): "
              f"{v2[nome]['g_eta'][25]:.2f}/{v2[nome]['g_eta'][50]:.2f} → {r['g_eta'][25]:.2f}/{r['g_eta'][50]:.2f};  "
              f"g(κ = 0,7): {v2[nome]['g_kappa'][40]:.2f} → {r['g_kappa'][40]:.2f}")
    dec = ('MÁSCARAS DO 102 CONTINUAM VÁLIDAS' if max_d < 0.02
           else 'LIMIARES MUDARAM: refazer a parte 2 (máscaras v3)')
    print(f'  Maior mudança de limiar: {max_d:.3f}  →  {dec}')
    out['comparacao_v2'] = {'limiares': comp, 'maior_mudanca': max_d, 'decisao': dec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='*')
    a = ap.parse_args()
    print('PARTE 1 — validade do filtro no modelo')
    lin = {nome: linearidade(nome, *g) for nome, g in GRADES_MA.items()}
    out = {'linearidade': lin, 'confiabilidade': []}
    SA.REP.mkdir(parents=True, exist_ok=True)
    comparar_v2(lin, out)
    rel = SA.REP / '110_sst_limite_campo_fraco_v3.json'
    rel.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    cf = SA.figura('cel110_linearidade', 'modelo')
    fig_linearidade(lin, cf)
    if a.arquivos:
        print('PARTE 2 — máscara de confiabilidade nos dados')
    for c in a.arquivos:
        nome = Path(c).name
        chave = 'CRISP2_20250819' if '2025-08-19' in nome else 'CRISP_20241008'
        res, figs = confiavel(c, lin[chave])
        out['confiabilidade'].append(res)
        rel.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        if figs:
            fig_confiavel(res, figs, SA.figura('cel110_confiavel', res['etiqueta']))
    print(f"\nRelatório: {rel}\nFiguras: {cf} e cel110_confiavel_<etiqueta>.png")


if __name__ == '__main__':
    main()
