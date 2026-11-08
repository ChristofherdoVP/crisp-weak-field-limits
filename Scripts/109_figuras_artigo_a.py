"""
109_figuras_artigo_a.py — figuras e tabela do artigo A (Solar Physics), em inglês.

Só lê relatórios (Outputs/Reports) e mapas (Outputs/Data); não relê os cubos. Recalcula apenas os perfis do motor
(v2 e v3, rápidos) e o atlas (Data/Reference/atlas_fts_neckel_hs5000_8542.txt) para a Fig. 5.

Formato Springer/Solar Physics: largura de 84 mm (uma coluna) ou 174 mm (página), fonte 8 pt, PDF vetorial +
PNG 300 dpi. Cores Okabe–Ito validadas para daltonismo, sempre com marcador diferente por conjunto (codificação
secundária; o par CRISP2 08:01/10:52 fica na faixa de 6–8 ΔE para deuteranopia).

Figuras (Outputs/Figures/ArtigoA/):
  figA1_noise        (a) σ_k/I_w × I_k/I_w e as leis σ ∝ I^p (CRISP: 96; CRISP2: 93); (b) ε_V por varredura/posição (83)
  figA2_crosstalk    (a) α(x, y) por varredura/posição: mediana e P5–P95 (84); (b) P95|Δâ|/s (CRISP: 99 M1; CRISP2: 88)
  figA3_map_noise    (a) R entre varreduras consecutivas (100); (b) CRISP2: σ observado × previsto nas sobreposições (93)
  figA4_averaging    σ do mapa médio × N, com s/√N (101); binning 2×2 anotado
  figA5_profiles     (a) atlas × motor v2 × v3; (b) perfis médios observados (105) × v3 degradado (INSTRUMENTAL_V3)
  figA6_maps         mapas médios de B_los·f das séries CRISP divididos pelo ganho g (108), contorno da máscara de
                     confiabilidade (102) suavizado; frações das máscaras brutas no título
  figA7_detection    (a) curvas de detecção v3 (107), CRISP e CRISP2, 1 varredura e média de 11;
                     (b) a₅₀ medido × previsto (v2: 104; v3: 107), em B_los·f verdadeiro (medidos ÷ g)
Tabela: Outputs/Reports/109_tabela_a50.csv e .tex (a₅₀ medidos e previstos).

Uso:
    python Scripts/109_figuras_artigo_a.py                  (todas)
    python Scripts/109_figuras_artigo_a.py --fig 1 5        (só algumas)
    python Scripts/109_figuras_artigo_a.py --escala-crisp 0.044   (eixos dos mapas em arcsec; padrão: pixels)
"""
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REP = ROOT / 'Outputs' / 'Reports'
DATA = ROOT / 'Outputs' / 'Data'
OUT = ROOT / 'Outputs' / 'Figures' / 'ArtigoA'
ATLAS = ROOT / 'Data' / 'Reference' / 'atlas_fts_neckel_hs5000_8542.txt'
MM = 1 / 25.4
W1, W2 = 84 * MM, 174 * MM

CRISP_TAGS = ['20241008T094951', '20241008T101050']
CRISP2_TAGS = ['20250819T080131', '20250819T104836']
DS = {'20241008T094951': ('CRISP 09:49', '#0072B2', 'o'),
      '20241008T101050': ('CRISP 10:10', '#E69F00', 's'),
      '20250819T080131': ('CRISP2 08:01', '#009E73', '^'),
      '20250819T104836': ('CRISP2 10:52', '#CC79A7', 'v')}
INK, MUTED = '#222222', '#777777'
OFF_CRISP = np.array([-650, -500, -350, -275, -200, -125, -50, 25, 100, 175, 250, 325, 400, 550, 700.])
OFF_CRISP2 = np.array([-656, -506, -356, -281, -206, -131, -56, 19, 94, 169, 244, 319, 394, 544, 694.])

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 4,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False, 'savefig.dpi': 300,
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'axes.titleweight': 'normal',
    'pdf.fonttype': 42,
})


# ----------------------------------------------------------------------------------------------------------------
def ler(nome):
    p = REP / nome
    d = json.loads(p.read_text(encoding='utf-8'))
    return d


def dados(nome):
    """Relatório por arquivo (_saidas_sst): devolve o primeiro registro de 'data'."""
    return ler(nome)['data'][0]


def por_arquivo(nome, tag_de):
    """Relatório combinado (lista 'data' com 'arquivo'): devolve o registro cujo arquivo contém a data/hora."""
    for r in ler(nome)['data']:
        if tag_de(r['arquivo']):
            return r
    raise KeyError(f'{nome}: arquivo não encontrado')


def tag_de_arquivo(arq):
    import re
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})T(\d{2})[_:](\d{2})[_:](\d{2})', arq)
    return f'{m[1]}{m[2]}{m[3]}T{m[4]}{m[5]}{m[6]}'


def ganhos_108():
    r = ler('108_regressao_bf_v3.json')['data']['resultados']
    return {'CRISP': r['CRISP/SST']['WF-mapa']['1.000']['Quieta+Rede']['ganho'],
            'CRISP2': r['CRISP2/SST']['WF-mapa']['1.000']['Quieta+Rede']['ganho']}


def letra(ax, s, x=0.02, y=0.96):
    ax.text(x, y, f'({s})', transform=ax.transAxes, ha='left', va='top', fontsize=8, color=INK)


# numeração final do artigo (ordem da primeira citação no texto)
NUM_ARTIGO = {'figA5_profiles': 1, 'figA9_synthetic': 2, 'figA1_noise': 3, 'figA3_map_noise': 4,
              'figA4_averaging': 5, 'figA2_crosstalk': 6, 'figA6_maps': 7, 'figA8_maps_crisp2': 8,
              'figA7_detection': 9, 'figA10_calibration': 10}
MAPAS = {'figA6_maps', 'figA8_maps_crisp2'}      # imagens com transparência: TIFF em vez de EPS


def salvar(fig, nome):
    OUT.mkdir(parents=True, exist_ok=True)
    n = NUM_ARTIGO[nome]
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'Fig{n}.{ext}', bbox_inches='tight', pad_inches=0.02, dpi=600 if ext == 'png' else None)
    if nome in MAPAS:
        # EPS não guarda transparência (véu e lacunas NaN): TIFF 600 dpi, aceito pela Solar Physics
        fig.savefig(OUT / f'Fig{n}.tiff', bbox_inches='tight', pad_inches=0.02, dpi=600,
                    pil_kwargs={'compression': 'tiff_lzw'})
        ext3 = 'tiff'
    else:
        fig.savefig(OUT / f'Fig{n}.eps', bbox_inches='tight', pad_inches=0.02)
        ext3 = 'eps'
    plt.close(fig)
    print(f'  Fig{n}.pdf/.png/.{ext3}   ({nome})')


# ----------------------------------------------------------------------------------------------------------------
def fig1():
    r96 = ler('96_sst_ruido_series_temporais.json')['data']
    r93 = ler('93_sst_ruido_por_lambda.json')['data']
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 62 * MM), constrained_layout=True)
    x = np.linspace(0.18, 1.05, 50)
    for r in r96:
        tag = tag_de_arquivo(r['arquivo'])
        lab, cor, mk = DS[tag]
        I = np.asarray(r['Imed']); s = np.asarray(r['sigma_k'])
        Iw = 0.5 * (I[0] + I[-1])
        a.plot(I / Iw, s / Iw * 1e3, mk, color=cor, mfc='none', mew=0.8, label=lab)
        p, e0 = r['ajuste_sem_N']['p'], r['ajuste_sem_N']['eps0']
        a.plot(x, e0 * x ** p * 1e3, '-', color=cor, lw=0.7)
        a.text(x[0] * 0.97, e0 * x[0] ** p * 1e3, f'$p={p:.2f}$', color=cor, fontsize=8, va='center', ha='right')
    for r in r93:
        tag = tag_de_arquivo(r['arquivo'])
        lab, cor, mk = DS[tag]
        acc = {}
        for par in r['pares']:
            for pl in par['por_lambda']:
                acc.setdefault(tuple(pl['par_lambda']), []).append((pl['I_rel'], pl['sigma_mais']))
        Ir = np.array([np.mean([v[0] for v in vs]) for vs in acc.values()])
        sr = np.array([np.mean([v[1] for v in vs]) for vs in acc.values()])
        a.plot(Ir, sr * 1e3, mk, color=cor, label=lab)
        p = float(np.median([par['p'] for par in r['pares']]))
        e = float(np.exp(np.mean(np.log(sr) - p * np.log(Ir))))
        a.plot(x, e * x ** p * 1e3, '--', color=cor, lw=0.7)
        a.text(x[0] * 0.97, e * x[0] ** p * 1e3, f'$p={p:.2f}$', color=cor, fontsize=8, va='center', ha='right')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlim(0.105, 1.25)
    a.set_xticks([0.2, 0.3, 0.5, 0.7, 1.0]); a.set_xticklabels(['0.2', '0.3', '0.5', '0.7', '1'])
    a.set_yticks([0.5, 1, 2, 3, 4]); a.set_yticklabels(['0.5', '1', '2', '3', '4'])
    a.minorticks_off()
    a.set_xlabel(r'$I_k/I_\mathrm{w}$')
    a.set_ylabel(r'$\sigma_{V,k}/I_\mathrm{w}$  [$10^{-3}$]')
    a.legend(loc='lower right', handletextpad=0.3, borderaxespad=0.2)
    letra(a, 'a')

    xs = 0
    ticks, rot = [], []
    for tag in CRISP_TAGS + CRISP2_TAGS:
        lab, cor, mk = DS[tag]
        r = dados(f'83_sst_ruido_v_sem_borda_{tag}.json')
        eps = np.array([0.5 * (p['w16']['eps_mais'] + p['w16']['eps_menos']) for p in r['posicoes']])
        xx = xs + np.arange(len(eps))
        b.plot(xx, eps * 1e3, mk, color=cor, ms=3.5, mfc='none' if tag in CRISP_TAGS else cor, mew=0.8)
        b.hlines(r['eps_V']['mediano'] * 1e3, xx[0] - 0.4, xx[-1] + 0.4, color=cor, lw=0.9)
        ticks.append(xx.mean()); rot.append(lab.replace(' ', '\n'))
        xs = xx[-1] + 6
    b.set_xticks(ticks); b.set_xticklabels(rot, fontsize=8)
    b.tick_params(axis='x', length=0, top=False)
    b.set_ylabel(r'$\varepsilon_V$  [$10^{-3}$]')
    b.set_ylim(1.8, 4.0)
    letra(b, 'b')
    salvar(fig, 'figA1_noise')


def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 58 * MM), constrained_layout=True)
    xs, ticks, rot = 0, [], []
    for tag in CRISP_TAGS + CRISP2_TAGS:
        lab, cor, mk = DS[tag]
        r = dados(f'84_sst_crosstalk_alfa_xy_{tag}.json')
        med = np.array([p['alfa_mediana'] for p in r['posicoes']])
        lo = np.array([p['alfa_p05_p95'][0] for p in r['posicoes']])
        hi = np.array([p['alfa_p05_p95'][1] for p in r['posicoes']])
        xx = xs + np.arange(len(med))
        a.vlines(xx, lo * 1e3, hi * 1e3, color=cor, lw=1.2)
        a.plot(xx, med * 1e3, mk, color=cor, ms=3.5, mfc='white', mew=0.8)
        ticks.append(xx.mean()); rot.append(lab.replace(' ', '\n'))
        xs = xx[-1] + 6
    a.axhline(0, color=MUTED, lw=0.5)
    a.set_xticks(ticks); a.set_xticklabels(rot, fontsize=8); a.tick_params(axis='x', length=0, top=False)
    a.set_ylabel(r'$\alpha_{I\to V}$  [$10^{-3}$]  (median, P5–P95)')
    letra(a, 'a')

    xs, ticks, rot = 0, [], []
    for tag in CRISP_TAGS:
        lab, cor, mk = DS[tag]
        r = dados(f'99_sst_vies_blos_series_{tag}.json')
        v = np.array([q['M1']['razao_P95_s'] for q in r['varreduras']])
        xx = xs + np.arange(len(v))
        b.plot(xx, v, mk, color=cor, ms=3.5, mfc='none', mew=0.8)
        ticks.append(xx.mean()); rot.append(lab.replace(' ', '\n'))
        xs = xx[-1] + 6
    for tag in CRISP2_TAGS:
        lab, cor, mk = DS[tag]
        r = por_arquivo('88_sst_vies_blos_simetrico.json', lambda arq, t=tag: tag_de_arquivo(arq) == t)
        v = np.array([q['razao_P95_s'] for q in r['posicoes']])
        xx = xs + np.arange(len(v))
        b.plot(xx, v, mk, color=cor, ms=3.5)
        ticks.append(xx.mean()); rot.append(lab.replace(' ', '\n'))
        xs = xx[-1] + 6
    b.axhline(0.2, color=INK, lw=0.6, ls=':')
    b.text(0.98, 0.2, 'correction threshold', transform=b.get_yaxis_transform(), ha='right', va='bottom', fontsize=8,
           color=INK)
    b.set_xticks(ticks); b.set_xticklabels(rot, fontsize=8); b.tick_params(axis='x', length=0, top=False)
    b.set_ylabel(r'P95$|\Delta \hat a|$ / $s$')
    b.set_ylim(0, None)
    letra(b, 'b')
    salvar(fig, 'figA2_crosstalk')


def fig3():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 58 * MM), constrained_layout=True)
    for tag in CRISP_TAGS:
        lab, cor, mk = DS[tag]
        r = dados(f'100_sst_mapas_blos_series_{tag}.json')
        R = np.array([p['R'] for p in r['pares']])
        a.plot(np.arange(1, len(R) + 1), R, mk + '-', color=cor, lw=0.6, mfc='none', mew=0.8,
               label=f'{lab}  (median {np.median(R):.2f})')
    a.axhline(1, color=MUTED, lw=0.5)
    a.axhspan(0.9, 1.1, color='0.92', lw=0, zorder=0)
    a.set_ylim(0.8, 1.2)
    a.set_xlabel('pair of consecutive scans')
    a.set_ylabel(r'$R=\sigma(\Delta\hat a)/\sqrt{s_t^2+s_{t+1}^2}$')
    a.legend(loc='lower left')
    letra(a, 'a')

    lim = [10, 45]
    b.plot(lim, lim, color=MUTED, lw=0.5)
    for r in ler('93_sst_ruido_por_lambda.json')['data']:
        tag = tag_de_arquivo(r['arquivo'])
        lab, cor, mk = DS[tag]
        o = np.array([p['sigma_obs_dif_a_G'] for p in r['pares']])
        pt = np.array([p['sigma_pred_teorico_G'] for p in r['pares']])
        pr = np.array([p['sigma_pred_ruido_G'] for p in r['pares']])
        b.plot(pt, o, mk, color=cor, mfc='none', mew=0.8)
        b.plot(pr, o, mk, color=cor, label=lab)
        for x0, x1, y in zip(pt, pr, o):
            b.annotate('', (x1, y), (x0, y), arrowprops=dict(arrowstyle='->', lw=0.5, color=cor, shrinkA=2, shrinkB=2))
    b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel(r'predicted $\sigma(\Delta\hat a)$ [G]')
    b.set_ylabel(r'observed $\sigma(\Delta\hat a)$ in tile overlaps [G]')
    b.text(0.97, 0.05, 'open: noise $\\propto I$\nfilled: measured per-$\\lambda$ noise',
           transform=b.transAxes, ha='right', va='bottom', fontsize=8)
    b.legend(loc='upper left', bbox_to_anchor=(0.06, 1.0))
    letra(b, 'b')
    salvar(fig, 'figA3_map_noise')


def fig4():
    fig, a = plt.subplots(figsize=(W1, 84 * MM), constrained_layout=True)
    NN = np.linspace(1, 15, 50)
    for tag in CRISP_TAGS:
        lab, cor, mk = DS[tag]
        r = dados(f'101_sst_media_temporal_{tag}.json')
        N = np.array([p['N'] for p in r['por_N']])
        s = np.array([p['sigma_media_N_G'] for p in r['por_N']])
        a.plot(N, s, mk, color=cor, mfc='none', mew=0.8, label=lab)
        a.plot(NN, s[0] / np.sqrt(NN), '-', color=cor, lw=0.7)
        bb = r['binning_2x2']
        a.plot([1], [bb['sigma_G']], mk, color=cor, ms=4.5, label=f'{lab}, 2×2 binning (gain {bb["ganho"]:.2f})')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xticks([1, 2, 3, 5, 7, 10]); a.set_xticklabels(['1', '2', '3', '5', '7', '10'])
    a.set_yticks([5, 10, 20]); a.set_yticklabels(['5', '10', '20'])
    a.minorticks_off()
    a.set_xlabel('number of averaged scans $N$')
    a.set_ylabel(r'$\sigma$ of the averaged map [G]')
    a.set_ylim(4, 30)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), title=r'lines: $\sigma_1/\sqrt{N}$', title_fontsize=8,
             handletextpad=0.3)
    salvar(fig, 'figA4_averaging')


def fig5():
    import Src.physics as P
    import Src.physics_v3 as V3
    at = np.loadtxt(ATLAS)
    lamA_at = at[:, 0] * 10 if np.median(at[:, 0]) < 2000 else at[:, 0]
    I2 = P.get_stokes(0.0, v_km_s=P.V_CORE_KMS)['I']
    I3 = V3.get_stokes(0.0)['I']
    lamA = P.LAM * 10
    c0 = lamA_at[np.argmin(at[:, 1])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 64 * MM), constrained_layout=True)
    m = np.abs(lamA_at - c0) <= 1.5
    a.plot(lamA_at[m] - c0, at[m, 1], color=INK, lw=1.4, label='FTS atlas (Neckel)')
    mm_ = np.abs(lamA - c0) <= 1.5
    a.plot(lamA[mm_] - c0, I2[mm_], color='#D55E00', lw=0.9, ls='--', label='core of Paper B')
    a.plot(lamA[mm_] - c0, I3[mm_], color='#0072B2', lw=0.9, label='atlas-fitted core')
    a.set_xlim(-1.5, 1.5)
    a.set_xlabel(r'$\Delta\lambda$ [Å]'); a.set_ylabel(r'$I/I_\mathrm{c}$')
    a.legend(loc='lower left')
    letra(a, 'a', x=0.03, y=0.96)
    r105 = ler('105_sst_perfil_observado_modelo.json')
    for tag, d in r105['conjuntos'].items():
        lab, cor, mk = DS[tag]
        inst = 'CRISP/SST' if d['instrumento'] == 'CRISP' else 'CRISP2/SST'
        off = OFF_CRISP if d['instrumento'] == 'CRISP' else OFF_CRISP2
        cen = 25.0 if d['instrumento'] == 'CRISP' else 19.0
        b.plot((off - cen) / 1000, d['perfil_obs'], mk, color=cor, mfc=cor, ms=3.5,
               mew=0.8, label=lab)   # cheios, como na Fig. 9b
    for inst, ls, cen in (('CRISP/SST', '-', 25.0), ('CRISP2/SST', '--', 19.0)):
        ins = V3.INSTRUMENTAL_V3[inst]
        dd = V3.degradar_stokes(V3.get_stokes(0.0), P.LAM, ins, rng_local=np.random.default_rng(0), fator_ruido=0.0)
        lo = np.asarray(dd['lam_obs']) * 10
        Iw = 0.5 * (dd['I'][0] + dd['I'][-1])
        b.plot(lo - lo[7], dd['I'] / Iw, ls, color=MUTED, lw=0.8, label=f'{inst.split("/")[0]} instrument model')
    b.set_xlabel(r'$\Delta\lambda$ [Å]'); b.set_ylabel(r'$I/I_\mathrm{w}$')
    b.legend(loc='upper right', ncol=2, columnspacing=0.6, handletextpad=0.2, handlelength=1.5, borderaxespad=0.3)
    b.set_ylim(0.3, 1.42)
    letra(b, 'b', x=0.03, y=0.08)
    salvar(fig, 'figA5_profiles')


def fig6(escala):
    g = ganhos_108()['CRISP']
    # máscaras v3 (script 110) quando existirem; senão as do 102
    conf, pref = {}, {}
    for nome_rel, prefixo in (('102_sst_limite_campo_fraco.json', 'confiavel_'),
                              ('110_sst_limite_campo_fraco_v3.json', 'confiavel_v3_')):
        if (REP / nome_rel).exists():
            for c in ler(nome_rel)['confiabilidade']:
                if (DATA / f'{prefixo}{c["etiqueta"]}_media.npy').exists():
                    conf[c['etiqueta']], pref[c['etiqueta']] = c, prefixo
    geo = {}                                                       # escala e apontamento do 111, se existir
    if (REP / '111_sst_escala_crisp.json').exists():
        for r in ler('111_sst_escala_crisp.json')['arquivos']:
            geo[r['etiqueta']] = (r['escala_mediana_arcsec_px'], r['geometria'][0]['centro'])
    fig, axs = plt.subplots(1, 2, figsize=(W2, 84 * MM), constrained_layout=True)
    vmax = 200
    for ax, tag in zip(axs, CRISP_TAGS):
        lab = DS[tag][0]
        B = np.load(DATA / f'blos_f_crisp_{tag}_media.npy').astype(float) / g
        NY, NX = B.shape
        M = np.load(DATA / f'{pref[tag]}{tag}_media.npy').astype(bool)
        print(f'    {tag}: máscara {pref[tag]}{tag}_media.npy')
        val = np.isfinite(B)
        ys, xs_ = np.where(val)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs_.min(), xs_.max() + 1      # recorte na área válida
        B, M, val = B[y0:y1, x0:x1], M[y0:y1, x0:x1], val[y0:y1, x0:x1]
        ny, nx = B.shape
        if escala is None and tag in geo:                          # coordenadas solares (111)
            e, (cx, cy) = geo[tag]
            X0, Y0 = cx - (NX - 1) / 2 * e, cy - (NY - 1) / 2 * e
            ext = [X0 + (x0 - 0.5) * e, X0 + (x1 - 0.5) * e, Y0 + (y0 - 0.5) * e, Y0 + (y1 - 0.5) * e]
            u = 'solar'
        else:
            e = 1.0 if escala is None else escala
            ext = [0, nx * e, 0, ny * e]
            u = 'pixel' if escala is None else 'arcsec'
        cm = plt.get_cmap('RdBu_r').copy(); cm.set_bad('0.85')
        im = ax.imshow(np.clip(B, -vmax, vmax), origin='lower', cmap=cm, norm=TwoSlopeNorm(0, -vmax, vmax),
                       extent=ext, interpolation='nearest', rasterized=True)
        Ms = gaussian_filter(M.astype(float), 8) > 0.5                       # máscara suavizada (só para mostrar)
        veu = np.zeros((ny, nx, 4)); veu[..., :3] = 1.0
        veu[..., 3] = np.where(val & ~Ms, 0.55, 0.0)                          # não confiável: véu branco
        ax.imshow(veu, origin='lower', extent=ext, interpolation='nearest', rasterized=True)
        yy = np.linspace(ext[2], ext[3], ny); xx = np.linspace(ext[0], ext[1], nx)
        Mc = np.where(val, Ms.astype(float), np.nan)
        ax.contour(xx, yy, Mc, levels=[0.5], colors='k', linewidths=0.35)
        fr = [m for m in conf[tag]['mapas'] if m['t'] == 'media']
        frac = fr[0]['frac_confiavel'] if fr else M.mean() / max(np.isfinite(B).mean(), 1e-9)
        ax.set_title(f'{lab}, mean of {dados(f"101_sst_media_temporal_{tag}.json")["n_varreduras"]} scans; '
                     f'reliable {100 * frac:.0f}%')
        if u == 'solar':
            ax.set_xlabel('solar X [arcsec]'); ax.set_ylabel('solar Y [arcsec]')
        else:
            ax.set_xlabel(f'x [{u}]'); ax.set_ylabel(f'y [{u}]')
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01, extend='both')
    cb.set_label(r'$B_\parallel f$ [G]  (divided by $g$ = ' + f'{g:.3f})')
    salvar(fig, 'figA6_maps')


def fig8(sinal_correcao=+1):
    """Mosaicos CRISP2 (89, corrigidos de α) ÷ g, em coordenadas heliocêntricas (geometria e correções do 92),
    com o véu das regiões não confiáveis (máscaras v3 do 110)."""
    g = ganhos_108()['CRISP2']
    r92 = {tag_de_arquivo(r['arquivo']): r for r in ler('92_sst_sobreposicao_ladrilhos.json')['data']}
    r110 = {c['etiqueta']: c for c in ler('110_sst_limite_campo_fraco_v3.json')['confiabilidade']}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 86 * MM), constrained_layout=True)
    vmax, passo = 200, 4
    cm = plt.get_cmap('RdBu_r').copy(); cm.set_bad((0, 0, 0, 0))          # NaN transparente (ladrilhos sobrepostos)
    for ax, tag in zip(axs, CRISP2_TAGS):
        geo = {gg['t']: gg for gg in r92[tag]['geometria']}
        corr = r92[tag]['correcoes_apontamento_arcsec']
        fr = []
        xl, yl = [np.inf, -np.inf], [np.inf, -np.inf]
        for t in range(4):
            B = np.load(DATA / f'blos_f_crisp2_{tag}_t{t}.npy').astype(float) / g
            M = np.load(DATA / f'confiavel_v3_{tag}_t{t}.npy').astype(bool)
            ny, nx = min(B.shape[0], M.shape[0]), min(B.shape[1], M.shape[1])
            B, M = B[:ny, :nx], M[:ny, :nx]
            val = np.isfinite(B)
            Ms = gaussian_filter(M.astype(float), 8) > 0.5
            esc = geo[t]['escala_arcsec_px']
            cx, cy = (np.asarray(geo[t]['centro']) + sinal_correcao * np.asarray(corr[str(t)]))
            ext = [cx - nx / 2 * esc, cx + nx / 2 * esc, cy - ny / 2 * esc, cy + ny / 2 * esc]
            Bd = B[::passo, ::passo]
            im = ax.imshow(np.clip(Bd, -vmax, vmax), origin='lower', cmap=cm, norm=TwoSlopeNorm(0, -vmax, vmax),
                           extent=ext, interpolation='nearest', rasterized=True)
            im.set_array(np.ma.masked_invalid(np.clip(Bd, -vmax, vmax)))
            veu = np.zeros(Bd.shape + (4,)); veu[..., :3] = 1.0
            veu[..., 3] = np.where((val & ~Ms)[::passo, ::passo], 0.55, 0.0)
            ax.imshow(veu, origin='lower', extent=ext, interpolation='nearest', rasterized=True)
            ys, xs_ = np.where(val)
            xl = [min(xl[0], ext[0] + xs_.min() * esc), max(xl[1], ext[0] + (xs_.max() + 1) * esc)]
            yl = [min(yl[0], ext[2] + ys.min() * esc), max(yl[1], ext[2] + (ys.max() + 1) * esc)]
            m = [q for q in r110.get(tag, {}).get('mapas', []) if q['t'] == t]
            fr.append(m[0]['frac_confiavel'] if m else float((M & val).sum() / val.sum()))
        ax.set_xlim(xl); ax.set_ylim(yl)
        ax.set_facecolor('0.85')
        hh = tag[9:11] + ':' + tag[11:13]
        ax.set_title(f'{DS[tag][0]}, 2×2 mosaic; reliable {100 * min(fr):.0f}–{100 * max(fr):.0f}%')
        ax.set_xlabel('solar X [arcsec]'); ax.set_ylabel('solar Y [arcsec]')
        ax.set_aspect('equal')
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01, extend='both')
    cb.set_label(r'$B_\parallel f$ [G]  (divided by $g$ = ' + f'{g:.3f})')
    salvar(fig, 'figA8_maps_crisp2')


INST4 = [('IBIS/DST', 'IBIS', '#56B4E9', 'D'), ('CRISP/SST', 'CRISP', '#0072B2', 'o'),
         ('CRISP2/SST', 'CRISP2', '#009E73', '^'), ('ViSP/DKIST', 'ViSP', '#D55E00', 's')]


def fig9():
    """Parte sintética (motor v3): perfis I e V por instrumento e B·f₅₀ (107)."""
    import Src.physics as P
    import Src.physics_v3 as V3
    st0, st1 = V3.get_stokes(0.0), V3.get_stokes(100.0, filling=1.0)
    lam0 = P.CALIB['lambda0'] * (1 + V3.V_CORE_KMS_V3 / P.CALIB['c_light']) * 10
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W2, 66 * MM), layout='constrained',
                                  gridspec_kw={'width_ratios': [1, 1, 0.9]})
    m = np.abs(P.LAM * 10 - lam0) <= 1.0
    a.plot(P.LAM[m] * 10 - lam0, st0['I'][m], color=INK, lw=1.2, label='atlas-fitted core')
    b.plot(P.LAM[m] * 10 - lam0, st1['V'][m] * 1e2, color=INK, lw=1.2)
    for nm, lab, cor, mk in INST4:
        ins = V3.INSTRUMENTAL_V3[nm]
        d0 = V3.degradar_stokes(st0, P.LAM, ins, rng_local=np.random.default_rng(0), fator_ruido=0.0)
        d1 = V3.degradar_stokes(st1, P.LAM, ins, rng_local=np.random.default_rng(0), fator_ruido=0.0)
        x = np.asarray(d0['lam_obs']) * 10 - lam0
        ms = 2.5 if len(x) < 100 else 0
        ls = '-' if len(x) > 100 else '-'
        a.plot(x, d0['I'], ls, marker=mk if ms else None, ms=ms, color=cor, lw=0.7, label=lab)
        b.plot(x, d1['V'] * 1e2, ls, marker=mk if ms else None, ms=ms, color=cor, lw=0.7)
    for ax in (a, b):
        ax.set_xlim(-1.0, 1.0); ax.set_xlabel(r'$\Delta\lambda$ [Å]')
    a.set_ylabel(r'$I/I_\mathrm{c}$'); b.set_ylabel(r'$V/I_\mathrm{c}$  [$10^{-2}$]  ($B_\parallel f$ = 100 G)')
    a.set_ylim(0.15, 0.75)
    fig.legend(*a.get_legend_handles_labels(), loc='outside lower center', ncol=5, handlelength=1.6, columnspacing=1.2)
    letra(a, 'a'); letra(b, 'b')
    r = ler('107_limites_sinteticos_v3.json')['data']['resultados']
    for i, (nm, lab, cor, mk) in enumerate(INST4):
        for rot, mf in (('1 varredura', 'none'), ('média de 11', cor)):
            q = r['v3'][nm][rot]
            c.errorbar(q['bf50'], i, xerr=[[q['bf50'] - q['ic95'][0]], [q['ic95'][1] - q['bf50']]], fmt=mk,
                       color=cor, mfc=mf, mew=0.8, ms=4, elinewidth=0.7, capsize=0)
        q2 = r['v2-nucleo'][nm]['1 varredura']['bf50']
        c.plot(q2, i, 'x', color=MUTED, ms=4, mew=0.8)
    c.set_yticks(range(len(INST4))); c.set_yticklabels([x[1] for x in INST4])
    c.invert_yaxis(); c.tick_params(axis='y', length=0, right=False)
    c.set_xscale('log'); c.set_xlim(1, 80)
    c.set_xticks([1, 2, 5, 10, 20, 50]); c.set_xticklabels(['1', '2', '5', '10', '20', '50']); c.minorticks_off()
    c.set_xlabel(r'$a_{50}$ [G] (5% false alarm)')
    from matplotlib.lines import Line2D
    c.legend(handles=[Line2D([], [], marker='o', color=MUTED, mfc='none', ls='', label='1 scan'),
                      Line2D([], [], marker='o', color=MUTED, ls='', label='mean of 11'),
                      Line2D([], [], marker='x', color=MUTED, ls='', label='1 scan, core of Paper B')],
             loc='lower left', fontsize=8, handletextpad=0.2)
    c.set_ylim(5.3, -0.5)
    letra(c, 'c', x=0.86)
    salvar(fig, 'figA9_synthetic')


def fig10():
    """Calibração do estimador de campo fraco (108): ganho e desvio do pull por instrumento."""
    r = ler('108_regressao_bf_v3.json')['data']['resultados']
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 55 * MM), constrained_layout=True)
    ests = [('WF-mapa', 'map filter', -0.12), ('WF-pixel (todos)', 'per-pixel WFA', 0.12)]
    for i, (nm, lab, cor, mk) in enumerate(INST4):
        for en, elab, dx in ests:
            mf = cor if en == 'WF-mapa' else 'none'
            g = r[nm][en]['1.000']['Quieta+Rede']['ganho']
            a.plot(i + dx, g, mk, color=cor, mfc=mf, mew=0.8, ms=4.5)
            for k, dk in (('1.000', -0.04), ('0.302', 0.04)):
                q = r[nm][en][k]['Quieta+Rede']
                b.plot(i + dx + dk, q['pull_std'], mk, color=cor, mfc=mf, mew=0.8, ms=4)
    for ax in (a, b):
        ax.set_xticks(range(len(INST4))); ax.set_xticklabels([x[1] for x in INST4])
        ax.tick_params(axis='x', length=0, top=False)
        ax.set_xlim(-0.6, len(INST4) - 0.4)
    a.axhspan(0.95, 1.05, color='0.92', lw=0, zorder=0); a.axhline(1, color=MUTED, lw=0.5)
    a.set_ylabel(r'gain $g=\hat a/(B_\parallel f)$ (noise-free)')
    a.set_ylim(0.98, 1.07)
    b.axhspan(0.9, 1.1, color='0.92', lw=0, zorder=0); b.axhline(1, color=MUTED, lw=0.5)
    b.set_ylabel('robust std of the pull')
    b.set_ylim(0.85, 1.15)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker='o', color=MUTED, ls='', label='map filter (fixed template)'),
                      Line2D([], [], marker='o', color=MUTED, mfc='none', ls='', label='per-pixel WFA')],
             loc='upper right', fontsize=8)
    b.text(0.98, 0.04, 'quiet + network; left: 1 scan, right: mean of 11', transform=b.transAxes, ha='right',
           fontsize=8)
    letra(a, 'a'); letra(b, 'b')
    salvar(fig, 'figA10_calibration')


def fig7():
    r107 = ler('107_limites_sinteticos_v3.json')['data']
    z_c, z50 = r107['z_c'], r107['z50']
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 74 * MM), constrained_layout=True,
                               gridspec_kw={'width_ratios': [1.3, 1]})
    xx = np.logspace(np.log10(0.3), np.log10(150), 300)
    for inst, tag in (('CRISP/SST', CRISP_TAGS[0]), ('CRISP2/SST', CRISP2_TAGS[0])):
        lab, cor, mk = DS[tag]
        for rot, ls, mf in (('1 varredura', '-', 'none'), ('média de 11', '--', cor)):
            q = r107['resultados']['v3'][inst][rot]
            c = [p for p in q['curva'] if p['n'] > 0 and p['bf'] >= 0.3]
            a.plot([p['bf'] for p in c], [p['p'] for p in c], mk, color=cor, mfc=mf, mew=0.7, ms=3.2)
            s = q['bf50_teoria'] / z50
            a.plot(xx, norm.sf(z_c - xx / s) + norm.cdf(-xx / s - z_c), ls, color=cor, lw=0.8,
                   label=f'{inst.split("/")[0]}, {"1 scan" if rot.startswith("1") else "mean of 11"} ({q["bf50"]:.1f} G)')
    a.axhline(0.5, color=MUTED, lw=0.4); a.axhline(0.05, color=MUTED, lw=0.4, ls=':')
    a.set_xscale('log'); a.set_xlim(0.3, 150); a.set_ylim(0, 1.02)
    a.set_xticks([1, 10, 100]); a.set_xticklabels(['1', '10', '100'])
    a.set_xlabel(r'$B_\parallel f$ [G]'); a.set_ylabel('detection probability (5% false alarm)')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.6, columnspacing=0.8)
    letra(a, 'a')

    linhas = tabela_a50()
    ylab, y = [], 0
    for L in linhas:
        cor, mk = L['cor'], L['mk']
        if L['medido'] is not None:
            lo, hi = L['medido']
            b.plot([lo, hi] if hi != lo else [lo], [y] * (2 if hi != lo else 1), mk + '-', color=cor, ms=4, lw=1.2, zorder=4)
        for k, (mk2, fc) in (('v2', ('x', None)), ('v3', ('D', 'white'))):
            if L[k] is not None:
                b.plot(L[k], y, mk2, color=INK, ms=4 if k == 'v3' else 4.5, mfc=fc, mew=0.8, zorder=3)
        ylab.append(L['rotulo']); y += 1
    b.set_yticks(range(len(ylab))); b.set_yticklabels(ylab, fontsize=8)
    b.invert_yaxis(); b.tick_params(axis='y', length=0, right=False)
    b.set_xscale('log'); b.set_xlim(4, 80)
    b.set_xticks([5, 10, 20, 50]); b.set_xticklabels(['5', '10', '20', '50']); b.minorticks_off()
    b.set_xlabel(r'$a_{50}$ in true $B_\parallel f$ [G]')
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], marker='o', color=MUTED, ls='', label='measured'),
                      Line2D([], [], marker='x', color=INK, ls='', label='predicted, core of Paper B'),
                      Line2D([], [], marker='D', color=INK, mfc='white', ls='', label='predicted, atlas-fitted core')],
             loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, handletextpad=0.3)
    letra(b, 'b', x=0.02, y=0.07)
    salvar(fig, 'figA7_detection')


def tabela_a50():
    """a₅₀ em B_los·f verdadeiro: medidos (z = 1,96) ÷ g; previstos v2 (104) e v3 (107)."""
    g = ganhos_108()
    r104 = ler('104_sst_limites_previsto_medido.json')
    r107 = ler('107_limites_sinteticos_v3.json')['data']['resultados']['v3']
    v3 = {k: {'uma': r107[f'{k}/SST']['1 varredura']['bf50'], 'N11': r107[f'{k}/SST']['média de 11']['bf50']}
          for k in ('CRISP', 'CRISP2')}
    P2, P3 = r104['previsto']['P2_CRISP_grade_real'], r104['previsto']['P3_CRISP2_grade_real']
    L = []
    for tag in CRISP_TAGS:
        lab, cor, mk = DS[tag]
        mc = r104['medido_crisp'][tag]
        L.append({'rotulo': f'{lab}, 1 scan', 'cor': cor, 'mk': mk,
                  'medido': (mc['a50_uma_gauss_G'] / g['CRISP'],) * 2, 'v2': P2['a50_branqueado_G'], 'v3': v3['CRISP']['uma'],
                  'medido_zc': mc['a50_uma_G'] / g['CRISP'], 'zc': mc['z_c_uma']})
    for tag in CRISP_TAGS:
        lab, cor, mk = DS[tag]
        mc = r104['medido_crisp'][tag]
        r101 = dados(f'101_sst_media_temporal_{tag}.json')
        L.append({'rotulo': f'{lab}, N = 11', 'cor': cor, 'mk': mk,
                  'medido': (r101['a50_N11_extrapolado_G'] / g['CRISP'],) * 2, 'v2': P2['a50_N11_branqueado_G'],
                  'v3': v3['CRISP']['N11'], 'medido_zc': mc['a50_N11_G'] / g['CRISP'], 'zc': mc['z_c_media'],
                  'nota': f'scaled from the mean of {r101["n_varreduras"]} scans'})
    for tag in CRISP2_TAGS:
        lab, cor, mk = DS[tag]
        mc = r104['medido_crisp2'][tag]
        L.append({'rotulo': f'{lab}, 1 scan', 'cor': cor, 'mk': mk,
                  'medido': (mc['a50_uma_min_G'] / g['CRISP2'], mc['a50_uma_max_G'] / g['CRISP2']),
                  'v2': P3['a50_pesos89_G'], 'v3': v3['CRISP2']['uma'], 'medido_zc': None, 'zc': None})
    return L


def escrever_tabela():
    L = tabela_a50()
    g = ganhos_108()
    linhas = ['dataset,measured_G,measured_empirical_zc_G,predicted_v2_G,predicted_v3_G,note']
    tex = [r'\begin{tabular}{lcccc}', r'\hline',
           r'Data set & measured & measured ($z_c$ emp.) & core of Paper B & atlas-fitted core \\', r'\hline']
    for d in L:
        lo, hi = d['medido']
        med = f'{lo:.1f}' if abs(hi - lo) < 0.05 else f'{lo:.1f}–{hi:.1f}'
        mzc = '' if d.get('medido_zc') is None else f'{d["medido_zc"]:.1f}'
        nota = d.get('nota', '')
        linhas.append(f'{d["rotulo"]},{med},{mzc},{d["v2"]:.1f},{d["v3"]:.1f},{nota}')
        tex.append(f'{d["rotulo"]} & {med.replace("–", "--")} & {mzc or "--"} & {d["v2"]:.1f} & {d["v3"]:.1f} \\\\')
    tex += [r'\hline', r'\end{tabular}',
            f'% a50 in true B_los*f (G), 5% false alarm; measured values divided by the estimator gain g '
            f'(CRISP {g["CRISP"]:.3f}, CRISP2 {g["CRISP2"]:.3f}; script 108).']
    (REP / '109_tabela_a50.csv').write_text('\n'.join(linhas) + '\n', encoding='utf-8')
    (REP / '109_tabela_a50.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    print('  109_tabela_a50.csv/.tex')
    for s in linhas:
        print('    ' + s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', type=int, nargs='*', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument('--escala-crisp', type=float, default=None, help='arcsec/pixel do CRISP (eixos da Fig. 6)')
    args = ap.parse_args()
    print(f'Figuras em {OUT}')
    fun = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: lambda: fig6(args.escala_crisp), 7: fig7, 8: fig8, 9: fig9, 10: fig10}
    for n in args.fig:
        fun[n]()
    escrever_tabela()


if __name__ == '__main__':
    main()
