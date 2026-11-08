"""
91_sst_inspecionar_cabecalho.py — o que há num cubo SST/CRISP2, lendo só o cabeçalho (sem carregar os dados).

Para cada arquivo:
  - extensões (nome, tipo, forma ou colunas);
  - chaves principais (instrumento, datas, r₀, AO, apontamento, unidades, eixos);
  - grade λ (WCS-TAB), em mÅ em relação a 8542,09 Å, e se é simétrica em torno do próprio centro;
  - por passo t: centro do campo (HPLN, HPLT), μ = cos θ (distância ao centro do disco), hora de início
    e duração da varredura (coordenada TIME do WCS-TAB), e passo entre varreduras;
  - classificação: MOSAICO se o salto mediano do centro entre passos consecutivos for > 5″; senão SÉRIE TEMPORAL
    (uma série longa pode derivar alguns segundos de arco no total sem deixar de ser série).
Não há critério de decisão: é inspeção, para planejar a análise da série temporal.

Uso:  python Scripts/91_sst_inspecionar_cabecalho.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
Saída: Outputs/Reports/91_sst_inspecionar_cabecalho.json
"""
import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
LAMBDA0 = 8542.09
R_SOL_PADRAO = 948.0     # ″, aproximado para agosto

CHAVES = ['TELESCOP', 'INSTRUME', 'OBJECT', 'DATE-BEG', 'DATE-AVG', 'DATE-END', 'DATE-REF', 'DATEREF',
          'ATMOS_R0', 'AO_LOCK', 'ELEV_ANG', 'AZIMUTH', 'RSUN_ARC', 'SOLARNET', 'BUNIT', 'WAVEUNIT',
          'CADENCE', 'CADAVG', 'XPOSURE', 'TEXPOSUR', 'NSUMEXP', 'OBS_HDU', 'RELEASE', 'RELEASEC']


def jsonavel(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)


def coords_tab(h, hd):
    """Devolve (array de coordenadas, lista de nomes) do WCS-TAB, com o eixo de coordenadas no fim."""
    for i in range(1, hd.get('NAXIS', 0) + 1):
        if 'TAB' in str(hd.get(f'CTYPE{i}', '')).upper():
            ext, col = hd.get(f'PS{i}_0'), hd.get(f'PS{i}_1')
            try:
                arr = np.asarray(h[ext].data[col][0], dtype=np.float64)
            except Exception:
                return None, None
            return arr, str(col).split('+')
    return None, None


def inspecionar(caminho):
    out = {'arquivo': Path(caminho).name}
    with fits.open(caminho, memmap=True) as h:
        hd = h[0].header
        out['extensoes'] = []
        for e in h:
            info = {'nome': e.name, 'tipo': type(e).__name__}
            if getattr(e, 'columns', None) is not None:
                info['colunas'] = [(c.name, c.format, str(c.dim)) for c in e.columns]
            elif e.header.get('NAXIS', 0):
                info['forma'] = [e.header.get(f'NAXIS{k}') for k in range(e.header['NAXIS'], 0, -1)]
            out['extensoes'].append(info)
        forma = [hd.get(f'NAXIS{k}') for k in range(hd.get('NAXIS', 0), 0, -1)]
        out['forma_numpy'] = forma
        out['eixos'] = {f'CTYPE{i}': hd.get(f'CTYPE{i}') for i in range(1, hd.get('NAXIS', 0) + 1)}
        out['unidades'] = {f'CUNIT{i}': hd.get(f'CUNIT{i}') for i in range(1, hd.get('NAXIS', 0) + 1)}
        out['chaves'] = {k: jsonavel(hd[k]) for k in CHAVES if k in hd}
        r_sol = float(hd.get('RSUN_ARC', R_SOL_PADRAO))
        arr, nomes = coords_tab(h, hd)
        if arr is not None and arr.shape[-1] == len(nomes):
            idx = {n.strip().upper(): k for k, n in enumerate(nomes)}
            nt = forma[0] if len(forma) == 5 else 1
            nl = forma[2] if len(forma) == 5 else forma[0]
            # eixo de t e de λ no array do WCS-TAB: procura os eixos com esses tamanhos
            if 'WAVE' in idx:
                w = arr[..., idx['WAVE']]
                lam = np.unique(np.round(w.ravel(), 9))
                un = str(out['unidades'].get('CUNIT3', 'nm')).strip().lower()
                fat = {'nm': 10.0, 'm': 1e10, 'angstrom': 1.0}.get(un, 10.0)
                lamA = np.sort(lam) * fat
                if len(lamA) == nl:
                    rel = (lamA - LAMBDA0) * 1e3
                    centro = (rel[0] + rel[-1]) / 2
                    out['grade_lambda_mA'] = [round(float(x)) for x in rel]
                    out['grade_simetrica'] = bool(np.allclose(rel - centro, -(rel - centro)[::-1], atol=2))
                    out['grade_centro_mA'] = float(centro)
            if 'HPLN' in idx and 'HPLT' in idx and arr.ndim >= 2:
                eixo_t = next((k for k, n in enumerate(arr.shape[:-1]) if n == nt), None) if nt > 1 else None
                passos = []
                for t in range(nt):
                    sub = np.take(arr, t, axis=eixo_t) if eixo_t is not None else arr
                    x, y = float(np.mean(sub[..., idx['HPLN']])), float(np.mean(sub[..., idx['HPLT']]))
                    rr = np.hypot(x, y)
                    mu = float(np.sqrt(max(0.0, 1 - (rr / r_sol) ** 2)))
                    p = {'t': t, 'HPLN': x, 'HPLT': y, 'mu': mu}
                    if 'TIME' in idx:
                        tt = sub[..., idx['TIME']]
                        p['tempo_inicio_s'] = float(np.min(tt))
                        p['duracao_varredura_s'] = float(np.max(tt) - np.min(tt))
                    passos.append(p)
                out['passos'] = passos
                if nt > 1:
                    c = np.array([[p['HPLN'], p['HPLT']] for p in passos])
                    sep = np.hypot(*(c - c.mean(axis=0)).T)
                    salto = np.hypot(*np.diff(c, axis=0).T)
                    out['separacao_max_arcsec'] = float(sep.max())
                    out['salto_mediano_arcsec'] = float(np.median(salto))
                    out['tipo'] = 'MOSAICO' if np.median(salto) > 5 else 'SÉRIE TEMPORAL'
                    if 'TIME' in idx:
                        t0 = np.array([p['tempo_inicio_s'] for p in passos])
                        out['passo_entre_varreduras_s'] = float(np.median(np.diff(np.sort(t0))))
    return out


def imprimir(o):
    print('=' * 78)
    print(o['arquivo'])
    print(f"  forma (numpy): {o['forma_numpy']}   eixos: {', '.join(str(v) for v in o['eixos'].values())}")
    for k, v in o['chaves'].items():
        s = str(v)
        print(f"  {k} = {s[:110] + ('…' if len(s) > 110 else '')}")
    print('  extensões: ' + '; '.join(e['nome'] or 'PRIMARY' for e in o['extensoes']))
    if 'grade_lambda_mA' in o:
        print(f"  grade λ (mÅ vs {LAMBDA0}): {o['grade_lambda_mA']}")
        print(f"     simétrica em torno de {o['grade_centro_mA']:+.0f} mÅ: {'sim' if o['grade_simetrica'] else 'NÃO'}")
    if 'passos' in o:
        for p in o['passos'][:12]:
            extra = (f"  início {p['tempo_inicio_s']:.1f} s, varredura {p['duracao_varredura_s']:.1f} s"
                     if 'tempo_inicio_s' in p else '')
            print(f"  t={p['t']:3d}: HPLN {p['HPLN']:+8.1f}″  HPLT {p['HPLT']:+8.1f}″  μ = {p['mu']:.3f}{extra}")
        if len(o['passos']) > 12:
            print(f"  … ({len(o['passos'])} passos no total)")
        if 'tipo' in o:
            msg = (f"  → {o['tipo']} (salto mediano entre passos {o['salto_mediano_arcsec']:.1f}″, "
                   f"separação máxima {o['separacao_max_arcsec']:.1f}″)")
            if 'passo_entre_varreduras_s' in o:
                msg += f";  passo entre varreduras {o['passo_entre_varreduras_s']:.1f} s"
            print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    todos = []
    for c in sorted(a.arquivos, key=lambda x: Path(x).name):
        o = inspecionar(c)
        imprimir(o)
        todos.append(o)
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '91_sst_inspecionar_cabecalho.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nRelatório: {REP / '91_sst_inspecionar_cabecalho.json'}")


if __name__ == '__main__':
    main()
