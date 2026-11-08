"""
95_sst_chaves_variaveis.py — lê as extensões VAR-EXT-* (chaves que variam por varredura e/ou λ) de cubos SST.

Motivação (script 79 nos cubos CRISP de 2024-10-08): σ de Q/U ao longo da linha tem um padrão em zigue-zague
(sobe em λ3–4 e λ10–11, desce em λ2 e λ12) igual em todas as varreduras, que uma lei suave σ ∝ I^p não produz.
Hipótese: o número de quadros somados (FNUMSUM) muda com λ, e o ruído vai com 1/√N. E o ε do script 83 varia
de varredura para varredura (3,0–4,5×10⁻³ na série das 09:49): conferir contra r₀ e AO lock por varredura.
Só lê cabeçalhos e tabelas pequenas; não carrega os dados. Sem critério de decisão (inspeção).

Para cada arquivo e cada chave em CHAVES que exista como VAR-EXT-<chave>:
  - forma do array de valores e eixos (pelos nomes/dimensões da tabela);
  - se houver um eixo com o tamanho da grade λ: valor por λ (média nos outros eixos);
  - se houver um eixo com o número de varreduras: valor por varredura.

Uso:  python Scripts/95_sst_chaves_variaveis.py Data/RAW_SST/<arquivo>.fits [mais arquivos]
Saída: Outputs/Reports/95_sst_chaves_variaveis.json
"""
import json
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
CHAVES = ['FNUMSUM', 'NSUMEXP', 'XPOSURE', 'TEXPOSUR', 'ATMOS_R0', 'AO_LOCK', 'ELEV_ANG', 'SCANNUM', 'RESPAPPL']


def valores(hdu):
    """Devolve o array de valores de uma extensão VAR-EXT (coluna 'VALUES' ou a primeira coluna)."""
    cols = hdu.columns.names
    nome = 'VALUES' if 'VALUES' in cols else cols[0]
    return np.asarray(hdu.data[nome][0]), nome, cols


def inspecionar(caminho):
    out = {'arquivo': Path(caminho).name, 'chaves': {}}
    with fits.open(caminho, memmap=True) as h:
        forma = h[0].data.shape                      # (t, s, λ, y, x)
        nt, nl = forma[0], forma[2]
        out['forma'] = list(forma)
        for k in CHAVES:
            ext = f'VAR-EXT-{k}'
            if ext not in h:
                continue
            try:
                v, nome, cols = valores(h[ext])
            except Exception as e:
                out['chaves'][k] = {'erro': str(e)}
                continue
            v = np.asarray(v, dtype=float) if np.issubdtype(np.asarray(v).dtype, np.number) else v
            info = {'coluna': nome, 'colunas': cols, 'forma': list(np.shape(v))}
            if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
                info['min'] = float(np.nanmin(v)); info['max'] = float(np.nanmax(v))
                eixos = list(v.shape)
                if nl in eixos and nl != nt:
                    ax = eixos.index(nl)
                    info['por_lambda'] = [float(x) for x in np.nanmean(np.moveaxis(v, ax, 0).reshape(nl, -1), axis=1)]
                if nt in eixos and nt > 1:
                    ax = eixos.index(nt)
                    info['por_varredura'] = [float(x) for x in np.nanmean(np.moveaxis(v, ax, 0).reshape(nt, -1), axis=1)]
            out['chaves'][k] = info
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arquivos', nargs='+')
    a = ap.parse_args()
    todos = []
    for c in sorted(a.arquivos, key=lambda x: Path(x).name):
        o = inspecionar(c)
        todos.append(o)
        print('=' * 78)
        print(f"{o['arquivo']}   forma {o['forma']}")
        for k, i in o['chaves'].items():
            if 'erro' in i:
                print(f'  {k}: erro ao ler ({i["erro"]})')
                continue
            print(f"  {k}: forma {i['forma']}" + (f",  min {i['min']:.4g}  max {i['max']:.4g}" if 'min' in i else ''))
            if 'por_lambda' in i:
                print('     por λ:      ' + '  '.join(f'{x:.4g}' for x in i['por_lambda']))
            if 'por_varredura' in i:
                print('     por varredura: ' + '  '.join(f'{x:.4g}' for x in i['por_varredura']))
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / '95_sst_chaves_variaveis.json', 'w', encoding='utf-8') as f:
        json.dump({'data': todos}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nRelatório: {REP / '95_sst_chaves_variaveis.json'}")


if __name__ == '__main__':
    main()
