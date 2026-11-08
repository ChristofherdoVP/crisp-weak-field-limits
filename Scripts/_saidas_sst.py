"""
_saidas_sst.py — nomes de saída e critério proporcional comuns aos scripts SST 79–84.

Motivo (2026-10-26): os scripts gravavam sempre o mesmo relatório/figura; rodar nas séries do CRISP
(2024-10-08) sobrescreveu os resultados dos mosaicos do CRISP2 (2025-08-19). Agora:
  - um relatório e figuras POR ARQUIVO, com a etiqueta AAAAMMDDThhmmss do nome do arquivo:
        Outputs/Reports/<prefixo>_<etiqueta>.json,  Outputs/Figures/<fig>_<etiqueta>.png
  - um resumo <prefixo>_resumo.json refeito a partir de TODOS os relatórios por arquivo existentes
    (rodar com um arquivo só atualiza a linha dele e não apaga as outras).
Critério por arquivo: "≥ 3 das 4 posições" passa a ser "≥ 75% das posições/varreduras"
(idêntico nos mosaicos de 4 posições; definido antes de reexecutar nas séries).
"""
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'Outputs' / 'Reports'
FIG = ROOT / 'Outputs' / 'Figures'
FRAC_MIN = 0.75
_PADRAO = re.compile(r'(\d{4})-(\d{2})-(\d{2})T(\d{2})[_:](\d{2})[_:](\d{2})')


def etiqueta(nome):
    """'nb_8542_2024-10-08T09_49_51_...fits' → '20241008T094951'."""
    m = _PADRAO.search(Path(nome).name)
    if m is None:
        return re.sub(r'[^A-Za-z0-9]+', '_', Path(nome).stem)[:40]
    return f'{m[1]}{m[2]}{m[3]}T{m[4]}{m[5]}{m[6]}'


def passa_fracao(n_ok, n):
    return bool(n > 0 and n_ok / n >= FRAC_MIN)


def figura(base, tag):
    FIG.mkdir(parents=True, exist_ok=True)
    return FIG / f'{base}_{tag}.png'


def gravar(prefixo, r, extra=None):
    """Grava o relatório de um arquivo; devolve o caminho."""
    r['etiqueta'] = etiqueta(r['arquivo'])
    REP.mkdir(parents=True, exist_ok=True)
    p = REP / f"{prefixo}_{r['etiqueta']}.json"
    with open(p, 'w', encoding='utf-8') as f:
        json.dump({'data': [r], **(extra or {})}, f, ensure_ascii=False, indent=2, default=str)
    return p


def resumo(prefixo, campos_fora=('posicoes', '_hist', 'lambda_nm')):
    """Refaz <prefixo>_resumo.json com o nível 'arquivo' de todos os relatórios por arquivo."""
    linhas = {}
    for a in sorted(REP.glob(f'{prefixo}_*.json')):
        if a.stem.endswith('_resumo'):
            continue
        try:
            d = json.loads(a.read_text(encoding='utf-8'))
        except Exception:
            continue
        for r in d.get('data', []):
            if 'arquivo' not in r:
                continue
            tag = r.get('etiqueta', etiqueta(r['arquivo']))
            if a.stem != f'{prefixo}_{tag}':        # ignora relatórios antigos combinados
                continue
            lin = {k: v for k, v in r.items() if k not in campos_fora}
            lin['n_posicoes'] = len(r.get('posicoes', r.get('pares', r.get('varreduras', []))))
            linhas[tag] = lin
    p = REP / f'{prefixo}_resumo.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump({'criterio_fracao_minima': FRAC_MIN, 'arquivos': linhas}, f, ensure_ascii=False, indent=2,
                  default=str)
    print(f"\nResumo de todos os arquivos já analisados ({p.name}):")
    for tag, lin in linhas.items():
        n_ok = lin.get('n_posicoes_ok', '')
        frac = f"  {n_ok}/{lin['n_posicoes']}" if n_ok != '' else ''
        print(f"  {tag}{frac}  →  {lin.get('decisao', '—')}")
    return p
