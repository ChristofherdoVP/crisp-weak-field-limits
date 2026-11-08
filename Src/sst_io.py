"""
sst_io.py — Leitura dos cubos do SST/CRISP (formato SOLARNET / SSTRED).

Funções compartilhadas pelos scripts que leem dados reais (55 em diante):

  abrir_fits           : abre SEM carregamento preguiçoso (lista todas as HDUs);
                         se o astropy falhar, percorre o arquivo em blocos de 2880 bytes
  identificar_eixos    : tempo, Stokes, λ, y, x pelo CTYPE (ou pelos tamanhos)
  cobertura, faixas    : quais varreduras têm dados (lê ~3 linhas por varredura)
  comprimentos_de_onda : decodifica o eixo espectral, inclusive WAVE-TAB
                         (norma FITS WCS -TAB, Greisen et al. 2006, A&A 446, 747, §6),
                         conferindo com astropy.wcs quando possível
  indice_stokes, plano : lê um plano 2D (varredura, Stokes, λ) sem carregar o cubo

O arquivo nunca é modificado (modo somente leitura, memmap).
"""

import os
import tempfile

import numpy as np
from astropy.io import fits

BLOCO = 2880
UNIDADE_PARA_NM = {'nm': 1.0, 'angstrom': 0.1, 'a': 0.1, 'å': 0.1, 'm': 1e9,
                   'um': 1e3, 'µm': 1e3, 'pm': 1e-3, 'cm': 1e7, 'mm': 1e6}


# -----------------------------------------------------------------------
# Abertura e listagem
# -----------------------------------------------------------------------
def _tam_dados(hdr):
    """Bytes de dados de uma HDU (sem o preenchimento até 2880)."""
    naxis = int(hdr.get('NAXIS', 0))
    if naxis == 0:
        return 0
    n = 1
    for i in range(1, naxis + 1):
        n *= int(hdr.get(f'NAXIS{i}', 0))
    return abs(int(hdr.get('BITPIX', 8))) // 8 * int(hdr.get('GCOUNT', 1)) * (int(hdr.get('PCOUNT', 0)) + n)


def _pad(n):
    return ((n + BLOCO - 1) // BLOCO) * BLOCO


def _ler_cabecalho(f, pos, tam):
    """Lê um cabeçalho FITS que começa em pos. Devolve (Header, início dos dados) ou None."""
    f.seek(pos)
    buf = b''
    while True:
        b = f.read(BLOCO)
        if len(b) < BLOCO:
            return None
        buf += b
        for i in range(0, BLOCO, 80):
            carta = b[i:i + 80]
            if carta[:3] == b'END' and carta[3:].strip() == b'':
                return fits.Header.fromstring(buf.decode('ascii', errors='replace')), pos + len(buf)
        if len(buf) > 1000 * BLOCO or not buf[:BLOCO].strip(b'\x00 '):
            return None                                   # não é cabeçalho (ex.: bloco de zeros)


def extensoes_manual(caminho, max_hdus=64):
    """
    Percorre o arquivo cabeçalho a cabeçalho, sem o astropy.
    Devolve [(offset_cabecalho, Header, offset_dados, n_bytes_dados)].
    Serve de diagnóstico quando fits.open não enxerga as extensões.
    """
    out = []
    with open(caminho, 'rb') as f:
        tam = os.fstat(f.fileno()).st_size
        pos = 0
        while pos < tam and len(out) < max_hdus:
            r = _ler_cabecalho(f, pos, tam)
            if r is None:
                break
            hdr, ini = r
            n = _tam_dados(hdr)
            out.append((pos, hdr, ini, n))
            pos = ini + _pad(n)
    return out


def varrer_cauda(caminho, inicio, chunk_blocos=23302):
    """
    Examina os bytes de 'inicio' até o fim do arquivo: fração de bytes ≠ 0 e
    posições de 'XTENSION=' (alinhadas ou não a 2880). Lê em blocos de ~64 MB.
    """
    achados, nao_nulos, total = [], 0, 0
    passo = chunk_blocos * BLOCO
    with open(caminho, 'rb') as f:
        tam = os.fstat(f.fileno()).st_size
        pos = inicio
        while pos < tam:
            f.seek(pos)
            buf = f.read(min(passo + 16, tam - pos))
            if not buf:
                break
            util = buf[:passo] if len(buf) > passo else buf
            nao_nulos += int(np.count_nonzero(np.frombuffer(util, np.uint8)))
            total += len(util)
            k = buf.find(b'XTENSION=')
            while k != -1 and k < len(util):
                achados.append(pos + k)
                k = buf.find(b'XTENSION=', k + 1)
            pos += len(util)
    return {'inicio': int(inicio), 'bytes': int(total), 'frac_nao_nulos': nao_nulos / max(total, 1),
            'xtension_offsets': achados,
            'alinhados_2880': [int((a - inicio) % BLOCO == 0) for a in achados]}


def carregar_extensao_manual(caminho, offset, ini_dados, n_dados):
    """Lê uma extensão pelos offsets e devolve a HDU (via arquivo temporário)."""
    with open(caminho, 'rb') as f:
        f.seek(offset)
        bruto = f.read(ini_dados - offset + _pad(n_dados))
    tmp = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
    try:
        prim = fits.PrimaryHDU()
        tmp.write(prim.header.tostring().encode('ascii'))
        tmp.write(bruto)
        tmp.close()
        with fits.open(tmp.name, memmap=False) as h:
            hdu = h[1].copy()
            _ = hdu.data                                    # força a leitura
        return hdu
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def abrir_fits(caminho):
    """
    Abre o arquivo lendo TODOS os cabeçalhos (lazy_load_hdus=False).
    Devolve (hdul, lista_de_hdus, extras, avisos, diag).

    Se o astropy só enxergar a HDU primária, percorre o arquivo manualmente;
    se ainda assim não houver extensões e sobrarem bytes depois dos dados da
    primária, varre esse trecho procurando 'XTENSION=' (diag['cauda']).
    Extensões pequenas (< 200 MB) encontradas assim são carregadas em 'extras'.
    """
    avisos, diag = [], {}
    try:
        hdul = fits.open(caminho, memmap=True, lazy_load_hdus=False)
    except Exception as e:                                           # noqa: BLE001
        avisos.append(f'fits.open (não preguiçoso) falhou: {type(e).__name__}: {e}')
        hdul = fits.open(caminho, memmap=True, lazy_load_hdus=True)
    lista = []
    for i, h in enumerate(hdul):
        lista.append({'indice': i, 'nome': h.name, 'tipo': type(h).__name__,
                      'shape': list(h.shape) if getattr(h, 'shape', None) else [],
                      'origem': 'astropy'})
    extras = []
    if len(lista) > 1:
        return hdul, lista, extras, avisos, diag

    manuais = extensoes_manual(caminho)
    candidatos = [(off, hdr, ini, n) for off, hdr, ini, n in manuais[1:]]
    if manuais:
        _, h0, ini0, n0 = manuais[0]
        fim_prim = ini0 + _pad(n0)
        tam = os.path.getsize(caminho)
        diag['primaria'] = {'bytes_cabecalho': int(ini0), 'bytes_dados': int(n0),
                            'fim_primaria': int(fim_prim), 'tamanho_arquivo': int(tam),
                            'bytes_depois_da_primaria': int(tam - fim_prim)}
        if not candidatos and tam - fim_prim > 0:
            cauda = varrer_cauda(caminho, fim_prim)
            diag['cauda'] = cauda
            avisos.append(f'{(tam - fim_prim)/1e6:.1f} MB depois dos dados da primária: '
                          f'{100*cauda["frac_nao_nulos"]:.3f}% de bytes ≠ 0, '
                          f'{len(cauda["xtension_offsets"])} cabeçalho(s) XTENSION')
            with open(caminho, 'rb') as f:
                for off in cauda['xtension_offsets']:
                    r = _ler_cabecalho(f, off, tam)
                    if r is not None:
                        candidatos.append((off, r[0], r[1], _tam_dados(r[0])))
    for j, (off, hdr, ini, n) in enumerate(candidatos, start=1):
        d = {'indice': j, 'nome': str(hdr.get('EXTNAME', '')), 'tipo': str(hdr.get('XTENSION', '?')),
             'shape': [int(hdr.get(f'NAXIS{k}', 0)) for k in range(int(hdr.get('NAXIS', 0)), 0, -1)],
             'origem': f'manual (offset {off})', 'bytes': int(n)}
        lista.append(d)
        if n < 200e6:
            try:
                extras.append(carregar_extensao_manual(caminho, off, ini, n))
            except Exception as e:                                   # noqa: BLE001
                avisos.append(f'extensão {j} ({d["nome"]}) não carregou: {type(e).__name__}: {e}')
    if candidatos:
        avisos.append(f'astropy listou 1 HDU; a busca manual encontrou mais {len(candidatos)}')
    return hdul, lista, extras, avisos, diag


def descrever_tabela(h):
    """Colunas de uma BinTableHDU com formato e dimensões."""
    out = []
    for c in h.columns:
        try:
            shp = list(np.asarray(h.data[c.name]).shape)
        except Exception:                                            # noqa: BLE001
            shp = None
        out.append({'nome': c.name, 'formato': str(c.format), 'dim': str(c.dim), 'shape': shp,
                    'unidade': str(c.unit) if c.unit else ''})
    return out


# -----------------------------------------------------------------------
# Eixos e varreduras
# -----------------------------------------------------------------------
def identificar_eixos(hdr, shape):
    """({'t','s','w','y','x'} → eixo numpy, origem). Eixo FITS i ↔ numpy ndim − i."""
    nd = len(shape)
    eixos, fonte = {}, 'CTYPE'
    for i in range(1, nd + 1):
        ct = str(hdr.get(f'CTYPE{i}', '')).upper()
        ax = nd - i
        if 'WAVE' in ct or 'AWAV' in ct or 'FREQ' in ct:
            eixos['w'] = ax
        elif 'STOKES' in ct:
            eixos['s'] = ax
        elif 'TIME' in ct or 'UTC' in ct:
            eixos['t'] = ax
        elif ct.startswith(('HPLN', 'SOLX', 'X')):
            eixos['x'] = ax
        elif ct.startswith(('HPLT', 'SOLY', 'Y')):
            eixos['y'] = ax
    if 'w' in eixos and 'x' in eixos and 'y' in eixos:
        return eixos, fonte
    fonte = 'inferido pelos tamanhos'
    ordem = sorted(range(nd), key=lambda a: shape[a], reverse=True)
    eixos = {'y': min(ordem[:2]), 'x': max(ordem[:2])}
    resto = [a for a in range(nd) if a not in eixos.values()]
    s = [a for a in resto if shape[a] == 4]
    if s:
        eixos['s'] = s[0]
        resto.remove(s[0])
    w = [a for a in resto if 5 <= shape[a] <= 100]
    if w:
        eixos['w'] = w[-1]
        resto.remove(w[-1])
    if resto:
        eixos['t'] = resto[0]
    return eixos, fonte


def _indice(eixos, shape, t, s, w):
    idx = []
    for a in range(len(shape)):
        if a == eixos.get('t'):
            idx.append(t)
        elif a == eixos.get('s'):
            idx.append(s)
        elif a == eixos.get('w'):
            idx.append(w)
        else:
            idx.append(slice(None))
    return tuple(idx)


def plano(hdu, eixos, shape, t, s, w):
    """Plano 2D (y, x) em float32 para (varredura t, Stokes s, λ w)."""
    p = np.asarray(hdu.section[_indice(eixos, shape, t, s, w)], dtype=np.float32)
    return p.T if eixos['y'] > eixos['x'] else p


def cobertura(hdu, eixos, shape, linhas=(0.25, 0.5, 0.75)):
    """Fração de valores finitos e ≠ 0 em ~3 linhas por varredura (Stokes 0, λ do meio)."""
    nt = shape[eixos['t']] if 't' in eixos else 1
    frac = []
    for t in range(nt):
        vals = []
        for f in linhas:
            idx = list(_indice(eixos, shape, t, 0, shape[eixos['w']] // 2))
            idx[eixos['y']] = int(f * shape[eixos['y']])
            try:
                vals.append(np.asarray(hdu.section[tuple(idx)], dtype=np.float64).ravel())
            except (TypeError, ValueError, OSError):
                vals.append(np.zeros(1))
        v = np.concatenate(vals)
        frac.append(float((np.isfinite(v) & (v != 0)).mean()))
    return np.array(frac)


def mapa_de_dados(hdu, eixos, shape, t, n_linhas=16):
    """
    Para a varredura t: fração de valores finitos e ≠ 0 em cada plano (Stokes, λ), na ORDEM
    EM QUE ESTÃO NO ARQUIVO, amostrando n_linhas linhas por plano. Serve para achar onde os
    dados acabam: uma gravação ou download interrompido deixa um prefixo com dados e o resto zero.
    Devolve lista de (s, w, fração, fração por linha).
    """
    ns = shape[eixos['s']] if 's' in eixos else 1
    nw = shape[eixos['w']]
    ny = shape[eixos['y']]
    linhas = np.linspace(0, ny - 1, n_linhas).astype(int)
    ordem = sorted([(s, w) for s in range(ns) for w in range(nw)],
                   key=lambda sw: (sw[0], sw[1]) if eixos.get('s', -1) < eixos['w'] else (sw[1], sw[0]))
    out = []
    for s_, w_ in ordem:
        fr = []
        for y in linhas:
            idx = list(_indice(eixos, shape, t, s_, w_))
            idx[eixos['y']] = int(y)
            try:
                v = np.asarray(hdu.section[tuple(idx)], dtype=np.float64)
                fr.append(float((np.isfinite(v) & (v != 0)).mean()))
            except (TypeError, ValueError, OSError):
                fr.append(0.0)
        out.append((s_, w_, float(np.mean(fr)), fr))
    return out


def faixas(indices):
    """[0,1,2,5,6] → '0–2, 5–6'."""
    indices = list(indices)
    if not indices:
        return 'nenhuma'
    out, ini, ant = [], indices[0], indices[0]
    for i in indices[1:] + [None]:
        if i is not None and i == ant + 1:
            ant = i
            continue
        out.append(f'{ini}' if ini == ant else f'{ini}–{ant}')
        if i is not None:
            ini = ant = i
    return ', '.join(out)


def indice_stokes(hdr, eixos, shape, codigo):
    """Índice numpy do parâmetro de Stokes (1 = I, 2 = Q, 3 = U, 4 = V)."""
    if 's' not in eixos:
        return None
    i = len(shape) - eixos['s']
    n = shape[eixos['s']]
    crval = float(hdr.get(f'CRVAL{i}', 1.0))
    cdelt = float(hdr.get(f'CDELT{i}', 1.0))
    crpix = float(hdr.get(f'CRPIX{i}', 1.0))
    codigos = crval + (np.arange(n) + 1 - crpix) * cdelt
    k = np.flatnonzero(np.isclose(codigos, codigo))
    return int(k[0]) if k.size else codigo - 1


# -----------------------------------------------------------------------
# Eixo espectral (WAVE-TAB)
# -----------------------------------------------------------------------
def _fator_nm(unidade, hdr, valores):
    u = (unidade or '').strip().lower().replace('ångström', 'angstrom')
    if u in UNIDADE_PARA_NM:
        return UNIDADE_PARA_NM[u], u
    wu = hdr.get('WAVEUNIT')
    if wu is not None:
        try:
            return 10.0 ** (int(wu) + 9), f'WAVEUNIT={wu}'
        except (TypeError, ValueError):
            pass
    med = np.nanmedian(valores)
    if 500 < med < 1200:
        return 1.0, 'suposto nm'
    if 5000 < med < 12000:
        return 0.1, 'suposto Å'
    if 5e-7 < med < 1.2e-6:
        return 1e9, 'suposto m'
    return 1.0, 'desconhecida'


def _decodificar_tab(hdr, i, n, tabela):
    """
    Norma -TAB: ψ = CRVAL + CDELT·(p − CRPIX) (p = 1..n); com vetor de índices
    Ψ (PSi_2), Υ = índice fracionário de ψ em Ψ; senão Υ = ψ. A coordenada é
    a componente m = PVi_3 do arranjo de coordenadas (PSi_1), interpolada em Υ
    ao longo da dimensão K_m (as demais dimensões K ficam no primeiro elemento;
    a variação ao longo delas é devolvida).
    """
    col = str(hdr.get(f'PS{i}_1', '')).strip()
    col_idx = str(hdr.get(f'PS{i}_2', '')).strip()
    m = int(hdr.get(f'PV{i}_3', 1))
    C = np.asarray(tabela.data[col][0], dtype=np.float64)          # numpy: (K_M..K_1, M)
    if C.ndim == 1:
        C = C[:, None]
    Cm = C[..., m - 1]                                               # (K_M..K_1)
    nK = Cm.ndim
    ax = nK - m if nK >= m else nK - 1                               # dimensão K_m em numpy
    Cm = np.moveaxis(Cm, ax, -1)                                     # (..., K_m)
    vec = Cm.reshape(-1, Cm.shape[-1])
    variacao = float(np.nanmax(np.abs(vec - vec[0]))) if vec.shape[0] > 1 else 0.0
    vec = vec[0]
    crval = float(hdr.get(f'CRVAL{i}', 0.0))
    cdelt = float(hdr.get(f'CDELT{i}', 1.0))
    crpix = float(hdr.get(f'CRPIX{i}', 0.0))
    psi = crval + cdelt * (np.arange(1, n + 1) - crpix)
    if col_idx and col_idx in tabela.columns.names:
        Psi = np.asarray(tabela.data[col_idx][0], dtype=np.float64).ravel()
        ups = np.interp(psi, Psi, np.arange(1, len(Psi) + 1))
    else:
        ups = psi
    lam = np.interp(ups, np.arange(1, len(vec) + 1), vec)
    return lam, {'coluna': col, 'coluna_indice': col_idx or None, 'm': m, 'shape_arranjo': list(C.shape),
                 'variacao_entre_varreduras': variacao, 'crpix': crpix, 'cdelt': cdelt, 'crval': crval}


def _achar_tabela(hdul, extras, nome, extver=1):
    candidatos = [h for h in list(hdul) + list(extras) if isinstance(h, fits.BinTableHDU)]
    for h in candidatos:
        if h.name.upper() == str(nome).upper() and int(h.header.get('EXTVER', 1)) == int(extver):
            return h
    for h in candidatos:
        if h.name.upper() == str(nome).upper():
            return h
    return None


def comprimentos_de_onda(hdul, extras, hdr, i, n):
    """
    λ (em nm) do eixo FITS i com n pontos. Devolve dict com 'lam_nm' (ou None),
    'metodo', 'unidade', 'detalhes', 'astropy_nm' e 'dif_astropy_nm'.
    """
    ct = str(hdr.get(f'CTYPE{i}', '')).upper()
    un = str(hdr.get(f'CUNIT{i}', '')).strip()
    res = {'ctype': ct, 'cunit': un, 'lam_nm': None, 'metodo': None, 'detalhes': {},
           'astropy_nm': None, 'dif_astropy_nm': None, 'erros': []}
    lam = None
    if ct.endswith('-TAB'):
        nome = str(hdr.get(f'PS{i}_0', '')).strip()
        tab = _achar_tabela(hdul, extras, nome, hdr.get(f'PV{i}_1', 1)) if nome else None
        if tab is not None:
            try:
                lam, det = _decodificar_tab(hdr, i, n, tab)
                res['metodo'] = f'-TAB (norma), tabela {nome}'
                res['detalhes'] = det
            except Exception as e:                                   # noqa: BLE001
                res['erros'].append(f'decodificação -TAB: {type(e).__name__}: {e}')
        else:
            res['erros'].append(f'tabela PS{i}_0 = "{nome}" não encontrada')
        if lam is None:                                              # busca genérica
            for h in [h for h in list(hdul) + list(extras) if isinstance(h, fits.BinTableHDU)]:
                for c in h.columns.names:
                    if 'WAVE' in c.upper() and 'INDEX' not in c.upper():
                        A = np.asarray(h.data[c][0], dtype=np.float64)
                        for ax in range(A.ndim):
                            if A.shape[ax] == n:
                                v = np.moveaxis(A, ax, -1).reshape(-1, n)
                                for linha in v:
                                    if np.ptp(linha) > 0:
                                        lam = linha
                                        break
                            if lam is not None:
                                break
                    if lam is not None:
                        res['metodo'] = f'busca genérica: {h.name}.{c}'
                        break
                if lam is not None:
                    break
    elif f'CDELT{i}' in hdr or f'CRVAL{i}' in hdr:
        crpix = float(hdr.get(f'CRPIX{i}', 0.0))
        lam = float(hdr.get(f'CRVAL{i}', 0.0)) + (np.arange(1, n + 1) - crpix) * float(hdr.get(f'CDELT{i}', 1.0))
        res['metodo'] = 'WCS linear'
    if lam is None:
        return res
    fator, u = _fator_nm(un, hdr, lam)
    res['unidade'] = u
    res['lam_nm'] = np.asarray(lam, dtype=np.float64) * fator

    # Conferência independente com astropy.wcs (wcslib)
    try:
        from astropy.wcs import WCS
        hl = fits.HDUList([fits.PrimaryHDU(header=hdr)] + [h for h in list(hdul)[1:] + list(extras)
                                                           if isinstance(h, fits.BinTableHDU)])
        w = WCS(hdr, fobj=hl)
        pix = np.zeros((n, w.naxis))
        pix[:, i - 1] = np.arange(n)
        wl = np.asarray(w.wcs_pix2world(pix, 0))[:, i - 1]
        f2, _ = _fator_nm(str(w.wcs.cunit[i - 1]), hdr, wl)
        res['astropy_nm'] = wl * f2
        res['dif_astropy_nm'] = float(np.nanmax(np.abs(res['astropy_nm'] - res['lam_nm'])))
    except Exception as e:                                           # noqa: BLE001
        res['erros'].append(f'astropy.wcs: {type(e).__name__}: {str(e)[:200]}')
    return res
