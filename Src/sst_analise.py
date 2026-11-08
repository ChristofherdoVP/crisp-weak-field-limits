"""
sst_analise.py — Análise de cubos SST/CRISP (CRISP e CRISP2) em Ca II 854,2 nm.

Camada acima de Src.sst_io (que só lê). Usada pelos scripts 63–66.

  CuboSST              : abre o arquivo, identifica eixos, λ (WAVE-TAB, WCS linear ou
                         --lambdas), Stokes I e V, varreduras com dados; lê um cubo
                         (λ, y, x) por varredura, em λ crescente
  mascaras_regioes     : quieto, calmo, plage, penumbra, umbra (limiares heurísticos
                         na intensidade da asa e do núcleo — documentados no relatório)
  normalizar_Ic        : Ic pelos pontos extremos contra o modelo (atlas + LSF + luz espúria)
  r0_por_varredura     : r0 de Fried das tabelas do arquivo, se existirem
  campo_fraco          : B_los (= B·f na nossa notação) por mínimos quadrados ponderados,
                         V = −C g λ0² B dI/dλ, com σ_B; aceita σ por ponto (ruído ∝ I)
  sigma_mad            : desvio padrão robusto

O arquivo nunca é modificado.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from astropy.io import fits

from Src.physics import CALIB, LAM, DLAM, V_CORE_KMS, INSTRUMENTAL_V2, get_stokes
from Src.sst_io import (abrir_fits, identificar_eixos, cobertura, comprimentos_de_onda,
                        indice_stokes, plano)

C_ZEE_A        = 4.6686e-13        # Å⁻¹ G⁻¹
LAMBDA0_AR_NM  = 854.2091          # Ca II 8542,09 Å (ar)
G_EFF          = CALIB['g_eff']
C_KMS          = CALIB['c_light']
LSF_CRISP_PM   = INSTRUMENTAL_V2['CRISP/SST']['fwhm_pm']
STRAY_CRISP    = INSTRUMENTAL_V2['CRISP/SST']['stray_light']
EPS_CRISP      = INSTRUMENTAL_V2['CRISP/SST']['ruido']['eps']

# Limiares das regiões (intensidades normalizadas à mediana do Sol quieto do campo)
LIMIARES = {'quieto_tol': 0.10, 'plage_nucleo': 1.3, 'plage_asa': 0.85,
            'umbra_asa': 0.50, 'penumbra_asa': 0.85}


def sigma_mad(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return float('nan')
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def ler_lambdas(texto, nw):
    """--lambdas: lista ou arquivo; aceita nm, Å ou Δλ em mÅ ao núcleo."""
    import re
    p = Path(texto)
    txt = p.read_text() if p.exists() else texto
    vals = np.array([float(v) for v in re.split(r'[,;\s]+', txt.strip()) if v])
    if len(vals) != nw:
        raise ValueError(f'--lambdas tem {len(vals)} valores; o eixo espectral tem {nw}')
    med = np.median(np.abs(vals))
    if 800 < med < 1000:
        return vals, '--lambdas (nm)'
    if 8000 < med < 10000:
        return vals / 10.0, '--lambdas (Å)'
    return LAMBDA0_AR_NM + vals * 1e-4, '--lambdas (Δλ em mÅ)'


class CuboSST:
    """Acesso a um cubo SST (t, Stokes, λ, y, x em qualquer ordem)."""

    def __init__(self, caminho, lambdas=None):
        self.caminho = Path(caminho)
        self.hdul, self.lista, self.extras, self.avisos, self.diag = abrir_fits(self.caminho)
        self.alvo = next((h for h in self.hdul if h.is_image and h.header.get('NAXIS', 0) >= 4), None)
        if self.alvo is None:
            raise ValueError('nenhum cubo de imagem com ≥ 4 dimensões')
        self.hdr, self.shape = self.alvo.header, tuple(self.alvo.shape)
        self.eixos, self.fonte_eixos = identificar_eixos(self.hdr, self.shape)
        self.nw = self.shape[self.eixos['w']]
        self.lam_info = comprimentos_de_onda(self.hdul, self.extras, self.hdr,
                                             len(self.shape) - self.eixos['w'], self.nw)
        lam, metodo = self.lam_info['lam_nm'], self.lam_info['metodo']
        if lam is None and lambdas:
            lam, metodo = ler_lambdas(lambdas, self.nw)
        self.metodo_lam = metodo
        self.ordem = np.argsort(lam) if lam is not None else np.arange(self.nw)
        self.lam_nm = np.asarray(lam, dtype=np.float64)[self.ordem] if lam is not None else None
        self.iI = indice_stokes(self.hdr, self.eixos, self.shape, 1)
        self.iV = indice_stokes(self.hdr, self.eixos, self.shape, 4)
        self.cob = cobertura(self.alvo, self.eixos, self.shape)
        self.scans = [int(i) for i in np.flatnonzero(self.cob > 0.5)]
        self.nt = self.shape[self.eixos['t']] if 't' in self.eixos else 1

    def cubo(self, t, stokes='I'):
        """(nw, ny, nx) float32 em λ crescente."""
        s = self.iI if stokes == 'I' else self.iV
        return np.stack([plano(self.alvo, self.eixos, self.shape, t, s, int(k)) for k in self.ordem])

    def media(self, scans, stokes='I'):
        acc = None
        for t in scans:
            c = self.cubo(t, stokes).astype(np.float64)
            acc = c if acc is None else acc + c
        return acc / len(scans)

    def meta(self):
        h = self.hdr
        return {k: (str(h.get(k)) if h.get(k) is not None else None)
                for k in ('INSTRUME', 'TELESCOP', 'DATE-OBS', 'DATE-BEG', 'DATE-END', 'WAVELNTH',
                          'OBSERVER', 'CADAVG', 'BUNIT', 'SOLARNET')}

    def fechar(self):
        self.hdul.close()


# -----------------------------------------------------------------------
def modelo_crisp_continuo():
    """I (quieto) e V por G do modelo com a LSF e a luz espúria do CRISP, na grade LAM."""
    st0 = get_stokes(0.0, v_km_s=V_CORE_KMS)
    st1 = get_stokes(100.0, v_km_s=V_CORE_KMS)
    sig = LSF_CRISP_PM * 1e-3 / DLAM / 2.3548
    I = gaussian_filter1d((1 - STRAY_CRISP) * st0['I'] + STRAY_CRISP, sig)
    V = gaussian_filter1d((1 - STRAY_CRISP) * st1['V'] / 100.0, sig)
    return I, V, float(LAM[np.argmin(st0['I'])])


def extremo(x, y, maximo=True):
    z = -np.asarray(y) if maximo else np.asarray(y)
    k = int(np.nanargmin(z))
    if 0 < k < len(x) - 1:
        x0, x1, x2 = x[k - 1], x[k], x[k + 1]
        y0, y1, y2 = z[k - 1], z[k], z[k + 1]
        num = (x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)
        den = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
        if den < 0:
            return float(x1 - 0.5 * num / den)
    return float(x[k])


def mascaras_regioes(I, V=None):
    """
    I (nw, ny, nx) em λ crescente (qualquer unidade). Devolve dict de máscaras booleanas
    (y, x) e o índice do núcleo. Limiares em LIMIARES (heurísticos, relativos ao quieto).
    """
    valido = np.isfinite(I).all(0) & (I > 0).all(0)
    perfil = np.median(I[:, valido], axis=1)
    k_nuc = int(np.argmin(perfil[1:-1]) + 1)
    asa = 0.5 * (I[0] + I[-1])
    nuc_n = I[k_nuc] / np.median(I[k_nuc][valido])
    asa_n = asa / np.median(asa[valido])
    L = LIMIARES
    quieto = valido & (np.abs(nuc_n - 1) < L['quieto_tol']) & (np.abs(asa_n - 1) < L['quieto_tol'])
    plage = valido & (nuc_n > L['plage_nucleo']) & (asa_n > L['plage_asa'])
    umbra = valido & (asa_n < L['umbra_asa'])
    penumbra = valido & (asa_n >= L['umbra_asa']) & (asa_n < L['penumbra_asa'])
    out = {'valido': valido, 'quieto': quieto, 'plage': plage, 'penumbra': penumbra, 'umbra': umbra}
    if V is not None:
        vamp = np.nanmax(np.abs(V), axis=0) / np.where(asa > 0, asa, np.nan)
        vs = gaussian_filter(np.where(valido, vamp, 0.0), 3) / np.maximum(
            gaussian_filter(valido.astype(float), 3), 1e-6)
        out['calmo'] = quieto & (vs < np.median(vs[quieto]))
    return out, k_nuc


def normalizar_Ic(I, quieto, lam_nm):
    """Ic pelos pontos extremos do Sol quieto contra o modelo; também o núcleo e Δλ (mÅ)."""
    I_mod_c, V_mod_c, lam_nuc_mod = modelo_crisp_continuo()
    perfil_q = np.median(I[:, quieto], axis=1)
    lam_nuc = extremo(lam_nm, perfil_q, maximo=False)
    dl = lam_nm - lam_nuc
    I_mod = np.interp(lam_nuc_mod + dl, LAM, I_mod_c)
    V_mod = np.interp(lam_nuc_mod + dl, LAM, V_mod_c)
    Ic = 0.5 * (perfil_q[0] / I_mod[0] + perfil_q[-1] / I_mod[-1])
    return {'Ic': float(Ic), 'lam_nucleo_nm': float(lam_nuc), 'dl_mA': dl * 1e4,
            'I_modelo': I_mod, 'V_modelo_por_G': V_mod, 'perfil_quieto_Ic': perfil_q / Ic}


def r0_por_varredura(cubo):
    """
    r0 de Fried por varredura, procurando colunas com 'R0' nas tabelas binárias.
    Devolve (array nt ou None, descrição).
    """
    tabs = [h for h in list(cubo.hdul) + list(cubo.extras) if isinstance(h, fits.BinTableHDU)]
    for h in tabs:
        for c in h.columns.names:
            if 'R0' in c.upper():
                try:
                    a = np.asarray(h.data[c][0] if len(h.data) == 1 else h.data[c], dtype=np.float64).ravel()
                except Exception:                                     # noqa: BLE001
                    continue
                if a.size % cubo.nt == 0 and a.size >= cubo.nt:
                    return np.nanmean(a.reshape(cubo.nt, -1), axis=1), f'{h.name}.{c} ({a.size} valores)'
                return None, f'{h.name}.{c}: {a.size} valores, não divisível por {cubo.nt} varreduras'
    return None, 'nenhuma coluna com R0 nas tabelas'


def campo_fraco(I, V, lam_nm, sigma=None, faixa_mA=None, lam_nucleo_nm=None):
    """
    Aproximação de campo fraco: V = −C g λ0² B dI/dλ  (λ em Å, B em G).
    I, V: (nw, ...) em unidades de Ic. sigma: σ por ponto (nw, ...) ou None (pesos iguais).
    faixa_mA: usa só |Δλ| ≤ faixa ao núcleo. Devolve (B, σ_B) com a forma espacial de I[0].
    Na notação do projeto, B_los aqui é B·f·cosθ (o filling entra multiplicando V).
    """
    lam_A = np.asarray(lam_nm, dtype=np.float64) * 10.0
    dI = np.gradient(I.astype(np.float64), lam_A, axis=0)
    U = -C_ZEE_A * G_EFF * (LAMBDA0_AR_NM * 10.0) ** 2 * dI
    usar = np.ones(len(lam_A), bool)
    if faixa_mA is not None:
        ref = lam_nucleo_nm if lam_nucleo_nm is not None else lam_nm[len(lam_nm) // 2]
        usar = np.abs((lam_nm - ref) * 1e4) <= faixa_mA
    U, Vu = U[usar], V[usar].astype(np.float64)
    w = 1.0 if sigma is None else 1.0 / np.maximum(sigma[usar], 1e-12) ** 2
    suu = np.sum(w * U * U, axis=0)
    B = np.sum(w * U * Vu, axis=0) / np.where(suu > 0, suu, np.nan)
    sB = (1.0 / np.sqrt(np.where(suu > 0, suu, np.nan))) if sigma is not None else np.full_like(B, np.nan)
    return B, sB
