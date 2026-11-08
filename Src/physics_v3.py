"""
physics_v3.py — motor físico v3 (artigo A): núcleo do Ca II 854,2 nm recalibrado contra o atlas FTS/Neckel e
instrumentos com os parâmetros medidos no SST (scripts 83–106 do artigo A).

Por que existe (script 105/106, 2026-10-26): o motor v2 (Src/physics.py, CALIB) foi calibrado só por I_min e pela
FWHM total do atlas; a forma do núcleo ficou ~40% larga demais (I(±0,33 Å) = 0,26 contra 0,42 do atlas), e o desvio
do filtro casado de campo fraco saía 1,63× o do atlas. Com os parâmetros abaixo (ajuste robusto ao atlas em ±1,5 Å,
script 106): mediana do |resíduo| 0,006 Ic, máximo 0,033 Ic em ±0,4 Å, desvio do filtro 1,015× o do atlas.

Src/physics.py NÃO é alterado: o artigo B (D²NN) e todos os scripts anteriores continuam com o v2 bit a bit.
Este módulo aplica os parâmetros v3 só durante cada chamada (troca temporária de CALIB) e restaura o v2.

Instrumentos v3:
  'IBIS/DST', 'ViSP/DKIST' : iguais ao v2 (ruído não medido);
  'CRISP/SST'  : grade REAL de 2024-10-08 (−650 … +700 mÅ de 8542,09 Å; centro +25 mÅ), LSF 10,5 pm e luz espúria 6%
                 (literatura; o ajuste à série 09:49 dá 13,1 pm e 8%, dentro da tolerância do critério C1),
                 σ_k = ε I_k com ε = 3,26×10⁻³ (script 83, média das duas séries);
  'CRISP2/SST' : grade REAL de 2025-08-19 (−656 … +694 mÅ; centro +19 mÅ), LSF 11,9 pm e luz espúria 0 (ajuste aos
                 perfis observados, script 106), σ_k = ε Ĩ_w (I_k/Ĩ_w)^p com ε = 2,28×10⁻³ (83) e p = 0,71 (93),
                 Ĩ_w = média das duas asas extremas.
"""
from contextlib import contextmanager
import copy

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import physics as P

CHAVES = ('sigma_f', 'gamma_f', 'sigma_c', 'gamma_c', 'weight_c', 'depth')
CALIB_V2_NUCLEO = {k: P.CALIB[k] for k in CHAVES}
CALIB_V3_NUCLEO = {
    'sigma_f': 0.005000083871945171,     # [nm] no limite inferior do ajuste: componente fotosférico ~ Lorentz
    'gamma_f': 0.2407999972571531,
    'sigma_c': 0.013318341566000882,
    'gamma_c': 0.013505257090574487,
    'weight_c': 0.39317096192036033,
    'depth': 0.8194372660481689,
}
V_CORE_KMS_V3 = 4.3174384042749665      # velocidade do núcleo ajustada ao atlas (v2: P.V_CORE_KMS ≈ 4,14 km/s)
LAM = P.LAM

_OFF_CRISP_MA = np.array([-650, -500, -350, -275, -200, -125, -50, 25, 100, 175, 250, 325, 400, 550, 700.])
_OFF_CRISP2_MA = np.array([-656, -506, -356, -281, -206, -131, -56, 19, 94, 169, 244, 319, 394, 544, 694.])

INSTRUMENTAL_V3 = {
    'IBIS/DST': copy.deepcopy(P.INSTRUMENTAL_V2['IBIS/DST']),
    'ViSP/DKIST': copy.deepcopy(P.INSTRUMENTAL_V2['ViSP/DKIST']),
    'CRISP/SST': {
        'modelo': 'v3', 'fwhm_pm': 10.5, 'stray_light': 0.06,
        'grade_dl_pm': (_OFF_CRISP_MA - 25.0) / 10.0,
        'ruido': {'tipo': 'proporcional_I', 'eps': 3.26e-3},
        'cor': '#ffa657',
    },
    'CRISP2/SST': {
        'modelo': 'v3', 'fwhm_pm': 11.9, 'stray_light': 0.0,
        'grade_dl_pm': (_OFF_CRISP2_MA - 19.0) / 10.0,
        'ruido': {'tipo': 'potencia_I', 'eps': 2.28e-3, 'p': 0.71},
        'cor': '#d2a8ff',
    },
}


@contextmanager
def _nucleo_v3():
    antigo = {k: P.CALIB[k] for k in CHAVES}
    try:
        P.CALIB.update(CALIB_V3_NUCLEO)
        yield
    finally:
        P.CALIB.update(antigo)


def get_stokes(B, theta_deg=0.0, chi_deg=0.0, filling=1.0, v_km_s=None, lam_grid=None,
               preench_nucleo=0.0, contraste_cont=1.0):
    """get_stokes do v2 com o núcleo v3. v_km_s = None → velocidade do núcleo v3 (V_CORE_KMS_V3)."""
    if v_km_s is None:
        v_km_s = V_CORE_KMS_V3
    with _nucleo_v3():
        return P.get_stokes(B, theta_deg=theta_deg, chi_deg=chi_deg, filling=filling, v_km_s=v_km_s,
                            lam_grid=lam_grid, preench_nucleo=preench_nucleo, contraste_cont=contraste_cont)


def sigma_por_ponto(instrumento, I_limpo):
    r = instrumento['ruido']
    I = np.abs(np.asarray(I_limpo, dtype=np.float64))
    if r['tipo'] == 'potencia_I':
        Iw = (I[0] + I[-1]) / 2.0
        return float(r['eps']) * Iw * (I / Iw) ** float(r['p'])
    return P._sigma_por_ponto_v2(instrumento, I_limpo)


def degradar_stokes(stokes_dict, lam_grid_nm, instrumento, rng_local=None, fator_ruido=1.0):
    """Mesma cadeia do v2 (luz espúria → LSF → amostragem → ruído), com o ruído 'potencia_I' do CRISP2."""
    if instrumento.get('modelo') == 'legado':
        return P.degradar_stokes(stokes_dict, lam_grid_nm, instrumento, rng_local, fator_ruido)
    if rng_local is None:
        rng_local = np.random.default_rng()
    alpha = instrumento.get('stray_light', 0.0)
    dlam = lam_grid_nm[1] - lam_grid_nm[0]
    sig_px = instrumento['fwhm_pm'] * 1e-3 / dlam / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    lam_obs = P._lam_obs_v2(instrumento, lam_grid_nm)

    def amostrar(p):
        return np.interp(lam_obs, lam_grid_nm, gaussian_filter1d(p, sigma=sig_px))

    I_limpo = amostrar((1.0 - alpha) * stokes_dict['I'].astype(np.float64) + alpha)
    sig = sigma_por_ponto(instrumento, I_limpo)
    out = {'I': (I_limpo + rng_local.normal(0.0, 1.0, len(lam_obs)) * sig * fator_ruido).astype(np.float32)}
    for key in ('V', 'Q', 'U'):
        limpo = amostrar(stokes_dict[key].astype(np.float64) * (1.0 - alpha))
        out[key] = (limpo + rng_local.normal(0.0, 1.0, len(lam_obs)) * sig * fator_ruido).astype(np.float32)
    out['lam_obs'] = lam_obs
    out['sigma_por_ponto'] = sig * fator_ruido
    out['sigma_tot'] = float(instrumento['ruido'].get('eps', instrumento['ruido'].get('sigma', 0.0))) * fator_ruido
    out['alpha'] = alpha
    return out


def validar_contra_atlas(caminho_atlas):
    """Métricas do v3 (e do v2) contra o atlas em ±1,5 Å do núcleo: mediana e máximo de |I − atlas|."""
    a = np.loadtxt(caminho_atlas)
    At = np.interp(LAM, a[:, 0], a[:, 1])
    c = LAM[np.argmin(At)]
    m15, m04 = np.abs(LAM - c) <= 0.15, np.abs(LAM - c) <= 0.04
    I3 = get_stokes(0.0)['I']
    I2 = P.get_stokes(0.0, v_km_s=P.V_CORE_KMS)['I']
    d = lambda I: {'mediana_1p5A': float(np.median(np.abs(I - At)[m15])), 'max_0p4A': float(np.max(np.abs(I - At)[m04]))}
    return {'v3': d(I3), 'v2': d(I2)}


# Travas: o v2 do módulo original fica intacto, e o v3 tem V = 0 em B = 0 e V monotônico em B.
assert all(P.CALIB[k] == CALIB_V2_NUCLEO[k] for k in CHAVES)
assert np.allclose(get_stokes(0.0)['V'], 0.0, atol=1e-12)
_amp = [np.ptp(get_stokes(b)['V']) for b in (100.0, 500.0, 1000.0)]
assert _amp[0] < _amp[1] < _amp[2]
assert all(P.CALIB[k] == CALIB_V2_NUCLEO[k] for k in CHAVES)
