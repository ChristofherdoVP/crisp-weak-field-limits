# Weak-field detection limits of Ca II 854.2 nm with CRISP and CRISP2

Code and reports for the paper

> *Weak-field detection limits of Ca II 854.2 nm with CRISP and CRISP2: a synthetic model validated against SST observations*, by C. do Vale Pena, F. L. Guarnieri and L. E. A. Vieira.

The code builds a synthetic model of Ca II 854.2 nm weak-field Stokes V profiles, degraded by instrument models of CRISP and CRISP2 (Swedish 1-m Solar Telescope), and computes the detection limit a₅₀ of a weighted weak-field matched filter at a 5% false-alarm probability. It then tests the model against CRISP (2024-10-08) and CRISP2 (2025-08-19) observations: noise, I→V crosstalk, map noise, temporal averaging, the region where the weak-field maps are quantitative, and measured versus predicted detection limits.

The synthetic model and the estimator were introduced in a companion paper (do Vale Pena et al. 2026; code at https://github.com/ChristofherdoVP/d2nn-matched-filter-limits). `Src/physics.py` is that model, unchanged; `Src/physics_v3.py` replaces only the line core, fitted to the FTS atlas.

## Requirements

Python ≥ 3.10 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

All scripts run on a CPU. `torch` is needed only because some shared modules in `Src/` import it.

## Data

The observations are not included. They are public in the SST data archive (https://dubshen.astro.su.se/sst_archive/). Place the four Ca II 854.2 nm cubes in `Data/RAW_SST/`:

| Data set | File |
|---|---|
| CRISP, 2024-10-08, 14 scans | `nb_8542_2024-10-08T09_49_51_09_49_51=0-13_stokes_corrected_export2026-01-26T18_19_45_im.fits` |
| CRISP, 2024-10-08, 10 scans | `nb_8542_2024-10-08T10_10_50_10_10_50=0-9_stokes_corrected_export2026-01-26T14_51_16_im.fits` |
| CRISP2, 2025-08-19, 2×2 mosaic (08:01 UT) | `nb_8542_2025-08-19T08_01_31_mosaic_0--3_stokes_corrected_export2026-05-14T12_14_55_im.fits` |
| CRISP2, 2025-08-19, 2×2 mosaic (10:52 UT) | `nb_8542_2025-08-19T10_48_36_mosaic_0--3_stokes_corrected_export2026-05-14T12_15_56_im.fits` |

The second CRISP2 mosaic starts at 10:52:54 UT; "10_48" is the time in the file name only. The cubes total about 36 GB.

The reference atlas `Data/Reference/atlas_fts_neckel_hs5000_8542.txt` is the FTS disk-centre atlas (Neckel 1999) in the version distributed with HelioSpectrotron 5000 (Pietrow 2026), cut to 853.6–854.8 nm.

## Repository layout

| Folder | Contents |
|---|---|
| `Src/` | synthetic line model (`physics.py`, `physics_v3.py`), SST cube reading (`sst_io.py`), weak-field estimator (`sst_analise.py`) and shared utilities |
| `Scripts/` | numbered scripts 78–111; each states its test and decision criterion in its header (in Portuguese) and writes a JSON report |
| `Outputs/Reports/` | JSON reports from which the numbers, tables and figures of the paper are drawn |
| `Outputs/Figures/ArtigoA/` | the figures of the paper (PDF) |

## Reproducing the paper

Scripts are run from the repository root. Scripts that read the cubes take the FITS files as arguments (see the usage line in each header). Maps and masks (`Outputs/Data/*.npy`) are written by the scripts and are not versioned.

| Paper item | Scripts |
|---|---|
| Sect. 2, observations (headers, geometry, plate scale) | `91_sst_inspecionar_cabecalho.py`, `92_sst_sobreposicao_ladrilhos.py`, `111_sst_escala_crisp.py` |
| Sect. 3, synthetic model and limits (Fig. 1, Fig. 2) | `105_sst_perfil_observado_modelo.py`, `106_recalibracao_motor_atlas.py`, `107_limites_sinteticos_v3.py` |
| Sect. 4, noise of CRISP2 (Fig. 3) | `78`–`83`, `93_sst_ruido_por_lambda.py` |
| Sect. 4, noise of CRISP and of the maps (Figs. 3, 4) | `96_sst_ruido_series_temporais.py`, `100_sst_mapas_blos_series.py` |
| Sect. 4, temporal averaging and binning (Fig. 5) | `101_sst_media_temporal.py` |
| Sect. 5, I→V crosstalk and its bias (Fig. 6) | `84`–`88`, `98_sst_diagnostico_simetrica.py`, `99_sst_vies_blos_series.py`; whitened filter: `94_sst_filtro_branqueado.py` |
| Sect. 6, maps and reliability mask (Figs. 7, 8) | `89_sst_mapas_blos.py`, `90_sst_reprodutibilidade_blos.py`, `110_sst_limite_campo_fraco_v3.py` (`102` is the earlier version) |
| Sect. 7, false alarm and measured versus predicted limits (Fig. 9, Table 2) | `103_sst_falso_alarme.py`, `104_sst_limites_previsto_medido.py` |
| Appendix A, estimator calibration (Fig. 10) | `108_regressao_bf_v3.py` |
| Appendix B, tile overlaps | `92_sst_sobreposicao_ladrilhos.py`, `93_sst_ruido_por_lambda.py` |
| All figures and Table 2 | `109_figuras_artigo_a.py` (reads only the reports and maps) |

Suggested order: 91 → 78–90, 92–94 (CRISP2) → 96, 98–101 (CRISP) → 102–106 → 107, 108, 110, 111 → 109.

## Citation

If you use this code, please cite the paper and the archived version of this repository (see `CITATION.cff` and the Zenodo DOI).

## License

MIT (see `LICENSE`).
