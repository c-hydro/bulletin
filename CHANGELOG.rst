=========
Changelog
=========

Version 5.0.0 [2026-08-06]
************************
Repository consolidation
    - Consolidated the current meteorological, FloodPROOFS, GLOFAS, FANFAR,
      pluvial, and hazard-to-impact workflows in a single package.
    - Added ``environment.yml`` and aligned the documented dependency set with
      ``requirements.txt``.
    - Updated ``README.md`` to describe the current repository structure,
      workflow purposes, configuration files, and basic execution.
    - Restored ``.gitignore`` and excluded Python cache files from distributions.

APP: **workflows/bulletin_flood_ibf.py**
    - Restored the FloodPROOFS/HMC with MUL impact workflow after it had been
      overwritten by the gridded GLOFAS workflow during the merge.
    - Kept the original linear workflow structure while adding clearer logging,
      algorithm-time validation, administrative-layer checks, and final error
      reporting.

APP: **workflows/bulletin_glofas_ibf.py**
    - Preserved the gridded-discharge/GLOFAS workflow as a separate entry point.
    - Added input-file validation, threshold-based alert levels, optional
      administrative hazard assessment, flood-map overlays, and impact outputs.

APP: **workflows/bulletin_meteo_ibf.py**
    - Extended multimodel execution, model-specific reruns, missing-model
      handling, output-completeness checks, and final multimodel merging.

APP: **workflows/bulletin_hazard_to_impact.py**
    - Added a standalone post-processing workflow for weighted flood overlays
      and preclassified meteorological hazard products.

APP: **workflows/bulletin_pluvial_hazard.py**
    - Added pluvial scenario selection from rainfall accumulations and
      administrative impact classification.

APP: **workflows/bulletin_fanfar_ibf.py**
    - Updated workflow-level error reporting while retaining the FANFAR
      section-based return-period and MUL-impact approach.

Version 4.0.0 [2024-08-20]
**************************
APP: **bulletin_hydro_fp-evr.py**
    - Implemented a new approach for IBF with FloodPROOFS, based on pre-computed impacts at the MUL level

APP: **bulletin_hydro_glofas.py**
    - Update to new Copernicus API

Version 3.0.0 [2024-05-17]
**************************
APP: **bulletin_hydro_glofas.py**
    - Update to GLOFAS 4.0

APP: **bulletin_hydro_glofas_v3.py**
    - Renamed for retrocompatibility

Version 2.0.2 [2024-01-09]
**************************
APP: **bulletin_drought_cdi.py**
    - Updated for the operational version

Version 2.0.1 [2023-09-01]
**************************
APP: **bulletin_meteo_merger.py**
    - App to merge results from IBFs with different meteorological models

Version 2.0.0 [2023-07-04]
**************************
APP: **bulletin_meteo_multimodel.py**
    - Improved adaptability to different meteorological models
    - Added possibility of include more exposed elements
    - Improved adaptability to several meteorological hazards

Version 1.4.0 [2023-06-30]
**************************
APP: **bulletin_drought_cdi.py**
    - Setup of IBF for drought based on CDI (prototype)

Version 1.3.0 [2022-11-22]
**************************
APP: **bulletin_hydro_fp.py**
    - Multi-impact assessment for FloodPROOFS

Version 1.2.0 [2022-03-24]
**************************
APP: **bulletin_multihazard_meteo_gfs.py**
    - Added impact assessment

APP: **bulletin_hydro_glofas.py**
    - Added impact assessment

Version 1.1.0 [2021-11-11]
**************************
APP: **bulletin_multihazard_meteo_gfs.py**
    - Added backup procedure for using local forecast file

- Separated hydro warning components

APP: **bulletin_hydro_glofas.py**
    - Release with GLOFAS support though CDS api

APP: **bulletin_hydro_glofas.py**
    - Release with FloodProofs support

Version 1.0.0 [2020-03-26]
**************************
APP: **bulletin_multihazard_meteo_gfs.py**
    - Starting version in experimental mode for FloodProofs Africa
