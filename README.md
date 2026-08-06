# BULLETIN

BULLETIN is a Python toolbox for operational impact-based forecasting of natural hazards. The current repository supports meteorological hazards, river flooding, and pluvial flooding by combining forecast data, hazard thresholds, exposure information, and precomputed impact or flood-map datasets.

## Workflow overview

| Workflow | Purpose |
| --- | --- |
| `workflows/bulletin_meteo_ibf.py` | Processes one or more meteorological models, classifies forecast hazards, calculates impacts, and optionally merges the model results. |
| `workflows/bulletin_flood_ibf.py` | Runs the FloodPROOFS/HMC workflow: calculates discharge return periods, merges river flood maps, and estimates MUL-based impacts. |
| `workflows/bulletin_glofas_ibf.py` | Processes gridded discharge forecasts such as GLOFAS, compares them with return-period thresholds, and optionally produces administrative hazard and impact outputs. |
| `workflows/bulletin_fanfar_ibf.py` | Processes FANFAR river-section hydrographs, calculates section return periods, merges flood maps, and estimates MUL-based impacts. |
| `workflows/bulletin_pluvial_hazard.py` | Converts forecast rainfall accumulations into precomputed pluvial flood scenarios and administrative impact classes. |
| `workflows/bulletin_hazard_to_impact.py` | Recomputes impacts from an existing hazard product without rerunning the complete hazard workflow. It supports weighted flood overlays and classified meteorological hazards. |

### Supporting and legacy utilities

| Script | Purpose |
| --- | --- |
| `workflows/historical_flood_mapping.py` | Historical or diagnostic FloodPROOFS mapping. It calculates return periods and flood maps but does not run the complete impact workflow. |
| `workflows/bulletin_flood_make_river_network.py` | Static-data utility for creating a river-network shapefile. Its paths and domains are currently hardcoded and must be adapted before use. |

## Repository structure

- `common/`: shared settings, logging, I/O, classification, grid, and time utilities.
- `hydro/`: river-flood return-period, hazard-mapping, and impact modules.
- `meteo/`: meteorological hazard, impact, and multimodel-merging modules.
- `pluvial/`: rainfall input, pluvial hazard, and administrative impact modules.
- `static/`: utilities used to prepare static flood datasets.
- `workflows/`: executable workflow entry points and example settings.

## Installation

The reference environment is defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate env_amhewas
```

The same core Python dependencies are also listed in `requirements.txt`.

## Basic execution

Run workflows from the repository root so that the internal modules can be imported correctly.

```bash
python workflows/bulletin_flood_ibf.py \
    -settings_file workflows/bulletin_flood_ibf_settings.json \
    -time "2026-08-06 00:00" \
    -domain DOMAIN_NAME
```

Most workflows use the same `-settings_file`, `-time`, and optional `-domain` arguments. `bulletin_meteo_ibf.py` also supports model-specific execution and final multimodel merging. Use `--help` on each script for its complete command-line options.

## Configuration files

Example configuration files are currently included for:

- FloodPROOFS/MUL: `workflows/bulletin_flood_ibf_settings.json`
- FANFAR: `workflows/bulletin_fanfar_ibf_settings.json`
- Meteorological IBF: `workflows/bulletin_meteo_ibf_settings.json`
- Pluvial hazard: `workflows/bulletin_pluvial.json`
- Historical FloodPROOFS mapping: `workflows/historical_igad_d2.json`

The GLOFAS and hazard-to-impact workflows require domain-specific settings files, which are not currently included as generic examples.

## Outputs

Depending on the selected workflow and settings, BULLETIN can produce:

- gridded hazard or return-period rasters;
- merged flood maps;
- administrative hazard layers;
- exposed-element and impact shapefiles;
- classified impact levels;
- multimodel meteorological impact products.

All paths, thresholds, variables, exposed elements, return periods, and processing flags are controlled through the workflow settings files.
