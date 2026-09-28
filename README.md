# Data-driven design of anticorrosive coating formulations

This repository contains the analysis and document-build code used for independent epoxy-primer and polyurethane/fluororesin-topcoat formulation studies. The release is intentionally code-only: raw workbooks, derived datasets, trained models, figures, manuscript text, rendered documents, logs, and literature files are not included.

## Scope

- Primer targets: salt-spray lifetime and pull-off adhesion, with wear loss as an auxiliary response.
- Topcoat targets: salt-spray lifetime and wear mass loss.
- Primer and topcoat records are modeled as independent material domains; no paired primer-topcoat system model is constructed.
- The workflow covers data audit, frozen holdout creation, nested validation, structured holdout, uncertainty analysis, negative controls, Pareto screening, material-centered visualization, and optional manuscript/SI construction.

## Repository layout

```text
config/                       Example project configuration
src/                          Analysis, validation, literature, and plotting code
slurm/                        Slurm entry points for long-running stages
manuscript/build_manuscript.py
supplementary/build_supplementary.py
requirements.txt              Tested dependency snapshot
.env.example                  Environment-variable template
```

Generated directories such as `data/`, `results/`, `figure/`, and `logs/` are ignored by Git.

## Installation

Python 3.11 was used for the original workflow.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned versions record the tested environment. Platform-specific builds of older packages, especially XGBoost, may require a compatible package channel or a locally managed environment.

## External inputs

No research data are stored in this repository. Set the following variables before running the analysis:

```bash
export COATING_PROJECT_ROOT="$(pwd)"
export COATING_DATA_DIR=/path/to/private/coating_workbooks
export COATING_PRIMER_FILE=primer.xlsx
export COATING_TOPCOAT_FILE=topcoat.xlsx
export COATING_SHEET_NAME=coating_samples
```

The workbooks are expected to use the project field names referenced by the scripts, including:

- responses: `salt_spray_pass_h`, `adhesion_mpa_or_grade`, `wear_mass_loss_mg`;
- primer variables: strontium chromate, zinc phosphate, curing-agent type, curing temperature/time, and surface roughness;
- topcoat variables: polyurethane A14, fluororesin, SiC, BN, lubricant powder, dispersant, anti-settling agent, and defoamer.

The code reads the workbooks without modifying them. A structural example is provided in `config/project_config.example.json`; it does not contain observations.

## Analysis workflow

Create ignored output directories first:

```bash
mkdir -p data results figure supplementary logs
```

Run the main stages from the repository root:

```bash
python src/01_data_audit.py
python src/02_design_and_holdout.py
python src/03_strict_modeling.py
python src/07_uncertainty_stability.py
python src/08_pareto_figures.py
python src/09_framework_supplementary.py
python src/14_material_coordinates.py
```

Long-running stages can instead be submitted through Slurm:

```bash
mkdir -p logs
sbatch slurm/03_strict_modeling.sh
sbatch slurm/07_uncertainty_stability.sh
```

Optional Slurm/runtime variables are documented in `.env.example`. The scripts do not contain cluster usernames or absolute home-directory paths.

## Optional literature workflow

Literature PDFs and curated evidence records are external inputs and are not included.

```bash
export COATING_LITERATURE_DIR=/path/to/private/literature_pdfs
bash src/04_extract_literature.sh
python src/05_literature_snippets.py

export COATING_LITERATURE_MATRIX_INPUT=/path/to/private/literature_matrix.csv
python src/06_literature_matrix.py
python src/10_verify_references.py
```

`src/06_literature_matrix.py` validates a supplied CSV rather than embedding literature records in source code. Optional DOI additions and metadata overrides can be supplied through `COATING_EXTRA_DOIS` and `COATING_REFERENCE_OVERRIDES`.

## Optional document builders

The Word builders are included as code, but the manuscript/SI source text, figures, tables, and template are intentionally excluded. To use them, supply the required external artifacts in the paths expected by the scripts and optionally set:

```bash
export COATING_TEMPLATE_DOCX=/path/to/private/template.docx
python src/11_distill_template.py
python manuscript/build_manuscript.py
python supplementary/build_supplementary.py
```

The validation scripts require the complete generated analysis and document artifacts:

```bash
python src/13_validate_project.py
python src/15_validate_supplementary.py
```

## Reproducibility controls

- frozen holdouts are created before model and representation selection;
- preprocessing and hyperparameter search remain inside outer training folds;
- random seeds are fixed in the analysis scripts;
- label permutation, random-noise features, bootstrap prediction intervals, conformal intervals, and composition-representation sensitivity are evaluated separately;
- raw workbooks remain outside version control.

## License

This code is released under the MIT License provided in the repository.
