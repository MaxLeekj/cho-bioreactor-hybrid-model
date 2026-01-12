# CHO Hybrid Stoichiometric Neural Network (Thesis Code)

This repository contains the code used in my thesis on hybrid modelling of Chinese hamster ovary (CHO) cell bioreactor dynamics. The workflow combines:
- a mechanistic data-generation model,
- a reduced stoichiometric representation,
- neural networks for time-conditioned prediction of viable cell density and pseudo-reaction rates,
- reconstruction of concentration trajectories and diagnostic visualisations.

The scripts and notebooks here reproduce the training runs and figures reported in the thesis.

## Repository structure
- `data/`
  - (optional) saved datasets, model-ready arrays, scalers
- `notebooks/`
  - exploratory analysis, quick experiments, figure reproduction
- `src/`
  - core functions (simulation, preprocessing, training, evaluation)
- `scripts/`
  - runnable pipelines (generate data, train models, plot results)
- `models/`
  - saved model checkpoints / weights
- `figures/`
  - exported plots used in the thesis
- `reports/`
  - metric tables, logs, and run summaries

## Setup
### Requirements
- Python 3.10+ recommended
- Common packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow` 

### Install
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
