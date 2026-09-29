# CAGE: Corruption-Level Estimation with Auxiliary Graph Ensemble

Unsupervised node-level anomaly detection on attributed graphs via joint training of
Autoencoder + Corruption Level Estimator (CLE), with optional auxiliary node
ensemble for diversity.



| Model | Paper | Architecture |
|-------|-------|-------------|
| DOMINANT | SDM 2019 | GCN AE (structure + attribute reconstruction) |
| AnomalyDAE | ICASSP 2020 | Dual AE with graph attention |
| GUIDE | IEEE BigData 2021 | Motif-based dual autoencoder |
| GAD-NR | WSDM 2024 | Neighborhood + feature + degree reconstruction |
| DONE | WSDM 2020 | Outlier-aware dual embedding AE |

## Installation

```bash
pip install torch torch_geometric pygod optuna scikit-learn networkx scipy numpy
```

## Project Structure

```
CAGE-AE_CLE/
├── run.py                  # Main entry point
├── README.md
├── results
└── ae_cle/
    ├── __init__.py          # Package exports
    ├── cli.py               # CLI routing + parameter loading
    ├── augment.py           # Auxiliary node generation
    ├── cle.py               # CLE (Conditional Likelihood Estimation) module
    ├── utils.py             # Metrics, alignment, normalization utilities
    ├── ensemble.py          # Multi-GPU auxiliary node ensemble dispatch
    ├── dominant.py          # DOMINANT model + training
    ├── anomalydae.py        # AnomalyDAE model + training
    ├── guide.py             # GUIDE model + training
    ├── gadnr.py             # GAD-NR model + training
    └── done.py              # DONE model + training
```

## Usage

### Evaluation (run.py)

```bash
# AE-only with tuned parameters
python run.py --dataset weibo --base_model dominant --ae_only --use_best_params --n_runs 10

# Ensemble with tuned parameters + custom config
python run.py --dataset weibo --base_model dominant --auxiliary_node --use_best_params --n_runs 10

# Optimize for AUPRC instead of AUC
python run.py --dataset weibo --base_model dominant --ae_only --use_best_params --metric auprc --n_runs 10
python run.py --dataset weibo --base_model dominant --auxiliary_node --use_best_params --metric auprc --n_runs 10
```

### Key Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset` | `disney` | Dataset name (weibo, reddit, disney, books, enron, etc.) |
| `--base_model` | required | `dominant` / `anomalydae` / `guide` / `gadnr` / `done` |
| `--ae_only` | false | Train AE only (no CLE) |
| `--auxiliary_node` | false | Enable auxiliary node ensemble |
| `--use_best_params` | false | Load optimized params from Optuna results |
| `--metric` | `auc` | `auc` or `auprc` |
| `--n_runs` | 10 | Number of runs for averaging |
| `--seed` | 42 | Base random seed |
| `--no_parallel` | false | Force single-GPU sequential mode |
| `--no_use_embedding_transform` | false | Disable Procrustes alignment |



Results directory structure:
```
results/
├── optuna_results_dominant/
│   ├── best_params_weibo_aeonly.json   # AE-only
│   └── best_params_weibo_auxnode.json  # Ensemble
├── optuna_results_anomalydae/
├── optuna_results_guide/
├── optuna_results_gadnr/
├── optuna_results_done/
└── optuna_results_dominant_auprc/     # AUPRC-tuned
```

## Three-Mode Comparison Paradigm

All 5 base models support three evaluation modes for controlled comparison:

```
AE-only → AE+CLE → AE+CLE + Aux Ensemble
```

## Reproducibility Notes

- Default seeds are deterministic (`torch.manual_seed` + `np.random.seed`).
- Procrustes alignment (`use_embedding_transform`) can be disabled via `--no_use_embedding_transform` for improved stability.

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- PyTorch Geometric
- PyGOD
- Optuna
- scikit-learn
- NetworkX
- NumPy, SciPy
