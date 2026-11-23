# CAGE: Corruption-Level Estimation with Auxiliary Graph Ensemble for Autoencoder-Based Graph Anomaly Detection

CAGE detects anomalous nodes in attributed graphs. It constructs multiple views of the input graph by adding synthetic auxiliary nodes and trains a separate autoencoder (AE) and corruption-level estimator (CLE) on each view. The CLE learns to predict the level of noise applied to a node's latent representation. At inference, each view combines the AE reconstruction score with the CLE score evaluated on the clean representation. CAGE aggregates these scores across views for each original node.

The AE learns from its reconstruction loss. The CLE learns from detached AE representations, so its loss updates the CLE parameters only. Both training objectives use unlabeled nodes. 
## Supported backbones

| Model | Paper | Architecture |
|-------|-------|-------------|
| DOMINANT | SDM 2019 | GCN AE (structure + attribute reconstruction) |
| AnomalyDAE | ICASSP 2020 | Dual AE with graph attention |
| GUIDE | IEEE BigData 2021 | Motif-based dual autoencoder |
| GAD-NR | WSDM 2024 | Neighborhood + feature + degree reconstruction |
| DONE | WSDM 2020 | Outlier-aware dual embedding AE |

## Installation

The following versions were recorded in the project's experiment environments. The GitHub reproduction used the PyTorch CUDA build shown below.

| Component | Recorded version |
| --- | --- |
| Python | `3.12.3` |
| PyTorch | `2.3.0+cu121` (CUDA 12.1 build) |
| PyTorch Geometric (`torch-geometric`) | `2.6.1` |
| PyGOD | `1.1.0` |
| NumPy | `1.26.4` |

For Linux with an NVIDIA GPU and a compatible driver, run:

```bash
conda create -n cage python=3.12.3 -y
conda activate cage
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-geometric==2.6.1 pygod==1.1.0 numpy==1.26.4 scipy scikit-learn networkx
```

SciPy, scikit-learn and NetworkX are listed without pins because their exact versions were not recorded. Evaluation loads the saved Optuna configurations directly. For other PyTorch builds, see the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v230).

## Run CAGE

Run the following command **from the repository root** (the directory containing `run.py` and `results/`). This evaluates the full auxiliary-view CAGE ensemble on Weibo with the saved DOMINANT configuration and seeds 42–51.

```bash
python run.py --dataset weibo --base_model dominant --auxiliary_node --use_best_params --seed 42 --n_runs 10
```

Check that the terminal prints `Loaded best parameters for dataset 'weibo' from results/optuna_results_dominant/best_params_weibo_auxnode.json`, followed by the effective parameters. If the file is missing or unreadable, the CLI falls back to its defaults. Datasets are loaded through PyGOD. The five evaluation datasets are `weibo`, `reddit`, `disney`, `books` and `enron`. The backbone identifiers are `dominant`, `anomalydae`, `guide`, `gadnr` and `done`.

By default, `--use_best_params` loads the configuration selected by AUC-ROC. To load the configuration previously selected by AUPRC, use:

```bash
python run.py --dataset weibo --base_model dominant --auxiliary_node --use_best_params --metric auprc --seed 42 --n_runs 10
```

Both commands report AUC-ROC and AUPRC. The `--metric` flag selects the parameter directory.

### Original-graph ablations

```bash
# AE only, with the saved AE-only configuration
python run.py --dataset weibo --base_model dominant --ae_only --use_best_params --seed 42 --n_runs 10

# Single-view AE+CLE, with the CLI defaults
python run.py --dataset weibo --base_model dominant --seed 42 --n_runs 10
```

## Reproducibility

- **Runs and timing.** Each run uses `seed + run_index`, starting at index 0. With `--seed 42 --n_runs 10`, the summary uses all ten runs for the mean, population standard deviation (`ddof=0`) and best value of each metric. Average runtime includes the first run. GPU results can vary across executions.
- **Latent corruption.** The paper's default CAGE configuration uses standard Gaussian noise and disables both the adaptive prior and the embedding transform. Loading a JSON file applies its saved settings, including any `use_embedding_transform` value, after the CLI defaults. Check the printed `use_embedding_transform` and `use_adaptive_prior` fields when comparing configurations.

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
    ├── cle.py               # Corruption-level estimator and latent perturbation
    ├── utils.py             # Metrics, alignment, normalization utilities
    ├── ensemble.py          # Multi-GPU auxiliary node ensemble dispatch
    ├── dominant.py          # DOMINANT model + training
    ├── anomalydae.py        # AnomalyDAE model + training
    ├── guide.py             # GUIDE model + training
    ├── gadnr.py             # GAD-NR model + training
    └── done.py              # DONE model + training
```
