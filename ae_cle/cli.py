"""CLI entry point and parameter loading for AE+CLE."""

import argparse
import json
import os
import torch
import numpy as np
import random
import time
from datetime import datetime, timedelta
from pygod.utils import load_data

from .dominant import train_joint_ae_cle
from .anomalydae import train_joint_anomalydae_cle
from .guide import train_joint_guide_cle
from .gadnr import train_joint_gadnr_cle
from .done import train_joint_done_cle
from .ensemble import (train_auxiliary_node_ensemble,
    train_auxiliary_node_ensemble_anomalydae,
    train_auxiliary_node_ensemble_guide,
    train_auxiliary_node_ensemble_gadnr,
    train_auxiliary_node_ensemble_done)
from .utils import format_timedelta_precise, compute_all_metrics


def load_best_params(dataset_name, params_dir='results/optuna_results_dominant', mode_suffix=''):
    """
    Load best parameters for a given dataset if available.
    mode_suffix: '' (standard joint), '_auxnode', or '_aeonly'
    Returns best_params dict if found, None otherwise
    """
    params_file = os.path.join(params_dir, f'best_params_{dataset_name}{mode_suffix}.json')

    if os.path.exists(params_file):
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Loaded best parameters for dataset '{dataset_name}' from {params_file}")
            print(f"Best AUC from tuning: {results['best_value']:.6f}")
            return results['best_params']
        except Exception as e:
            print(f"Warning: Could not load best parameters from {params_file}: {e}")
            print("Using default parameters instead.")

    return None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Train AE+CLE joint model (Unsupervised)')
    parser.add_argument('--dataset', type=str, default='disney', help='Dataset name')
    parser.add_argument('--base_model', type=str, required=True, choices=['dominant', 'anomalydae', 'guide', 'gadnr', 'done'],
                        help='Base AE model: dominant, anomalydae, guide, gadnr, or done')
    parser.add_argument('--params_dir', type=str, default=None, help='Directory containing best parameters')
    parser.add_argument('--use_best_params', action='store_true', help='Use best parameters from optuna tuning')
    parser.add_argument('--metric', type=str, default='auc', choices=['auc', 'auprc'],
                        help='Metric to load best_params for (default: auc)')
    parser.add_argument('--joint_training', action='store_true', default=True, help='Perform joint AE+CLE training (default: True)')
    parser.add_argument('--ae_only', action='store_true', help='Train AE only (equivalent to setting joint_training=False)')
    parser.add_argument('--auxiliary_node', action='store_true',
                        help='Use auxiliary node ensemble (add synthetic anomalous nodes for diversity)')
    parser.add_argument('--n_aux_models', type=int, default=10,
                        help='Number of models in auxiliary node ensemble (default: 10)')
    parser.add_argument('--n_aux_nodes', type=float, default=5,
                        help='Aux nodes per model. If < 1 treated as ratio (e.g. 0.05=5%%), else absolute count (default: 5)')
    parser.add_argument('--aux_feature_method', type=str, default='outlier_tail',
                        choices=['outlier_tail', 'gaussian_noise', 'perturb_existing',
                                 'smote_outlier', 'neighbor_dissimilar', 'feature_shuffle'],
                        help='Method for auxiliary node features (default: outlier_tail)')
    parser.add_argument('--aux_edge_method', type=str, default='random_connect',
                        choices=['random_connect', 'isolated', 'clique', 'low_similarity_connect'],
                        help='Method for connecting auxiliary nodes (default: random_connect)')
    parser.add_argument('--aux_n_connections', type=int, default=5,
                        help='Number of real nodes each aux node connects to (default: 5)')
    parser.add_argument('--aux_k_std', type=float, default=3.0,
                        help='Std multiplier for outlier_tail features (default: 3.0)')
    parser.add_argument('--no_parallel', action='store_true', default=False,
                        help='Force sequential execution even if multiple GPUs available')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs for averaging results (ignored in --bagging mode)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_use_embedding_transform', dest='use_embedding_transform', action='store_false', default=False,
                        help='Disable embedding transform (center+normalize+Procrustes+sign_fix)')
    parser.add_argument('--adaptive_prior', dest='use_adaptive_prior', action='store_true', default=False,
                        help='Use adaptive covariance prior instead of standard Gaussian noise in CLE')

    args = parser.parse_args()

    # Set default params_dir based on base_model and metric
    _metric_dir_suffix = '_auprc' if args.metric == 'auprc' else ''
    if args.params_dir is None:
        if args.base_model == 'anomalydae':
            args.params_dir = 'results/optuna_results_anomalydae' + _metric_dir_suffix
        elif args.base_model == 'guide':
            args.params_dir = 'results/optuna_results_guide' + _metric_dir_suffix
        elif args.base_model == 'gadnr':
            args.params_dir = 'results/optuna_results_gadnr' + _metric_dir_suffix
        elif args.base_model == 'done':
            args.params_dir = 'results/optuna_results_done' + _metric_dir_suffix
        else:
            args.params_dir = 'results/optuna_results_dominant' + _metric_dir_suffix

    # Handle AE only mode
    if args.ae_only:
        args.joint_training = False

    # Determine mode suffix for loading the correct best_params file
    if args.base_model == 'anomalydae':
        base_tag = '_anomalydae'
    elif args.base_model == 'guide':
        base_tag = '_guide'
    elif args.base_model == 'gadnr':
        base_tag = '_gadnr'
    elif args.base_model == 'done':
        base_tag = '_done'
    else:
        base_tag = ''
    if args.auxiliary_node and args.ae_only:
        mode_suffix = base_tag + '_auxnode_aeonly'
    elif args.auxiliary_node:
        mode_suffix = base_tag + '_auxnode'
    elif args.ae_only:
        mode_suffix = base_tag + '_aeonly'
    else:
        mode_suffix = base_tag

    # Load data
    print(f"Loading dataset: {args.dataset}")
    data = load_data(args.dataset)

    # ==================== Auxiliary Node Ensemble Mode ====================
    if args.auxiliary_node:
        aux_joint = not args.ae_only
        mode_str = "AE+CLE" if aux_joint else "AE only"
        print("\n" + "=" * 60)
        print("AUXILIARY NODE ENSEMBLE MODE ({})".format(mode_str))
        print("=" * 60)

        if args.base_model == 'anomalydae':
            aux_params = {
                'n_aux_models': args.n_aux_models,
                'n_aux_nodes': args.n_aux_nodes,
                'aux_feature_method': args.aux_feature_method,
                'aux_edge_method': args.aux_edge_method,
                'aux_n_connections': args.aux_n_connections,
                'aux_k_std': args.aux_k_std,
                'epochs': 100,
                'ae_hidden': 64,
                'ae_dropout': 0.3,
                'cle_hidden': [256, 512, 256],
                'lamda1': 0.5,
                'lamda2': 0.5,
                'normalize_loss': True,
                'normalize_method': 'exponential_moving_average',
                'normalize_scores': True,
                'score_norm_method': 'min_max',
                'use_adaptive_prior': args.use_adaptive_prior,
                'use_embedding_transform': args.use_embedding_transform,
                'agg_method': 'mean',
                'lr_ae': 5e-3,
            }
        elif args.base_model == 'guide':
            aux_params = {
                'n_aux_models': args.n_aux_models,
                'n_aux_nodes': args.n_aux_nodes,
                'aux_feature_method': args.aux_feature_method,
                'aux_edge_method': args.aux_edge_method,
                'aux_n_connections': args.aux_n_connections,
                'aux_k_std': args.aux_k_std,
                'epochs': 100,
                'guide_hidden_a': 64,
                'guide_hidden_s': 4,
                'guide_num_layers': 4,
                'guide_dropout': 0.0,
                'guide_alpha': 0.5,
                'cle_hidden': [256, 512, 256],
                'lamda1': 0.5,
                'lamda2': 0.5,
                'normalize_loss': True,
                'normalize_method': 'exponential_moving_average',
                'normalize_scores': True,
                'score_norm_method': 'min_max',
                'use_adaptive_prior': args.use_adaptive_prior,
                'use_embedding_transform': args.use_embedding_transform,
                'agg_method': 'mean',
                'lr_ae': 5e-3,
            }
        elif args.base_model == 'gadnr':
            aux_params = {
                'n_aux_models': args.n_aux_models,
                'n_aux_nodes': args.n_aux_nodes,
                'aux_feature_method': args.aux_feature_method,
                'aux_edge_method': args.aux_edge_method,
                'aux_n_connections': args.aux_n_connections,
                'aux_k_std': args.aux_k_std,
                'epochs': 100,
                'gadnr_hidden': 64,
                'sample_size': 10,
                'encoder': 'GCN',
                'cle_hidden': [256, 512, 256],
                'lamda1': 0.5,
                'lamda2': 0.5,
                'normalize_loss': True,
                'normalize_method': 'exponential_moving_average',
                'normalize_scores': True,
                'score_norm_method': 'min_max',
                'use_adaptive_prior': args.use_adaptive_prior,
                'use_embedding_transform': args.use_embedding_transform,
                'agg_method': 'mean',
                'lr_ae': 5e-3,
            }
        elif args.base_model == 'done':
            aux_params = {
                'n_aux_models': args.n_aux_models,
                'n_aux_nodes': args.n_aux_nodes,
                'aux_feature_method': args.aux_feature_method,
                'aux_edge_method': args.aux_edge_method,
                'aux_n_connections': args.aux_n_connections,
                'aux_k_std': args.aux_k_std,
                'epochs': 100,
                'done_hidden': 64,
                'done_num_layers': 4,
                'done_dropout': 0.0,
                'cle_hidden': [256, 512, 256],
                'lamda1': 0.5,
                'lamda2': 0.5,
                'normalize_loss': True,
                'normalize_method': 'exponential_moving_average',
                'normalize_scores': True,
                'score_norm_method': 'min_max',
                'use_adaptive_prior': args.use_adaptive_prior,
                'use_embedding_transform': args.use_embedding_transform,
                'agg_method': 'mean',
                'lr_ae': 5e-3,
            }
        else:
            aux_params = {
                'n_aux_models': args.n_aux_models,
                'n_aux_nodes': args.n_aux_nodes,
                'aux_feature_method': args.aux_feature_method,
                'aux_edge_method': args.aux_edge_method,
                'aux_n_connections': args.aux_n_connections,
                'aux_k_std': args.aux_k_std,
                'epochs': 100,
                'ae_hidden': 64,
                'cle_hidden': [256, 512, 256],
                'lamda1': 0.5,
                'lamda2': 0.5,
                'normalize_loss': True,
                'normalize_method': 'exponential_moving_average',
                'normalize_scores': True,
                'score_norm_method': 'min_max',
                'use_adaptive_prior': args.use_adaptive_prior,
                'use_embedding_transform': args.use_embedding_transform,
                'agg_method': 'mean',
                'dropout': 0.3,
                'lr_ae': 5e-3,
                'struct_weight': 0.8,
            }

        if args.use_best_params:
            best_params = load_best_params(args.dataset, args.params_dir, mode_suffix)
            if best_params:
                if 'cle_hidden1' in best_params:
                    best_params['cle_hidden'] = [
                        best_params.pop('cle_hidden1'),
                        best_params.pop('cle_hidden2'),
                        best_params.pop('cle_hidden3'),
                    ]
                _key_map = {
                    'feature_method': 'aux_feature_method',
                    'edge_method': 'aux_edge_method',
                    'n_connections': 'aux_n_connections',
                    'k_std': 'aux_k_std',
                }
                for _old, _new in _key_map.items():
                    if _old in best_params:
                        best_params[_new] = best_params.pop(_old)
                if best_params.pop('aux_use_ratio', False) and 'aux_node_ratio' in best_params:
                    best_params['n_aux_nodes'] = best_params.pop('aux_node_ratio')
                aux_params.update(best_params)
                # Remap generic 'ae_hidden' to model-specific param name
                _hidden_map = {'guide': 'guide_hidden_a', 'gadnr': 'gadnr_hidden', 'done': 'done_hidden'}
                if args.base_model in _hidden_map and 'ae_hidden' in aux_params:
                    aux_params[_hidden_map[args.base_model]] = aux_params.pop('ae_hidden')
                print("Using optimized parameters from tuning.")
            else:
                print("No optimized parameters found, using defaults.")

        print("Parameters:")
        for k, v in aux_params.items():
            print("  {}: {}".format(k, v))

        experiment_start = time.time()

        n_aux_val = aux_params['n_aux_nodes']
        if 0 < n_aux_val < 1:
            aux_desc = "{:.1f}% ({})".format(n_aux_val * 100, max(1, int(round(n_aux_val * data.x.shape[0]))))
        else:
            aux_desc = "{}".format(int(n_aux_val))

        run_ensemble_aucs = []
        run_times = []
        run_theoretical_times = []
        run_auprcs = []
        run_p_at_k = []
        run_r_at_k = []
        y_np = data.y.bool().cpu().numpy()

        for run in range(args.n_runs):
            run_start = time.time()
            run_seed = args.seed + run

            print("\n" + "=" * 60)
            print("Run {}/{} (seed={})".format(run + 1, args.n_runs, run_seed))
            print("=" * 60)

            torch.manual_seed(run_seed)
            random.seed(run_seed)
            np.random.seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(run_seed)

            if args.base_model == 'anomalydae':
                ensemble_auc, scores, model_aucs, model_times = train_auxiliary_node_ensemble_anomalydae(
                    data=data,
                    n_models=aux_params['n_aux_models'],
                    n_aux_nodes=aux_params['n_aux_nodes'],
                    feature_method=aux_params['aux_feature_method'],
                    edge_method=aux_params['aux_edge_method'],
                    n_connections=aux_params['aux_n_connections'],
                    k_std=aux_params['aux_k_std'],
                    epochs=aux_params['epochs'],
                    ae_hidden=aux_params['ae_hidden'],
                    ae_dropout=aux_params.get('ae_dropout', 0.3),
                    cle_hidden=aux_params.get('cle_hidden', [256, 512, 256]),
                    base_seed=run_seed,
                    lamda1=aux_params.get('lamda1', 0.5),
                    lamda2=aux_params.get('lamda2', 0.5),
                    normalize_loss=aux_params.get('normalize_loss', True),
                    normalize_method=aux_params.get('normalize_method', 'exponential_moving_average'),
                    normalize_scores=aux_params.get('normalize_scores', True),
                    score_norm_method=aux_params.get('score_norm_method', 'min_max'),
                    use_embedding_transform=aux_params.get('use_embedding_transform', True),
                    use_adaptive_prior=aux_params.get("use_adaptive_prior", True),
                    joint_training=aux_joint,
                    agg_method=aux_params.get('agg_method', 'mean'),
                    lr_ae=aux_params.get('lr_ae', 5e-3),
                    parallel=not args.no_parallel,
                    verbose=False
                )
            elif args.base_model == 'guide':
                ensemble_auc, scores, model_aucs, model_times = train_auxiliary_node_ensemble_guide(
                    data=data,
                    n_models=aux_params['n_aux_models'],
                    n_aux_nodes=aux_params['n_aux_nodes'],
                    feature_method=aux_params['aux_feature_method'],
                    edge_method=aux_params['aux_edge_method'],
                    n_connections=aux_params['aux_n_connections'],
                    k_std=aux_params['aux_k_std'],
                    epochs=aux_params['epochs'],
                    guide_hidden_a=aux_params.get('guide_hidden_a', 64),
                    guide_hidden_s=aux_params.get('guide_hidden_s', 4),
                    guide_num_layers=aux_params.get('guide_num_layers', 4),
                    guide_dropout=aux_params.get('guide_dropout', 0.0),
                    guide_alpha=aux_params.get('guide_alpha', 0.5),
                    cle_hidden=aux_params.get('cle_hidden', [256, 512, 256]),
                    base_seed=run_seed,
                    lamda1=aux_params.get('lamda1', 0.5),
                    lamda2=aux_params.get('lamda2', 0.5),
                    normalize_loss=aux_params.get('normalize_loss', True),
                    normalize_method=aux_params.get('normalize_method', 'exponential_moving_average'),
                    normalize_scores=aux_params.get('normalize_scores', True),
                    score_norm_method=aux_params.get('score_norm_method', 'min_max'),
                    use_embedding_transform=aux_params.get('use_embedding_transform', True),
                    use_adaptive_prior=aux_params.get("use_adaptive_prior", True),
                    joint_training=aux_joint,
                    agg_method=aux_params.get('agg_method', 'mean'),
                    lr_ae=aux_params.get('lr_ae', 5e-3),
                    parallel=not args.no_parallel,
                    verbose=False
                )
            elif args.base_model == 'gadnr':
                ensemble_auc, scores, model_aucs, model_times = train_auxiliary_node_ensemble_gadnr(
                    data=data,
                    n_models=aux_params['n_aux_models'],
                    n_aux_nodes=aux_params['n_aux_nodes'],
                    feature_method=aux_params['aux_feature_method'],
                    edge_method=aux_params['aux_edge_method'],
                    n_connections=aux_params['aux_n_connections'],
                    k_std=aux_params['aux_k_std'],
                    epochs=aux_params['epochs'],
                    gadnr_hidden=aux_params.get('gadnr_hidden', 64),
                    sample_size=aux_params.get('sample_size', 10),
                    encoder=aux_params.get('encoder', 'GCN'),
                    cle_hidden=aux_params.get('cle_hidden', [256, 512, 256]),
                    base_seed=run_seed,
                    lamda1=aux_params.get('lamda1', 0.5),
                    lamda2=aux_params.get('lamda2', 0.5),
                    normalize_loss=aux_params.get('normalize_loss', True),
                    normalize_method=aux_params.get('normalize_method', 'exponential_moving_average'),
                    normalize_scores=aux_params.get('normalize_scores', True),
                    score_norm_method=aux_params.get('score_norm_method', 'min_max'),
                    use_embedding_transform=aux_params.get('use_embedding_transform', True),
                    use_adaptive_prior=aux_params.get("use_adaptive_prior", True),
                    joint_training=aux_joint,
                    agg_method=aux_params.get('agg_method', 'mean'),
                    lr_ae=aux_params.get('lr_ae', 5e-3),
                    parallel=not args.no_parallel,
                    verbose=False
                )
            elif args.base_model == 'done':
                ensemble_auc, scores, model_aucs, model_times = train_auxiliary_node_ensemble_done(
                    data=data,
                    n_models=aux_params['n_aux_models'],
                    n_aux_nodes=aux_params['n_aux_nodes'],
                    feature_method=aux_params['aux_feature_method'],
                    edge_method=aux_params['aux_edge_method'],
                    n_connections=aux_params['aux_n_connections'],
                    k_std=aux_params['aux_k_std'],
                    epochs=aux_params['epochs'],
                    done_hidden=aux_params.get('done_hidden', 64),
                    done_num_layers=aux_params.get('done_num_layers', 4),
                    done_dropout=aux_params.get('done_dropout', 0.0),
                    cle_hidden=aux_params.get('cle_hidden', [256, 512, 256]),
                    base_seed=run_seed,
                    lamda1=aux_params.get('lamda1', 0.5),
                    lamda2=aux_params.get('lamda2', 0.5),
                    normalize_loss=aux_params.get('normalize_loss', True),
                    normalize_method=aux_params.get('normalize_method', 'exponential_moving_average'),
                    normalize_scores=aux_params.get('normalize_scores', True),
                    score_norm_method=aux_params.get('score_norm_method', 'min_max'),
                    use_embedding_transform=aux_params.get('use_embedding_transform', True),
                    use_adaptive_prior=aux_params.get('use_adaptive_prior', args.use_adaptive_prior),
                    joint_training=aux_joint,
                    agg_method=aux_params.get('agg_method', 'mean'),
                    lr_ae=aux_params.get('lr_ae', 5e-3),
                    parallel=not args.no_parallel,
                    verbose=False
                )
            else:
                ensemble_auc, scores, model_aucs, model_times = train_auxiliary_node_ensemble(
                    data=data,
                    n_models=aux_params['n_aux_models'],
                    n_aux_nodes=aux_params['n_aux_nodes'],
                    feature_method=aux_params['aux_feature_method'],
                    edge_method=aux_params['aux_edge_method'],
                    n_connections=aux_params['aux_n_connections'],
                    k_std=aux_params['aux_k_std'],
                    epochs=aux_params['epochs'],
                    ae_hidden=aux_params['ae_hidden'],
                    cle_hidden=aux_params.get('cle_hidden', [256, 512, 256]),
                    base_seed=run_seed,
                    lamda1=aux_params.get('lamda1', 0.5),
                    lamda2=aux_params.get('lamda2', 0.5),
                    normalize_loss=aux_params.get('normalize_loss', True),
                    normalize_method=aux_params.get('normalize_method', 'exponential_moving_average'),
                    normalize_scores=aux_params.get('normalize_scores', True),
                    score_norm_method=aux_params.get('score_norm_method', 'min_max'),
                    use_embedding_transform=aux_params.get('use_embedding_transform', True),
                    use_adaptive_prior=aux_params.get("use_adaptive_prior", True),
                    joint_training=aux_joint,
                    agg_method=aux_params.get('agg_method', 'mean'),
                    dropout=aux_params.get('dropout', 0.3),
                    lr_ae=aux_params.get('lr_ae', 5e-3),
                    struct_weight=aux_params.get('struct_weight', 0.8),
                    parallel=not args.no_parallel,
                    verbose=False
                )

            run_time = time.time() - run_start
            run_times.append(run_time)
            run_ensemble_aucs.append(ensemble_auc)

            # Theoretical ideal: non-model overhead + max(model_time) ≈ fully parallel
            t_other = run_time - np.sum(model_times)
            theoretical_run = t_other + np.max(model_times)
            run_theoretical_times.append(theoretical_run)

            # Compute per-run AUPRC/P@K/R@K
            run_m = compute_all_metrics(y_np, scores)
            run_auprcs.append(run_m['auprc'])
            run_p_at_k.append(run_m['precision_at_k'])
            run_r_at_k.append(run_m['recall_at_k'])

            print("Run {}: Ensemble AUC = {:.4f}, AUPRC = {:.4f}, P@{} = {:.4f}, R@{} = {:.4f} ({:.1f}s, theo ~{:.1f}s)".format(
                run + 1, ensemble_auc, run_m['auprc'], run_m['k'],
                run_m['precision_at_k'], run_m['k'], run_m['recall_at_k'], run_time, theoretical_run))

        total_time = time.time() - experiment_start

        arr_ens = np.array(run_ensemble_aucs)
        arr_auprc = np.array(run_auprcs)
        arr_pk = np.array(run_p_at_k)
        arr_rk = np.array(run_r_at_k)

        print("\n" + "=" * 60)
        print("AUXILIARY NODE ENSEMBLE RESULT ({} runs)".format(args.n_runs))
        print("=" * 60)
        print("Dataset: {}".format(args.dataset))
        print("Base model: {} | Training mode: {}".format(args.base_model.upper(), mode_str))
        print("Models per run: {} | Aux nodes per model: {} | Feature: {} | Edge: {}".format(
            aux_params['n_aux_models'], aux_desc,
            aux_params['aux_feature_method'], aux_params['aux_edge_method']))
        print("Per-run Ensemble AUCs: {}".format(["{:.4f}".format(a) for a in run_ensemble_aucs]))
        print("Ensemble AUC:       {:.4f} ± {:.4f}  (best: {:.4f})".format(
            float(np.mean(arr_ens)), float(np.std(arr_ens)), float(np.max(arr_ens))))
        print("Ensemble AUPRC:     {:.4f} ± {:.4f}  (best: {:.4f})".format(
            float(np.mean(arr_auprc)), float(np.std(arr_auprc)), float(np.max(arr_auprc))))
        print("Ensemble P@{k}:      {:.4f} ± {:.4f}  (best: {:.4f})".format(
            float(np.mean(arr_pk)), float(np.std(arr_pk)), float(np.max(arr_pk)),
            k=run_m['k']))
        print("Ensemble R@{k}:      {:.4f} ± {:.4f}  (best: {:.4f})".format(
            float(np.mean(arr_rk)), float(np.std(arr_rk)), float(np.max(arr_rk)),
            k=run_m['k']))
        print("Total time: {}".format(format_timedelta_precise(timedelta(seconds=total_time))))
        if args.n_runs > 1:
            avg_ensemble_time = np.mean(run_times)
            avg_theoretical_time = np.mean(run_theoretical_times)
            print("Avg per-ensemble time: {:.1f}s (± {:.1f}s)".format(
                avg_ensemble_time, np.std(run_times)))
            print("Theoretical ideal ({} GPUs, 1 model/GPU): ~{:.1f}s per ensemble".format(
                aux_params['n_aux_models'], avg_theoretical_time))
        print("=" * 60)
        return

    # ==================== Normal (non-bagging) Mode ====================

    if args.base_model == 'anomalydae':
        params = {
            'epochs': 100,
            'ae_hidden': 64,
            'ae_dropout': 0.3,
            'cle_hidden': [256, 512, 256],
            'batch_size': 64,
            'normalize_loss': True,
            'normalize_method': 'exponential_moving_average',
            'lamda1': 0.5,
            'lamda2': 0.5,
            'normalize_scores': True,
            'score_norm_method': 'min_max'
        }
    elif args.base_model == 'guide':
        params = {
            'epochs': 100,
            'guide_hidden_a': 64,
            'guide_hidden_s': 4,
            'guide_num_layers': 4,
            'guide_dropout': 0.0,
            'guide_alpha': 0.5,
            'cle_hidden': [256, 512, 256],
            'batch_size': 64,
            'normalize_loss': True,
            'normalize_method': 'exponential_moving_average',
            'lamda1': 0.5,
            'lamda2': 0.5,
            'normalize_scores': True,
            'score_norm_method': 'min_max'
        }
    elif args.base_model == 'gadnr':
        params = {
            'epochs': 100,
            'gadnr_hidden': 64,
            'sample_size': 10,
            'encoder': 'GCN',
            'cle_hidden': [256, 512, 256],
            'batch_size': 64,
            'normalize_loss': True,
            'normalize_method': 'exponential_moving_average',
            'lamda1': 0.5,
            'lamda2': 0.5,
            'normalize_scores': True,
            'score_norm_method': 'min_max'
        }
    elif args.base_model == 'done':
        params = {
            'epochs': 100,
            'done_hidden': 64,
            'done_num_layers': 4,
            'done_dropout': 0.0,
            'cle_hidden': [256, 512, 256],
            'batch_size': 64,
            'normalize_loss': True,
            'normalize_method': 'exponential_moving_average',
            'lamda1': 0.5,
            'lamda2': 0.5,
            'normalize_scores': True,
            'score_norm_method': 'min_max'
        }
    else:
        params = {
            'epochs': 100,
            'ae_hidden': 64,
            'cle_hidden': [256, 512, 256],
            'batch_size': 64,
            'normalize_loss': True,
            'normalize_method': 'exponential_moving_average',
            'lamda1': 0.5,
            'lamda2': 0.5,
            'normalize_scores': True,
            'score_norm_method': 'min_max'
        }

    if args.use_best_params:
        best_params = load_best_params(args.dataset, args.params_dir, mode_suffix)
        if best_params:
            params.update(best_params)
            # Remap generic 'ae_hidden' to model-specific param name
            _hidden_map = {'guide': 'guide_hidden_a', 'gadnr': 'gadnr_hidden', 'done': 'done_hidden'}
            if args.base_model in _hidden_map and 'ae_hidden' in params:
                params[_hidden_map[args.base_model]] = params.pop('ae_hidden')
            print("Using optimized parameters from Optuna tuning.")
        else:
            print("No optimized parameters found, using default parameters.")
    else:
        print("Using default parameters (use --use_best_params to load optimized parameters).")

    print("\nParameters:")
    _cle_keys = {'lamda1', 'lamda2', 'cle_hidden', 'normalize_loss', 'normalize_method'}
    for key, value in params.items():
        if args.ae_only and key in _cle_keys:
            continue
        print(f"  {key}: {value}")

    print(f"\nRunning {args.n_runs} time(s) for averaging results...")

    experiment_start_time = time.time()
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    auc_scores = []
    auprc_scores = []
    pk_scores = []
    rk_scores = []
    run_times = []

    for run in range(args.n_runs):
        run_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{args.n_runs}")
        print('='*60)

        seed = args.seed + run
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        print(f"Random seed: {seed}")

        if args.base_model == 'anomalydae':
            if args.joint_training:
                ae_model, cle_model, result = train_joint_anomalydae_cle(
                    data=data,
                    epochs=params['epochs'],
                    ae_hidden=params['ae_hidden'],
                    ae_dropout=params.get('ae_dropout', 0.3),
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']
            else:
                ae_model, result = train_joint_anomalydae_cle(
                    data=data,
                    epochs=params['epochs'],
                    ae_hidden=params['ae_hidden'],
                    ae_dropout=params.get('ae_dropout', 0.3),
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']
        elif args.base_model == 'guide':
            if args.joint_training:
                ae_model, cle_model, result = train_joint_guide_cle(
                    data=data,
                    epochs=params['epochs'],
                    guide_hidden_a=params.get('guide_hidden_a', 64),
                    guide_hidden_s=params.get('guide_hidden_s', 4),
                    guide_num_layers=params.get('guide_num_layers', 4),
                    guide_dropout=params.get('guide_dropout', 0.0),
                    guide_alpha=params.get('guide_alpha', 0.5),
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']
            else:
                ae_model, result = train_joint_guide_cle(
                    data=data,
                    epochs=params['epochs'],
                    guide_hidden_a=params.get('guide_hidden_a', 64),
                    guide_hidden_s=params.get('guide_hidden_s', 4),
                    guide_num_layers=params.get('guide_num_layers', 4),
                    guide_dropout=params.get('guide_dropout', 0.0),
                    guide_alpha=params.get('guide_alpha', 0.5),
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']
        elif args.base_model == 'gadnr':
            if args.joint_training:
                ae_model, cle_model, result = train_joint_gadnr_cle(
                    data=data,
                    epochs=params['epochs'],
                    gadnr_hidden=params.get('gadnr_hidden', 64),
                    sample_size=params.get('sample_size', 10),
                    encoder=params.get('encoder', 'GCN'),
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']
            else:
                ae_model, result = train_joint_gadnr_cle(
                    data=data,
                    epochs=params['epochs'],
                    gadnr_hidden=params.get('gadnr_hidden', 64),
                    sample_size=params.get('sample_size', 10),
                    encoder=params.get('encoder', 'GCN'),
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']
        elif args.base_model == 'done':
            if args.joint_training:
                ae_model, cle_model, result = train_joint_done_cle(
                    data=data,
                    epochs=params['epochs'],
                    done_hidden=params.get('done_hidden', 64),
                    done_num_layers=params.get('done_num_layers', 4),
                    done_dropout=params.get('done_dropout', 0.0),
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']
            else:
                ae_model, result = train_joint_done_cle(
                    data=data,
                    epochs=params['epochs'],
                    done_hidden=params.get('done_hidden', 64),
                    done_num_layers=params.get('done_num_layers', 4),
                    done_dropout=params.get('done_dropout', 0.0),
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']
        else:
            if args.joint_training:
                ae_model, cle_model, result = train_joint_ae_cle(
                    data=data,
                    epochs=params['epochs'],
                    ae_hidden=params['ae_hidden'],
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']
            else:
                ae_model, result = train_joint_ae_cle(
                    data=data,
                    epochs=params['epochs'],
                    ae_hidden=params['ae_hidden'],
                    cle_hidden=params['cle_hidden'],
                    batch_size=params['batch_size'],
                    normalize_loss=params['normalize_loss'],
                    normalize_method=params['normalize_method'],
                    lamda1=params['lamda1'],
                    lamda2=params['lamda2'],
                    normalize_scores=params['normalize_scores'],
                    score_norm_method=params['score_norm_method'],
                    joint_training=args.joint_training,
                    dataset_name=args.dataset,
                    use_adaptive_prior=params.get('use_adaptive_prior', args.use_adaptive_prior),
                    use_embedding_transform=params.get('use_embedding_transform', args.use_embedding_transform)
                )
                final_auc = result['auc']

        run_time = time.time() - run_start_time
        run_times.append(run_time)

        auc_scores.append(final_auc)
        auprc_scores.append(result['auprc'])
        pk_scores.append(result['precision_at_k'])
        rk_scores.append(result['recall_at_k'])
        k_val = result.get('k', int(data.y.bool().sum()))
        print(f"Run {run + 1} AUC: {final_auc:.6f} AUPRC: {result['auprc']:.6f} P@{k_val}: {result['precision_at_k']:.4f} R@{k_val}: {result['recall_at_k']:.4f} (Time: {format_timedelta_precise(timedelta(seconds=run_time))})")

    auc_scores = np.array(auc_scores)
    auprc_scores = np.array(auprc_scores)
    pk_scores = np.array(pk_scores)
    rk_scores = np.array(rk_scores)

    if len(auc_scores) >= 3:
        sorted_auc = np.sort(auc_scores)
        keep_idx = np.argsort(auc_scores)[2:]
        mean_auc = np.mean(auc_scores[keep_idx])
        std_auc = np.std(auc_scores[keep_idx])
        mean_auprc = np.mean(auprc_scores[keep_idx])
        std_auprc = np.std(auprc_scores[keep_idx])
        mean_pk = np.mean(pk_scores[keep_idx])
        std_pk = np.std(pk_scores[keep_idx])
        mean_rk = np.mean(rk_scores[keep_idx])
        std_rk = np.std(rk_scores[keep_idx])
    else:
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        mean_auprc = np.mean(auprc_scores)
        std_auprc = np.std(auprc_scores)
        mean_pk = np.mean(pk_scores)
        std_pk = np.std(pk_scores)
        mean_rk = np.mean(rk_scores)
        std_rk = np.std(rk_scores)

    if len(run_times) > 1:
        filtered_run_times = run_times[1:]
        avg_run_time = np.mean(filtered_run_times)
        run_time_note = f" (excluding first run, based on {len(filtered_run_times)} runs)"
    else:
        filtered_run_times = run_times
        avg_run_time = np.mean(filtered_run_times)
        run_time_note = ""

    total_experiment_time = time.time() - experiment_start_time

    print(f"\nExperiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Base model: {args.base_model.upper()}")
    print(f"Training mode: {'Joint AE+CLE' if not args.ae_only else 'AE only'}")
    print(f"Number of runs: {args.n_runs}")
    print(f"Individual AUC scores: {[f'{score:.6f}' for score in auc_scores]}")
    print(f"Individual run times: {[f'{format_timedelta_precise(timedelta(seconds=t))}' for t in run_times]}")
    if len(run_times) > 1:
        print(f"Filtered run times (excluding first): {[f'{format_timedelta_precise(timedelta(seconds=t))}' for t in filtered_run_times]}")
    best_auc = float(np.max(auc_scores[keep_idx])) if len(auc_scores) >= 3 else float(np.max(auc_scores))
    best_auprc = float(np.max(auprc_scores[keep_idx])) if len(auc_scores) >= 3 else float(np.max(auprc_scores))
    best_pk = float(np.max(pk_scores[keep_idx])) if len(auc_scores) >= 3 else float(np.max(pk_scores))
    best_rk = float(np.max(rk_scores[keep_idx])) if len(auc_scores) >= 3 else float(np.max(rk_scores))
    print(f"AUC:   {mean_auc:.6f} ± {std_auc:.6f}  (best: {best_auc:.6f})")
    print(f"AUPRC: {mean_auprc:.6f} ± {std_auprc:.6f}  (best: {best_auprc:.6f})")
    print(f"P@{k_val}:  {mean_pk:.6f} ± {std_pk:.6f}  (best: {best_pk:.6f})")
    print(f"R@{k_val}:  {mean_rk:.6f} ± {std_rk:.6f}  (best: {best_rk:.6f})")
    print(f"Average run time: {format_timedelta_precise(timedelta(seconds=avg_run_time))}{run_time_note}")
    print(f"Total experiment time: {format_timedelta_precise(timedelta(seconds=total_experiment_time))}")
    print("=" * 60)
