"""Ensemble training: auxiliary node ensemble for AE+CLE."""

import torch
import numpy as np
import time
import gc
import multiprocessing as mp
from torch_geometric.data import Data as PyGData

from .dominant import _train_single_joint
from .augment import add_auxiliary_nodes
from .utils import compute_all_metrics


def _train_one_aux_worker(task):
    """Train a single auxiliary node model on a specific GPU (for multiprocessing).

    Parameters
    ----------
    task : dict with keys:
        gpu_id, x_np, y_np, edge_index_np, n_orig, model_idx, base_seed,
        n_aux_nodes, feature_method, edge_method, n_connections, k_std,
        epochs, ae_hidden, cle_hidden, lamda1, lamda2,
        normalize_loss, normalize_method, normalize_scores, score_norm_method,
        use_embedding_transform, joint_training, dropout, lr_ae, struct_weight

    Returns
    -------
    combined_orig : np.ndarray (n_orig,)
    model_auc : float
    """
    import gc, time
    t_start = time.time()
    gpu_id = task['gpu_id']
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Reconstruct PyG Data from numpy arrays
    x = torch.tensor(task['x_np'], dtype=torch.float32, device=device)
    y = torch.tensor(task['y_np'], dtype=torch.long, device=device)
    edge_index = torch.tensor(task['edge_index_np'], dtype=torch.long, device=device)
    data = PyGData(x=x, y=y, edge_index=edge_index)

    seed = task['base_seed'] + task['model_idx'] * 100
    n_orig = task['n_orig']
    y_orig = task['y_orig_np']  # already numpy bool

    aug_data, _ = add_auxiliary_nodes(
        data, n_aux=task['n_aux_nodes'],
        feature_method=task['feature_method'],
        edge_method=task['edge_method'],
        n_connections=task['n_connections'],
        k_std=task['k_std'],
        seed=seed
    )

    _, _, combined_all = _train_single_joint(
        aug_data,
        epochs=task['epochs'],
        ae_hidden=task['ae_hidden'],
        cle_hidden=task['cle_hidden'],
        device=device,
        seed=seed,
        lamda1=task['lamda1'],
        lamda2=task['lamda2'],
        normalize_loss=task['normalize_loss'],
        normalize_method=task['normalize_method'],
        normalize_scores=task['normalize_scores'],
        score_norm_method=task['score_norm_method'],
        use_embedding_transform=task['use_embedding_transform'],
        joint_training=task['joint_training'],
        verbose=False,  # quiet in parallel mode
        dropout=task['dropout'],
        lr_ae=task['lr_ae'],
        struct_weight=task['struct_weight'],
        use_adaptive_prior=task.get('use_adaptive_prior', True)
    )

    combined_orig = combined_all[:n_orig]  # already numpy from _train_single_joint
    model_m = compute_all_metrics(y_orig, combined_orig)
    model_auc = model_m['auc']
    model_time = time.time() - t_start

    # Clean up CUDA memory
    del aug_data, data, combined_all
    torch.cuda.empty_cache()
    gc.collect()

    return combined_orig, model_auc, model_time


def train_auxiliary_node_ensemble(data, n_models=10, n_aux_nodes=5,
                                   feature_method='outlier_tail',
                                   edge_method='random_connect',
                                   n_connections=5, k_std=3.0,
                                   epochs=100, ae_hidden=64, cle_hidden=None,
                                   device=None, base_seed=42,
                                   lamda1=0.5, lamda2=0.5,
                                   normalize_loss=True,
                                   normalize_method='exponential_moving_average',
                                   normalize_scores=True,
                                   score_norm_method='min_max',
                                   use_embedding_transform=True,
                                   joint_training=True, verbose=True,
                                   agg_method='mean',
                                   dropout=0.3, lr_ae=5e-3, struct_weight=0.8,
                                   parallel=True,
                                   use_adaptive_prior=True):
    """Auxiliary node ensemble: each model adds different synthetic anomaly nodes without modifying original graph.

    Parameters
    ----------
    data : PyGData
    n_models : int, number of ensemble models
    n_aux_nodes : int or float, number of auxiliary nodes per model
        If 0 < n_aux_nodes < 1, treated as ratio (e.g. 0.05 = 5% of original nodes)
    feature_method : str, feature construction method
        'outlier_tail' | 'gaussian_noise' | 'perturb_existing' |
        'smote_outlier' | 'neighbor_dissimilar' | 'feature_shuffle'
    edge_method : str, edge construction method
        'random_connect' | 'isolated' | 'clique' | 'low_similarity_connect'
    n_connections : int, number of real nodes each aux node connects to
    k_std : float, std multiplier for outlier_tail/smote_outlier/neighbor_dissimilar
    [other params same as _train_single_joint]

    Returns
    -------
    ensemble_auc : float
    final_scores : np.ndarray (n_orig_nodes,)
    model_aucs : np.ndarray (n_models,)
    """
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_orig = data.x.shape[0]
    y_orig = data.y.bool().cpu().numpy()

    all_scores = []
    model_aucs = []
    model_times = []

    training_start = time.time()
    mode_str = "AE+CLE" if joint_training else "AE only"

    if verbose:
        print("=" * 55)
        if 0 < n_aux_nodes < 1:
            n_aux_actual = max(1, int(round(n_aux_nodes * n_orig)))
            aux_desc = "{:.1f}% ({} nodes)".format(n_aux_nodes * 100, n_aux_actual)
        else:
            aux_desc = "{} nodes".format(int(n_aux_nodes))
        print("Auxiliary Node Ensemble: {} models x {}, +{} aux per model".format(
            n_models, mode_str, aux_desc))
        print("Feature method: {} | Edge method: {} | n_conn: {} | k_std: {}".format(
            feature_method, edge_method, n_connections, k_std))
        print("=" * 55)

    # ---- Determine execution mode ----
    n_gpus = torch.cuda.device_count() if (parallel and torch.cuda.is_available()) else 0
    use_parallel = n_gpus > 1

    if use_parallel and n_models > 1:
        # ============ Multi-GPU Parallel Mode ============
        if verbose:
            print("Parallel mode: {} GPUs, {} models -> ~{:.0f} models/GPU".format(
                n_gpus, n_models, n_models / n_gpus))

        # Prepare task dicts (plain Python types for pickling)
        x_np = data.x.cpu().numpy().astype(np.float32)
        y_np = data.y.cpu().numpy().astype(np.int64)
        edge_np = data.edge_index.cpu().numpy().astype(np.int64)
        y_orig_np = y_orig.astype(np.float64)

        tasks = []
        for model_idx in range(n_models):
            tasks.append({
                'gpu_id': model_idx % n_gpus,
                'x_np': x_np, 'y_np': y_np, 'edge_index_np': edge_np,
                'y_orig_np': y_orig_np, 'n_orig': n_orig,
                'model_idx': model_idx, 'base_seed': base_seed,
                'n_aux_nodes': n_aux_nodes,
                'feature_method': feature_method,
                'edge_method': edge_method,
                'n_connections': n_connections,
                'k_std': k_std,
                'epochs': epochs, 'ae_hidden': ae_hidden,
                'cle_hidden': cle_hidden,
                'lamda1': lamda1, 'lamda2': lamda2,
                'normalize_loss': normalize_loss,
                'normalize_method': normalize_method,
                'normalize_scores': normalize_scores,
                'score_norm_method': score_norm_method,
                'use_embedding_transform': use_embedding_transform,
                'use_adaptive_prior': use_adaptive_prior,
                'joint_training': joint_training,
                'dropout': dropout, 'lr_ae': lr_ae,
                'struct_weight': struct_weight,
            })

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_gpus) as pool:
            # imap_unordered with chunksize=1 for dynamic load balancing
            results = list(pool.imap_unordered(_train_one_aux_worker, tasks, chunksize=1))

        completed = 0
        for combined_orig, model_auc, model_time in results:
            all_scores.append(combined_orig)
            model_aucs.append(model_auc)
            model_times.append(model_time)
            completed += 1
            if verbose:
                print("Model AUC: {:.4f}  [{}/{} done]".format(
                    model_auc, completed, n_models))

    else:
        # ============ Sequential Mode (single GPU or CPU) ============
        if verbose and n_gpus <= 1:
            print("Sequential mode: {} GPU(s), {} models".format(
                max(n_gpus, 0 if not torch.cuda.is_available() else 1), n_models))

        for model_idx in range(n_models):
            model_start = time.time()
            seed = base_seed + model_idx * 100

            if verbose:
                print("\n" + "=" * 55)
                print("Model {}/{} (seed={}, {})".format(
                    model_idx + 1, n_models, seed, mode_str))
                print("=" * 55)

            aug_data, _ = add_auxiliary_nodes(
                data, n_aux=n_aux_nodes,
                feature_method=feature_method,
                edge_method=edge_method,
                n_connections=n_connections,
                k_std=k_std,
                seed=seed
            )

            if verbose:
                print("Augmented graph: {} nodes ({} orig + {} aux)".format(
                    aug_data.x.shape[0], n_orig, n_aux_nodes))

            _, _, combined_all = _train_single_joint(
                aug_data, epochs=epochs, ae_hidden=ae_hidden,
                cle_hidden=cle_hidden, device=device, seed=seed,
                lamda1=lamda1, lamda2=lamda2,
                normalize_loss=normalize_loss,
                normalize_method=normalize_method,
                normalize_scores=normalize_scores,
                score_norm_method=score_norm_method,
                use_embedding_transform=use_embedding_transform,
                joint_training=joint_training, verbose=verbose,
                    use_adaptive_prior=use_adaptive_prior,
                dropout=dropout, lr_ae=lr_ae, struct_weight=struct_weight
            )

            combined_orig = combined_all[:n_orig]
            all_scores.append(combined_orig)
            model_time = time.time() - model_start
            model_m = compute_all_metrics(y_orig, combined_orig)
            model_aucs.append(model_m['auc'])
            model_times.append(model_time)

            if verbose:
                print("Model {}/{} AUC: {auc:.4f} AUPRC: {auprc:.4f} P@{k}: {precision_at_k:.4f} R@{k}: {recall_at_k:.4f} | Time: {:.1f}s".format(
                    model_idx + 1, n_models, model_time, **model_m))

    if len(all_scores) == 0:
        raise RuntimeError("All ensemble models failed — no valid scores")
    all_scores = np.array(all_scores)
    if agg_method == 'max':
        final_scores = np.max(all_scores, axis=0)
    elif agg_method == 'median':
        final_scores = np.median(all_scores, axis=0)
    else:  # 'mean'
        final_scores = np.mean(all_scores, axis=0)
    # Ensemble metrics
    ens_m = compute_all_metrics(y_orig, final_scores)

    model_aucs_arr = np.array(model_aucs)

    if verbose:
        model_auc_mean = float(np.mean(model_aucs_arr))
        model_auc_std = float(np.std(model_aucs_arr))
        model_auc_best = float(np.max(model_aucs_arr))
        print("\n" + "=" * 55)
        print("Auxiliary Node Ensemble Complete")
        print("=" * 55)
        print("Model AUCs: mean={:.4f} +/- {:.4f}, best={:.4f}".format(
            model_auc_mean, model_auc_std, model_auc_best))
        print("Ensemble: AUC={auc:.4f} AUPRC={auprc:.4f} P@{k}={precision_at_k:.4f} R@{k}={recall_at_k:.4f}".format(**ens_m))
        total_time = time.time() - training_start
        print("Total time: {:.1f}s".format(total_time))

    return ens_m['auc'], final_scores, model_aucs_arr, np.array(model_times)


# ==================== AnomalyDAE Ensemble ====================

def _train_one_aux_worker_anomalydae(task):
    """Train a single AnomalyDAE auxiliary node model on a specific GPU (for multiprocessing).

    Parallel to _train_one_aux_worker but uses AnomalyDAE instead of DOMINANT.
    Uses ae_dropout instead of dropout+struct_weight.

    Returns
    -------
    combined_orig : np.ndarray (n_orig,)
    model_auc : float
    """
    import gc, time
    t_start = time.time()
    gpu_id = task['gpu_id']
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor(task['x_np'], dtype=torch.float32, device=device)
    y = torch.tensor(task['y_np'], dtype=torch.long, device=device)
    edge_index = torch.tensor(task['edge_index_np'], dtype=torch.long, device=device)
    data = PyGData(x=x, y=y, edge_index=edge_index)

    seed = task['base_seed'] + task['model_idx'] * 100
    n_orig = task['n_orig']
    y_orig = task['y_orig_np']

    aug_data, _ = add_auxiliary_nodes(
        data, n_aux=task['n_aux_nodes'],
        feature_method=task['feature_method'],
        edge_method=task['edge_method'],
        n_connections=task['n_connections'],
        k_std=task['k_std'],
        seed=seed
    )

    from .anomalydae import _train_single_joint_anomalydae
    ae_score, _, combined_all = _train_single_joint_anomalydae(
        aug_data,
        epochs=task['epochs'],
        ae_hidden=task['ae_hidden'],
        ae_dropout=task['ae_dropout'],
        cle_hidden=task['cle_hidden'],
        device=device,
        seed=seed,
        lamda1=task['lamda1'],
        lamda2=task['lamda2'],
        normalize_loss=task['normalize_loss'],
        normalize_method=task['normalize_method'],
        normalize_scores=task['normalize_scores'],
        score_norm_method=task['score_norm_method'],
        use_embedding_transform=task['use_embedding_transform'],
        joint_training=task['joint_training'],
        verbose=False
    )

    combined_orig = combined_all[:n_orig]
    model_m = compute_all_metrics(y_orig, combined_orig)
    model_auc = model_m['auc']
    model_time = time.time() - t_start

    del aug_data, data, combined_all, ae_score
    torch.cuda.empty_cache()
    gc.collect()

    return combined_orig, model_auc, model_time


def train_auxiliary_node_ensemble_anomalydae(data, n_models=10, n_aux_nodes=5,
                                              feature_method='outlier_tail',
                                              edge_method='random_connect',
                                              n_connections=5, k_std=3.0,
                                              epochs=100, ae_hidden=64, ae_dropout=0.3,
                                              cle_hidden=None,
                                              device=None, base_seed=42,
                                              lamda1=0.5, lamda2=0.5,
                                              normalize_loss=True,
                                              normalize_method='exponential_moving_average',
                                              normalize_scores=True,
                                              score_norm_method='min_max',
                                              use_embedding_transform=True,
                                              joint_training=True, verbose=True,
                                              agg_method='mean',
                                              lr_ae=5e-3,
                                              parallel=True,
                                   use_adaptive_prior=True):
    """Auxiliary node ensemble using AnomalyDAE base model.

    Parallel to train_auxiliary_node_ensemble but uses AnomalyDAE.
    Uses ae_dropout instead of dropout+struct_weight.

    Parameters
    ----------
    data : PyGData
    n_models : int
    n_aux_nodes : int or float
    feature_method : str
    edge_method : str
    n_connections : int
    k_std : float
    epochs : int
    ae_hidden : int
    ae_dropout : float — dropout for AnomalyDAE layers
    cle_hidden : list[int] | None
    device : torch.device | None
    base_seed : int
    lamda1 : float
    lamda2 : float
    normalize_loss : bool
    normalize_method : str
    normalize_scores : bool
    score_norm_method : str
    use_embedding_transform : bool
    joint_training : bool
    verbose : bool
    agg_method : str
    lr_ae : float
    parallel : bool

    Returns
    -------
    ensemble_auc : float
    final_scores : np.ndarray (n_orig_nodes,)
    model_aucs : np.ndarray (n_models,)
    """
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_orig = data.x.shape[0]
    y_orig = data.y.bool().cpu().numpy()

    all_scores = []
    model_aucs = []
    model_times = []

    training_start = time.time()
    mode_str = "AE+CLE" if joint_training else "AE only"

    if verbose:
        print("=" * 55)
        if 0 < n_aux_nodes < 1:
            n_aux_actual = max(1, int(round(n_aux_nodes * n_orig)))
            aux_desc = "{:.1f}% ({} nodes)".format(n_aux_nodes * 100, n_aux_actual)
        else:
            aux_desc = "{} nodes".format(int(n_aux_nodes))
        print("AnomalyDAE Auxiliary Node Ensemble: {} models x {}, +{} aux per model".format(
            n_models, mode_str, aux_desc))
        print("Feature method: {} | Edge method: {} | n_conn: {} | k_std: {}".format(
            feature_method, edge_method, n_connections, k_std))
        print("ae_dropout: {}".format(ae_dropout))
        print("=" * 55)

    # ---- Determine execution mode ----
    n_gpus = torch.cuda.device_count() if (parallel and torch.cuda.is_available()) else 0
    use_parallel = n_gpus > 1

    if use_parallel and n_models > 1:
        # ============ Multi-GPU Parallel Mode ============
        if verbose:
            print("Parallel mode: {} GPUs, {} models -> ~{:.0f} models/GPU".format(
                n_gpus, n_models, n_models / n_gpus))

        x_np = data.x.cpu().numpy().astype(np.float32)
        y_np = data.y.cpu().numpy().astype(np.int64)
        edge_np = data.edge_index.cpu().numpy().astype(np.int64)
        y_orig_np = y_orig.astype(np.float64)

        tasks = []
        for model_idx in range(n_models):
            tasks.append({
                'gpu_id': model_idx % n_gpus,
                'x_np': x_np, 'y_np': y_np, 'edge_index_np': edge_np,
                'y_orig_np': y_orig_np, 'n_orig': n_orig,
                'model_idx': model_idx, 'base_seed': base_seed,
                'n_aux_nodes': n_aux_nodes,
                'feature_method': feature_method,
                'edge_method': edge_method,
                'n_connections': n_connections,
                'k_std': k_std,
                'epochs': epochs, 'ae_hidden': ae_hidden,
                'ae_dropout': ae_dropout,
                'cle_hidden': cle_hidden,
                'lamda1': lamda1, 'lamda2': lamda2,
                'normalize_loss': normalize_loss,
                'normalize_method': normalize_method,
                'normalize_scores': normalize_scores,
                'score_norm_method': score_norm_method,
                'use_embedding_transform': use_embedding_transform,
                'use_adaptive_prior': use_adaptive_prior,
                'joint_training': joint_training,
            })

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_gpus) as pool:
            results = list(pool.imap_unordered(_train_one_aux_worker_anomalydae, tasks, chunksize=1))

        completed = 0
        for combined_orig, model_auc, model_time in results:
            all_scores.append(combined_orig)
            model_aucs.append(model_auc)
            model_times.append(model_time)
            completed += 1
            if verbose:
                print("Model AUC: {:.4f}  [{}/{} done]".format(
                    model_auc, completed, n_models))

    else:
        # ============ Sequential Mode ============
        if verbose and n_gpus <= 1:
            print("Sequential mode: {} GPU(s), {} models".format(
                max(n_gpus, 0 if not torch.cuda.is_available() else 1), n_models))

        from .anomalydae import _train_single_joint_anomalydae

        for model_idx in range(n_models):
            model_start = time.time()
            seed = base_seed + model_idx * 100

            if verbose:
                print("\n" + "=" * 55)
                print("Model {}/{} (seed={}, AnomalyDAE {})".format(
                    model_idx + 1, n_models, seed, mode_str))
                print("=" * 55)

            aug_data, _ = add_auxiliary_nodes(
                data, n_aux=n_aux_nodes,
                feature_method=feature_method,
                edge_method=edge_method,
                n_connections=n_connections,
                k_std=k_std,
                seed=seed
            )

            if verbose:
                n_aux_actual = aug_data.x.shape[0] - n_orig
                print("Augmented graph: {} nodes ({} orig + {} aux)".format(
                    aug_data.x.shape[0], n_orig, n_aux_actual))

            ae_score, _, combined_all = _train_single_joint_anomalydae(
                aug_data, epochs=epochs, ae_hidden=ae_hidden,
                ae_dropout=ae_dropout,
                cle_hidden=cle_hidden, device=device, seed=seed,
                lamda1=lamda1, lamda2=lamda2,
                normalize_loss=normalize_loss,
                normalize_method=normalize_method,
                normalize_scores=normalize_scores,
                score_norm_method=score_norm_method,
                use_embedding_transform=use_embedding_transform,
                joint_training=joint_training, verbose=verbose,
                    use_adaptive_prior=use_adaptive_prior
            )

            combined_orig = combined_all[:n_orig]
            all_scores.append(combined_orig)
            model_time = time.time() - model_start
            model_m = compute_all_metrics(y_orig, combined_orig)
            model_aucs.append(model_m['auc'])
            model_times.append(model_time)

            if verbose:
                print("Model {}/{} AUC: {auc:.4f} AUPRC: {auprc:.4f} P@{k}: {precision_at_k:.4f} R@{k}: {recall_at_k:.4f} | Time: {:.1f}s".format(
                    model_idx + 1, n_models, model_time, **model_m))

    if len(all_scores) == 0:
        raise RuntimeError("All ensemble models failed — no valid scores")
    all_scores = np.array(all_scores)
    if agg_method == 'max':
        final_scores = np.max(all_scores, axis=0)
    elif agg_method == 'median':
        final_scores = np.median(all_scores, axis=0)
    else:
        final_scores = np.mean(all_scores, axis=0)

    ens_m = compute_all_metrics(y_orig, final_scores)
    model_aucs_arr = np.array(model_aucs)

    if verbose:
        model_auc_mean = float(np.mean(model_aucs_arr))
        model_auc_std = float(np.std(model_aucs_arr))
        model_auc_best = float(np.max(model_aucs_arr))
        print("\n" + "=" * 55)
        print("AnomalyDAE Auxiliary Node Ensemble Complete")
        print("=" * 55)
        print("Model AUCs: mean={:.4f} +/- {:.4f}, best={:.4f}".format(
            model_auc_mean, model_auc_std, model_auc_best))
        print("Ensemble: AUC={auc:.4f} AUPRC={auprc:.4f} P@{k}={precision_at_k:.4f} R@{k}={recall_at_k:.4f}".format(**ens_m))
        total_time = time.time() - training_start
        print("Total time: {:.1f}s".format(total_time))

    return ens_m['auc'], final_scores, model_aucs_arr, np.array(model_times)


# ==================== GUIDE Ensemble ====================

def _train_one_aux_worker_guide(task):
    """Train a single GUIDE auxiliary node model on a specific GPU (for multiprocessing).

    Parallel to _train_one_aux_worker and _train_one_aux_worker_anomalydae.
    Uses GUIDE-specific params: guide_hidden_a, guide_hidden_s, guide_num_layers,
    guide_dropout, guide_alpha.

    Returns
    -------
    combined_orig : np.ndarray (n_orig,)
    model_auc : float
    """
    import gc, time
    t_start = time.time()
    gpu_id = task['gpu_id']
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor(task['x_np'], dtype=torch.float32, device=device)
    y = torch.tensor(task['y_np'], dtype=torch.long, device=device)
    edge_index = torch.tensor(task['edge_index_np'], dtype=torch.long, device=device)
    data = PyGData(x=x, y=y, edge_index=edge_index)

    seed = task['base_seed'] + task['model_idx'] * 100
    n_orig = task['n_orig']
    y_orig = task['y_orig_np']

    aug_data, _ = add_auxiliary_nodes(
        data, n_aux=task['n_aux_nodes'],
        feature_method=task['feature_method'],
        edge_method=task['edge_method'],
        n_connections=task['n_connections'],
        k_std=task['k_std'],
        seed=seed
    )

    from .guide import _train_single_joint_guide
    ae_score, _, combined_all = _train_single_joint_guide(
        aug_data,
        epochs=task['epochs'],
        guide_hidden_a=task['guide_hidden_a'],
        guide_hidden_s=task['guide_hidden_s'],
        guide_num_layers=task['guide_num_layers'],
        guide_dropout=task['guide_dropout'],
        guide_alpha=task['guide_alpha'],
        cle_hidden=task['cle_hidden'],
        device=device,
        seed=seed,
        lamda1=task['lamda1'],
        lamda2=task['lamda2'],
        normalize_loss=task['normalize_loss'],
        normalize_method=task['normalize_method'],
        normalize_scores=task['normalize_scores'],
        score_norm_method=task['score_norm_method'],
        use_embedding_transform=task['use_embedding_transform'],
        joint_training=task['joint_training'],
        verbose=False
    )

    combined_orig = combined_all[:n_orig]
    model_m = compute_all_metrics(y_orig, combined_orig)
    model_auc = model_m['auc']
    model_time = time.time() - t_start

    del aug_data, data, combined_all, ae_score
    torch.cuda.empty_cache()
    gc.collect()

    return combined_orig, model_auc, model_time


def train_auxiliary_node_ensemble_guide(data, n_models=10, n_aux_nodes=5,
                                         feature_method='outlier_tail',
                                         edge_method='random_connect',
                                         n_connections=5, k_std=3.0,
                                         epochs=100, guide_hidden_a=64, guide_hidden_s=4,
                                         guide_num_layers=4, guide_dropout=0.0, guide_alpha=0.5,
                                         cle_hidden=None,
                                         device=None, base_seed=42,
                                         lamda1=0.5, lamda2=0.5,
                                         normalize_loss=True,
                                         normalize_method='exponential_moving_average',
                                         normalize_scores=True,
                                         score_norm_method='min_max',
                                         use_embedding_transform=True,
                                         joint_training=True, verbose=True,
                                         agg_method='mean',
                                         lr_ae=5e-3,
                                         parallel=True,
                                   use_adaptive_prior=True):
    """Auxiliary node ensemble using GUIDE base model.

    Parallel to train_auxiliary_node_ensemble and
    train_auxiliary_node_ensemble_anomalydae.

    Uses GUIDE-specific params: guide_hidden_a, guide_hidden_s, guide_num_layers,
    guide_dropout, guide_alpha.

    Parameters
    ----------
    data : PyGData
    n_models : int
    n_aux_nodes : int or float
    feature_method : str
    edge_method : str
    n_connections : int
    k_std : float
    epochs : int
    guide_hidden_a : int — attribute encoder hidden size
    guide_hidden_s : int — structure encoder hidden size
    guide_num_layers : int
    guide_dropout : float
    guide_alpha : float — attribute/structure loss balance
    cle_hidden : list[int] | None
    device : torch.device | None
    base_seed : int
    lamda1 : float
    lamda2 : float
    normalize_loss : bool
    normalize_method : str
    normalize_scores : bool
    score_norm_method : str
    use_embedding_transform : bool
    joint_training : bool
    verbose : bool
    agg_method : str
    lr_ae : float (kept for API consistency, not used by GUIDE)
    parallel : bool

    Returns
    -------
    ensemble_auc : float
    final_scores : np.ndarray (n_orig_nodes,)
    model_aucs : np.ndarray (n_models,)
    """
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_orig = data.x.shape[0]
    y_orig = data.y.bool().cpu().numpy()

    all_scores = []
    model_aucs = []
    model_times = []

    training_start = time.time()
    mode_str = "AE+CLE" if joint_training else "AE only"

    if verbose:
        print("=" * 55)
        if 0 < n_aux_nodes < 1:
            n_aux_actual = max(1, int(round(n_aux_nodes * n_orig)))
            aux_desc = "{:.1f}% ({} nodes)".format(n_aux_nodes * 100, n_aux_actual)
        else:
            aux_desc = "{} nodes".format(int(n_aux_nodes))
        print("GUIDE Auxiliary Node Ensemble: {} models x {}, +{} aux per model".format(
            n_models, mode_str, aux_desc))
        print("Feature method: {} | Edge method: {} | n_conn: {} | k_std: {}".format(
            feature_method, edge_method, n_connections, k_std))
        print("guide: hid_a={}, hid_s={}, layers={}, dropout={}, alpha={}".format(
            guide_hidden_a, guide_hidden_s, guide_num_layers, guide_dropout, guide_alpha))
        print("=" * 55)

    # ---- Determine execution mode ----
    n_gpus = torch.cuda.device_count() if (parallel and torch.cuda.is_available()) else 0
    use_parallel = n_gpus > 1

    if use_parallel and n_models > 1:
        # ============ Multi-GPU Parallel Mode ============
        if verbose:
            print("Parallel mode: {} GPUs, {} models -> ~{:.0f} models/GPU".format(
                n_gpus, n_models, n_models / n_gpus))

        x_np = data.x.cpu().numpy().astype(np.float32)
        y_np = data.y.cpu().numpy().astype(np.int64)
        edge_np = data.edge_index.cpu().numpy().astype(np.int64)
        y_orig_np = y_orig.astype(np.float64)

        tasks = []
        for model_idx in range(n_models):
            tasks.append({
                'gpu_id': model_idx % n_gpus,
                'x_np': x_np, 'y_np': y_np, 'edge_index_np': edge_np,
                'y_orig_np': y_orig_np, 'n_orig': n_orig,
                'model_idx': model_idx, 'base_seed': base_seed,
                'n_aux_nodes': n_aux_nodes,
                'feature_method': feature_method,
                'edge_method': edge_method,
                'n_connections': n_connections,
                'k_std': k_std,
                'epochs': epochs,
                'guide_hidden_a': guide_hidden_a,
                'guide_hidden_s': guide_hidden_s,
                'guide_num_layers': guide_num_layers,
                'guide_dropout': guide_dropout,
                'guide_alpha': guide_alpha,
                'cle_hidden': cle_hidden,
                'lamda1': lamda1, 'lamda2': lamda2,
                'normalize_loss': normalize_loss,
                'normalize_method': normalize_method,
                'normalize_scores': normalize_scores,
                'score_norm_method': score_norm_method,
                'use_embedding_transform': use_embedding_transform,
                'use_adaptive_prior': use_adaptive_prior,
                'joint_training': joint_training,
            })

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_gpus) as pool:
            results = list(pool.imap_unordered(_train_one_aux_worker_guide, tasks, chunksize=1))

        completed = 0
        for combined_orig, model_auc, model_time in results:
            all_scores.append(combined_orig)
            model_aucs.append(model_auc)
            model_times.append(model_time)
            completed += 1
            if verbose:
                print("Model AUC: {:.4f}  [{}/{} done]".format(
                    model_auc, completed, n_models))

    else:
        # ============ Sequential Mode ============
        if verbose and n_gpus <= 1:
            print("Sequential mode: {} GPU(s), {} models".format(
                max(n_gpus, 0 if not torch.cuda.is_available() else 1), n_models))

        from .guide import _train_single_joint_guide

        for model_idx in range(n_models):
            model_start = time.time()
            seed = base_seed + model_idx * 100

            if verbose:
                print("\n" + "=" * 55)
                print("Model {}/{} (seed={}, GUIDE {})".format(
                    model_idx + 1, n_models, seed, mode_str))
                print("=" * 55)

            aug_data, _ = add_auxiliary_nodes(
                data, n_aux=n_aux_nodes,
                feature_method=feature_method,
                edge_method=edge_method,
                n_connections=n_connections,
                k_std=k_std,
                seed=seed
            )

            if verbose:
                n_aux_actual = aug_data.x.shape[0] - n_orig
                print("Augmented graph: {} nodes ({} orig + {} aux)".format(
                    aug_data.x.shape[0], n_orig, n_aux_actual))

            try:
                ae_score, _, combined_all = _train_single_joint_guide(
                    aug_data, epochs=epochs,
                    guide_hidden_a=guide_hidden_a,
                    guide_hidden_s=guide_hidden_s,
                    guide_num_layers=guide_num_layers,
                    guide_dropout=guide_dropout,
                    guide_alpha=guide_alpha,
                    cle_hidden=cle_hidden, device=device, seed=seed,
                    lamda1=lamda1, lamda2=lamda2,
                    normalize_loss=normalize_loss,
                    normalize_method=normalize_method,
                    normalize_scores=normalize_scores,
                    score_norm_method=score_norm_method,
                    use_embedding_transform=use_embedding_transform,
                    joint_training=joint_training, verbose=verbose,
                    use_adaptive_prior=use_adaptive_prior
                )
            except (RuntimeError, AttributeError) as e:
                if verbose:
                    print("Model {}/{} FAILED (seed={}): {} — skipping".format(
                        model_idx + 1, n_models, seed, e))
                continue

            combined_orig = combined_all[:n_orig]
            all_scores.append(combined_orig)
            model_time = time.time() - model_start
            model_m = compute_all_metrics(y_orig, combined_orig)
            model_aucs.append(model_m['auc'])
            model_times.append(model_time)

            if verbose:
                print("Model {}/{} AUC: {auc:.4f} AUPRC: {auprc:.4f} P@{k}: {precision_at_k:.4f} R@{k}: {recall_at_k:.4f} | Time: {:.1f}s".format(
                    model_idx + 1, n_models, model_time, **model_m))

    if len(all_scores) == 0:
        raise RuntimeError("All ensemble models failed — no valid scores")
    all_scores = np.array(all_scores)
    if agg_method == 'max':
        final_scores = np.max(all_scores, axis=0)
    elif agg_method == 'median':
        final_scores = np.median(all_scores, axis=0)
    else:
        final_scores = np.mean(all_scores, axis=0)

    ens_m = compute_all_metrics(y_orig, final_scores)
    model_aucs_arr = np.array(model_aucs)

    if verbose:
        model_auc_mean = float(np.mean(model_aucs_arr))
        model_auc_std = float(np.std(model_aucs_arr))
        model_auc_best = float(np.max(model_aucs_arr))
        print("\n" + "=" * 55)
        print("GUIDE Auxiliary Node Ensemble Complete")
        print("=" * 55)
        print("Model AUCs: mean={:.4f} +/- {:.4f}, best={:.4f}".format(
            model_auc_mean, model_auc_std, model_auc_best))
        print("Ensemble: AUC={auc:.4f} AUPRC={auprc:.4f} P@{k}={precision_at_k:.4f} R@{k}={recall_at_k:.4f}".format(**ens_m))
        total_time = time.time() - training_start
        print("Total time: {:.1f}s".format(total_time))

    return ens_m['auc'], final_scores, model_aucs_arr, np.array(model_times)


# ==================== GAD-NR Ensemble ====================

def _train_one_aux_worker_gadnr(task):
    """Train a single GAD-NR auxiliary node model on a specific GPU (for multiprocessing).

    Parallel to _train_one_aux_worker, _train_one_aux_worker_anomalydae, and
    _train_one_aux_worker_guide. Uses GAD-NR-specific params.

    Returns
    -------
    combined_orig : np.ndarray (n_orig,)
    model_auc : float
    """
    import gc, time
    t_start = time.time()
    gpu_id = task['gpu_id']
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor(task['x_np'], dtype=torch.float32, device=device)
    y = torch.tensor(task['y_np'], dtype=torch.long, device=device)
    edge_index = torch.tensor(task['edge_index_np'], dtype=torch.long, device=device)
    data = PyGData(x=x, y=y, edge_index=edge_index)

    seed = task['base_seed'] + task['model_idx'] * 100
    n_orig = task['n_orig']
    y_orig = task['y_orig_np']

    aug_data, _ = add_auxiliary_nodes(
        data, n_aux=task['n_aux_nodes'],
        feature_method=task['feature_method'],
        edge_method=task['edge_method'],
        n_connections=task['n_connections'],
        k_std=task['k_std'],
        seed=seed
    )

    from .gadnr import _train_single_joint_gadnr
    ae_score, _, combined_all = _train_single_joint_gadnr(
        aug_data,
        epochs=task['epochs'],
        gadnr_hidden=task['gadnr_hidden'],
        sample_size=task['sample_size'],
        encoder=task['encoder'],
        cle_hidden=task['cle_hidden'],
        device=device,
        seed=seed,
        lamda1=task['lamda1'],
        lamda2=task['lamda2'],
        normalize_loss=task['normalize_loss'],
        normalize_method=task['normalize_method'],
        normalize_scores=task['normalize_scores'],
        score_norm_method=task['score_norm_method'],
        use_embedding_transform=task['use_embedding_transform'],
        joint_training=task['joint_training'],
        verbose=False
    )

    combined_orig = combined_all[:n_orig]
    model_m = compute_all_metrics(y_orig, combined_orig)
    model_auc = model_m['auc']
    model_time = time.time() - t_start

    del aug_data, data, combined_all, ae_score
    torch.cuda.empty_cache()
    gc.collect()

    return combined_orig, model_auc, model_time


def train_auxiliary_node_ensemble_gadnr(data, n_models=10, n_aux_nodes=5,
                                         feature_method='outlier_tail',
                                         edge_method='random_connect',
                                         n_connections=5, k_std=3.0,
                                         epochs=100, gadnr_hidden=64, sample_size=10,
                                         encoder='GCN',
                                         cle_hidden=None,
                                         device=None, base_seed=42,
                                         lamda1=0.5, lamda2=0.5,
                                         normalize_loss=True,
                                         normalize_method='exponential_moving_average',
                                         normalize_scores=True,
                                         score_norm_method='min_max',
                                         use_embedding_transform=True,
                                         joint_training=True, verbose=True,
                                         agg_method='mean',
                                         lr_ae=5e-3,
                                         parallel=True,
                                   use_adaptive_prior=True):
    """Auxiliary node ensemble using GAD-NR base model.

    Parallel to train_auxiliary_node_ensemble,
    train_auxiliary_node_ensemble_anomalydae, and train_auxiliary_node_ensemble_guide.

    Uses GAD-NR-specific params: gadnr_hidden, sample_size, encoder.

    Parameters
    ----------
    data : PyGData
    n_models : int
    n_aux_nodes : int or float
    feature_method : str
    edge_method : str
    n_connections : int
    k_std : float
    epochs : int
    gadnr_hidden : int — hidden dimension
    sample_size : int — neighborhood sample size
    encoder : str — GNN encoder type ('GCN', 'GIN', 'GAT', 'SAGE')
    cle_hidden : list[int] | None
    device : torch.device | None
    base_seed : int
    lamda1 : float
    lamda2 : float
    normalize_loss : bool
    normalize_method : str
    normalize_scores : bool
    score_norm_method : str
    use_embedding_transform : bool
    joint_training : bool
    verbose : bool
    agg_method : str
    lr_ae : float (kept for API consistency, not used by GAD-NR)
    parallel : bool

    Returns
    -------
    ensemble_auc : float
    final_scores : np.ndarray (n_orig_nodes,)
    model_aucs : np.ndarray (n_models,)
    """
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_orig = data.x.shape[0]
    y_orig = data.y.bool().cpu().numpy()

    all_scores = []
    model_aucs = []
    model_times = []

    training_start = time.time()
    mode_str = "AE+CLE" if joint_training else "AE only"

    if verbose:
        print("=" * 55)
        if 0 < n_aux_nodes < 1:
            n_aux_actual = max(1, int(round(n_aux_nodes * n_orig)))
            aux_desc = "{:.1f}% ({} nodes)".format(n_aux_nodes * 100, n_aux_actual)
        else:
            aux_desc = "{} nodes".format(int(n_aux_nodes))
        print("GAD-NR Auxiliary Node Ensemble: {} models x {}, +{} aux per model".format(
            n_models, mode_str, aux_desc))
        print("Feature method: {} | Edge method: {} | n_conn: {} | k_std: {}".format(
            feature_method, edge_method, n_connections, k_std))
        print("gadnr: hidden={}, sample_size={}, encoder={}".format(
            gadnr_hidden, sample_size, encoder))
        print("=" * 55)

    # ---- Determine execution mode ----
    n_gpus = torch.cuda.device_count() if (parallel and torch.cuda.is_available()) else 0
    use_parallel = n_gpus > 1

    if use_parallel and n_models > 1:
        # ============ Multi-GPU Parallel Mode ============
        if verbose:
            print("Parallel mode: {} GPUs, {} models -> ~{:.0f} models/GPU".format(
                n_gpus, n_models, n_models / n_gpus))

        x_np = data.x.cpu().numpy().astype(np.float32)
        y_np = data.y.cpu().numpy().astype(np.int64)
        edge_np = data.edge_index.cpu().numpy().astype(np.int64)
        y_orig_np = y_orig.astype(np.float64)

        tasks = []
        for model_idx in range(n_models):
            tasks.append({
                'gpu_id': model_idx % n_gpus,
                'x_np': x_np, 'y_np': y_np, 'edge_index_np': edge_np,
                'y_orig_np': y_orig_np, 'n_orig': n_orig,
                'model_idx': model_idx, 'base_seed': base_seed,
                'n_aux_nodes': n_aux_nodes,
                'feature_method': feature_method,
                'edge_method': edge_method,
                'n_connections': n_connections,
                'k_std': k_std,
                'epochs': epochs,
                'gadnr_hidden': gadnr_hidden,
                'sample_size': sample_size,
                'encoder': encoder,
                'cle_hidden': cle_hidden,
                'lamda1': lamda1, 'lamda2': lamda2,
                'normalize_loss': normalize_loss,
                'normalize_method': normalize_method,
                'normalize_scores': normalize_scores,
                'score_norm_method': score_norm_method,
                'use_embedding_transform': use_embedding_transform,
                'use_adaptive_prior': use_adaptive_prior,
                'joint_training': joint_training,
            })

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_gpus) as pool:
            results = list(pool.imap_unordered(_train_one_aux_worker_gadnr, tasks, chunksize=1))

        completed = 0
        for combined_orig, model_auc, model_time in results:
            all_scores.append(combined_orig)
            model_aucs.append(model_auc)
            model_times.append(model_time)
            completed += 1
            if verbose:
                print("Model AUC: {:.4f}  [{}/{} done]".format(
                    model_auc, completed, n_models))

    else:
        # ============ Sequential Mode ============
        if verbose and n_gpus <= 1:
            print("Sequential mode: {} GPU(s), {} models".format(
                max(n_gpus, 0 if not torch.cuda.is_available() else 1), n_models))

        from .gadnr import _train_single_joint_gadnr

        for model_idx in range(n_models):
            model_start = time.time()
            seed = base_seed + model_idx * 100

            if verbose:
                print("\n" + "=" * 55)
                print("Model {}/{} (seed={}, GAD-NR {})".format(
                    model_idx + 1, n_models, seed, mode_str))
                print("=" * 55)

            aug_data, _ = add_auxiliary_nodes(
                data, n_aux=n_aux_nodes,
                feature_method=feature_method,
                edge_method=edge_method,
                n_connections=n_connections,
                k_std=k_std,
                seed=seed
            )

            if verbose:
                n_aux_actual = aug_data.x.shape[0] - n_orig
                print("Augmented graph: {} nodes ({} orig + {} aux)".format(
                    aug_data.x.shape[0], n_orig, n_aux_actual))

            ae_score, _, combined_all = _train_single_joint_gadnr(
                aug_data, epochs=epochs,
                gadnr_hidden=gadnr_hidden,
                sample_size=sample_size,
                encoder=encoder,
                cle_hidden=cle_hidden, device=device, seed=seed,
                lamda1=lamda1, lamda2=lamda2,
                normalize_loss=normalize_loss,
                normalize_method=normalize_method,
                normalize_scores=normalize_scores,
                score_norm_method=score_norm_method,
                use_embedding_transform=use_embedding_transform,
                joint_training=joint_training, verbose=verbose,
                    use_adaptive_prior=use_adaptive_prior
            )

            combined_orig = combined_all[:n_orig]
            all_scores.append(combined_orig)
            model_time = time.time() - model_start
            model_m = compute_all_metrics(y_orig, combined_orig)
            model_aucs.append(model_m['auc'])
            model_times.append(model_time)

            if verbose:
                print("Model {}/{} AUC: {auc:.4f} AUPRC: {auprc:.4f} P@{k}: {precision_at_k:.4f} R@{k}: {recall_at_k:.4f} | Time: {:.1f}s".format(
                    model_idx + 1, n_models, model_time, **model_m))

    if len(all_scores) == 0:
        raise RuntimeError("All ensemble models failed — no valid scores")
    all_scores = np.array(all_scores)
    if agg_method == 'max':
        final_scores = np.max(all_scores, axis=0)
    elif agg_method == 'median':
        final_scores = np.median(all_scores, axis=0)
    else:
        final_scores = np.mean(all_scores, axis=0)

    ens_m = compute_all_metrics(y_orig, final_scores)
    model_aucs_arr = np.array(model_aucs)

    if verbose:
        model_auc_mean = float(np.mean(model_aucs_arr))
        model_auc_std = float(np.std(model_aucs_arr))
        model_auc_best = float(np.max(model_aucs_arr))
        print("\n" + "=" * 55)
        print("GAD-NR Auxiliary Node Ensemble Complete")
        print("=" * 55)
        print("Model AUCs: mean={:.4f} +/- {:.4f}, best={:.4f}".format(
            model_auc_mean, model_auc_std, model_auc_best))
        print("Ensemble: AUC={auc:.4f} AUPRC={auprc:.4f} P@{k}={precision_at_k:.4f} R@{k}={recall_at_k:.4f}".format(**ens_m))
        total_time = time.time() - training_start
        print("Total time: {:.1f}s".format(total_time))

    return ens_m['auc'], final_scores, model_aucs_arr, np.array(model_times)


# ==================== DONE Ensemble ====================

def _train_one_aux_worker_done(task):
    """Train a single DONE auxiliary node model on a specific GPU (for multiprocessing).

    Uses DONE-specific params: done_hidden, done_num_layers, done_dropout.

    Returns
    -------
    combined_orig : np.ndarray (n_orig,)
    model_auc : float
    """
    import gc, time
    t_start = time.time()
    gpu_id = task['gpu_id']
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor(task['x_np'], dtype=torch.float32, device=device)
    y = torch.tensor(task['y_np'], dtype=torch.long, device=device)
    edge_index = torch.tensor(task['edge_index_np'], dtype=torch.long, device=device)
    data = PyGData(x=x, y=y, edge_index=edge_index)

    seed = task['base_seed'] + task['model_idx'] * 100
    n_orig = task['n_orig']
    y_orig = task['y_orig_np']

    aug_data, _ = add_auxiliary_nodes(
        data, n_aux=task['n_aux_nodes'],
        feature_method=task['feature_method'],
        edge_method=task['edge_method'],
        n_connections=task['n_connections'],
        k_std=task['k_std'],
        seed=seed
    )

    from .done import _train_single_joint_done
    ae_score, _, combined_all = _train_single_joint_done(
        aug_data,
        epochs=task['epochs'],
        done_hidden=task['done_hidden'],
        done_num_layers=task['done_num_layers'],
        done_dropout=task['done_dropout'],
        cle_hidden=task['cle_hidden'],
        device=device,
        seed=seed,
        lamda1=task['lamda1'],
        lamda2=task['lamda2'],
        normalize_loss=task['normalize_loss'],
        normalize_method=task['normalize_method'],
        normalize_scores=task['normalize_scores'],
        score_norm_method=task['score_norm_method'],
        use_embedding_transform=task['use_embedding_transform'],
        joint_training=task['joint_training'],
        verbose=False,
        use_adaptive_prior=task.get('use_adaptive_prior', False)
    )

    combined_orig = combined_all[:n_orig]
    model_m = compute_all_metrics(y_orig, combined_orig)
    model_auc = model_m['auc']
    model_time = time.time() - t_start

    del aug_data, data, combined_all, ae_score
    torch.cuda.empty_cache()
    gc.collect()

    return combined_orig, model_auc, model_time


def train_auxiliary_node_ensemble_done(data, n_models=10, n_aux_nodes=5,
                                        feature_method='outlier_tail',
                                        edge_method='random_connect',
                                        n_connections=5, k_std=3.0,
                                        epochs=100, done_hidden=64, done_num_layers=4,
                                        done_dropout=0.0,
                                        cle_hidden=None,
                                        device=None, base_seed=42,
                                        lamda1=0.5, lamda2=0.5,
                                        normalize_loss=True,
                                        normalize_method='exponential_moving_average',
                                        normalize_scores=True,
                                        score_norm_method='min_max',
                                        use_embedding_transform=True,
                                        joint_training=True, verbose=True,
                                        agg_method='mean',
                                        lr_ae=5e-3,
                                        parallel=True,
                                        use_adaptive_prior=False):
    """Auxiliary node ensemble using DONE base model.

    Uses DONE-specific params: done_hidden, done_num_layers, done_dropout.

    CLE uses standard Gaussian noise by default; set use_adaptive_prior=True
    to use the reference embedding covariance.

    Parameters
    ----------
    data : PyGData
    n_models : int
    n_aux_nodes : int or float
    feature_method : str
    edge_method : str
    n_connections : int
    k_std : float
    epochs : int
    done_hidden : int — hidden dimension
    done_num_layers : int — total layers
    done_dropout : float
    cle_hidden : list[int] | None
    [other standard params]

    Returns
    -------
    ensemble_auc : float
    final_scores : np.ndarray (n_orig_nodes,)
    model_aucs : np.ndarray (n_models,)
    """
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_orig = data.x.shape[0]
    y_orig = data.y.bool().cpu().numpy()

    all_scores = []
    model_aucs = []
    model_times = []

    training_start = time.time()
    mode_str = "AE+CLE" if joint_training else "AE only"

    if verbose:
        print("=" * 55)
        if 0 < n_aux_nodes < 1:
            n_aux_actual = max(1, int(round(n_aux_nodes * n_orig)))
            aux_desc = "{:.1f}% ({} nodes)".format(n_aux_nodes * 100, n_aux_actual)
        else:
            aux_desc = "{} nodes".format(int(n_aux_nodes))
        print("DONE Auxiliary Node Ensemble: {} models x {}, +{} aux per model".format(
            n_models, mode_str, aux_desc))
        print("Feature method: {} | Edge method: {} | n_conn: {} | k_std: {}".format(
            feature_method, edge_method, n_connections, k_std))
        print("done: hidden={}, layers={}, dropout={}".format(
            done_hidden, done_num_layers, done_dropout))
        print("=" * 55)

    n_gpus = torch.cuda.device_count() if (parallel and torch.cuda.is_available()) else 0
    use_parallel = n_gpus > 1

    if use_parallel and n_models > 1:
        if verbose:
            print("Parallel mode: {} GPUs, {} models -> ~{:.0f} models/GPU".format(
                n_gpus, n_models, n_models / n_gpus))

        x_np = data.x.cpu().numpy().astype(np.float32)
        y_np = data.y.cpu().numpy().astype(np.int64)
        edge_np = data.edge_index.cpu().numpy().astype(np.int64)
        y_orig_np = y_orig.astype(np.float64)

        tasks = []
        for model_idx in range(n_models):
            tasks.append({
                'gpu_id': model_idx % n_gpus,
                'x_np': x_np, 'y_np': y_np, 'edge_index_np': edge_np,
                'y_orig_np': y_orig_np, 'n_orig': n_orig,
                'model_idx': model_idx, 'base_seed': base_seed,
                'n_aux_nodes': n_aux_nodes,
                'feature_method': feature_method,
                'edge_method': edge_method,
                'n_connections': n_connections,
                'k_std': k_std,
                'epochs': epochs,
                'done_hidden': done_hidden,
                'done_num_layers': done_num_layers,
                'done_dropout': done_dropout,
                'cle_hidden': cle_hidden,
                'lamda1': lamda1, 'lamda2': lamda2,
                'normalize_loss': normalize_loss,
                'normalize_method': normalize_method,
                'normalize_scores': normalize_scores,
                'score_norm_method': score_norm_method,
                'use_embedding_transform': use_embedding_transform,
                'use_adaptive_prior': use_adaptive_prior,
                'joint_training': joint_training,
            })

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_gpus) as pool:
            results = list(pool.imap_unordered(_train_one_aux_worker_done, tasks, chunksize=1))

        completed = 0
        for combined_orig, model_auc, model_time in results:
            all_scores.append(combined_orig)
            model_aucs.append(model_auc)
            model_times.append(model_time)
            completed += 1
            if verbose:
                print("Model AUC: {:.4f}  [{}/{} done]".format(
                    model_auc, completed, n_models))

    else:
        if verbose and n_gpus <= 1:
            print("Sequential mode: {} GPU(s), {} models".format(
                max(n_gpus, 0 if not torch.cuda.is_available() else 1), n_models))

        from .done import _train_single_joint_done

        for model_idx in range(n_models):
            model_start = time.time()
            seed = base_seed + model_idx * 100

            if verbose:
                print("\n" + "=" * 55)
                print("Model {}/{} (seed={}, DONE {})".format(
                    model_idx + 1, n_models, seed, mode_str))
                print("=" * 55)

            aug_data, _ = add_auxiliary_nodes(
                data, n_aux=n_aux_nodes,
                feature_method=feature_method,
                edge_method=edge_method,
                n_connections=n_connections,
                k_std=k_std,
                seed=seed
            )

            if verbose:
                n_aux_actual = aug_data.x.shape[0] - n_orig
                print("Augmented graph: {} nodes ({} orig + {} aux)".format(
                    aug_data.x.shape[0], n_orig, n_aux_actual))

            ae_score, _, combined_all = _train_single_joint_done(
                aug_data, epochs=epochs,
                done_hidden=done_hidden,
                done_num_layers=done_num_layers,
                done_dropout=done_dropout,
                cle_hidden=cle_hidden, device=device, seed=seed,
                lamda1=lamda1, lamda2=lamda2,
                normalize_loss=normalize_loss,
                normalize_method=normalize_method,
                normalize_scores=normalize_scores,
                score_norm_method=score_norm_method,
                use_embedding_transform=use_embedding_transform,
                joint_training=joint_training, verbose=verbose,
                use_adaptive_prior=use_adaptive_prior
            )

            combined_orig = combined_all[:n_orig]
            all_scores.append(combined_orig)
            model_time = time.time() - model_start
            model_m = compute_all_metrics(y_orig, combined_orig)
            model_aucs.append(model_m['auc'])
            model_times.append(model_time)

            if verbose:
                print("Model {}/{} AUC: {auc:.4f} AUPRC: {auprc:.4f} P@{k}: {precision_at_k:.4f} R@{k}: {recall_at_k:.4f} | Time: {:.1f}s".format(
                    model_idx + 1, n_models, model_time, **model_m))

    if len(all_scores) == 0:
        raise RuntimeError("All ensemble models failed — no valid scores")
    all_scores = np.array(all_scores)
    if agg_method == 'max':
        final_scores = np.max(all_scores, axis=0)
    elif agg_method == 'median':
        final_scores = np.median(all_scores, axis=0)
    else:
        final_scores = np.mean(all_scores, axis=0)

    ens_m = compute_all_metrics(y_orig, final_scores)
    model_aucs_arr = np.array(model_aucs)

    if verbose:
        model_auc_mean = float(np.mean(model_aucs_arr))
        model_auc_std = float(np.std(model_aucs_arr))
        model_auc_best = float(np.max(model_aucs_arr))
        print("\n" + "=" * 55)
        print("DONE Auxiliary Node Ensemble Complete")
        print("=" * 55)
        print("Model AUCs: mean={:.4f} +/- {:.4f}, best={:.4f}".format(
            model_auc_mean, model_auc_std, model_auc_best))
        print("Ensemble: AUC={auc:.4f} AUPRC={auprc:.4f} P@{k}={precision_at_k:.4f} R@{k}={recall_at_k:.4f}".format(**ens_m))
        total_time = time.time() - training_start
        print("Total time: {:.1f}s".format(total_time))

    return ens_m['auc'], final_scores, model_aucs_arr, np.array(model_times)
