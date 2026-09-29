"""DONE model and training functions for DONE+CLE joint training.

DONE (Deep Outlier aware Attributed Network Embedding):
Dual autoencoder with separate attribute and structure branches, plus
neighborhood dissimilarity for outlier awareness.

Architecture:
- Attr encoder (MLP): x → h_a
- Struct encoder (MLP): s → h_s  (s = dense adjacency)
- Attr decoder: h_a → x_
- Struct decoder: h_s → s_
- NeighDiff: neighbor dissimilarity scores dna, dns
- Anomaly score = (oa + os + oc) / 3

Embedding for CLE = h_a (attribute encoder output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime, timedelta
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import MLP as DONE_MLP

from .cle import CLERegression, LinearFlowNoise, MLP as CLEMLP
from .utils import (_normalize_vector, _center_cols, _normalize_cols,
    _procrustes_align, _sign_fix, _align_embedding, _compute_combined_score,
    compute_all_metrics, LossNormalizer, format_time_precise, format_timedelta_precise)


# ==================== DONE Model Components ====================

class NeighDiff(MessagePassing):
    """Per-node neighbor dissimilarity via message passing."""
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, h, edge_index):
        return self.propagate(edge_index, h=h)

    def message(self, h_i, h_j, edge_index):
        return torch.sum(torch.pow(h_i - h_j, 2), dim=1, keepdim=True)


class DONE_Base(nn.Module):
    """DONE autoencoder with dual attribute/structure branches.

    Forward returns: (x_, s_, h_a, h_s, dna, dns)
    Anomaly score = (oa + os + oc) / 3
    Embedding for CLE = h_a
    """

    def __init__(self, x_dim, s_dim, hid_dim, num_layers, dropout, act):
        super(DONE_Base, self).__init__()

        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers

        self.attr_encoder = DONE_MLP(in_channels=x_dim,
                                      hidden_channels=hid_dim,
                                      out_channels=hid_dim,
                                      num_layers=encoder_layers,
                                      dropout=dropout,
                                      act=act)

        self.attr_decoder = DONE_MLP(in_channels=hid_dim,
                                      hidden_channels=hid_dim,
                                      out_channels=x_dim,
                                      num_layers=decoder_layers,
                                      dropout=dropout,
                                      act=act)

        self.struct_encoder = DONE_MLP(in_channels=s_dim,
                                        hidden_channels=hid_dim,
                                        out_channels=hid_dim,
                                        num_layers=encoder_layers,
                                        dropout=dropout,
                                        act=act)

        self.struct_decoder = DONE_MLP(in_channels=hid_dim,
                                        hidden_channels=hid_dim,
                                        out_channels=s_dim,
                                        num_layers=decoder_layers,
                                        dropout=dropout,
                                        act=act)

        self.neigh_diff = NeighDiff()

    def forward(self, x, s, edge_index):
        h_a = self.attr_encoder(x)
        x_ = self.attr_decoder(h_a)
        dna = self.neigh_diff(h_a, edge_index).squeeze()
        h_s = self.struct_encoder(s)
        s_ = self.struct_decoder(h_s)
        dns = self.neigh_diff(h_s, edge_index).squeeze()
        return x_, s_, h_a, h_s, dna, dns

    def loss_func(self, x, x_, s, s_, h_a, h_s, dna, dns):
        """Compute DONE multi-component loss.

        Returns
        -------
        score : torch.Tensor (n_nodes,) — per-node anomaly score
        loss : torch.Tensor scalar — mean training loss
        """
        # Attribute outlier score
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        tmp = 0.2 * dx + 0.2 * dna
        oa = tmp / torch.sum(tmp)

        # Structure outlier score
        ds = torch.sum(torch.pow(s - s_, 2), 1)
        tmp = 0.2 * ds + 0.2 * dns
        os = tmp / torch.sum(tmp)

        # Cross-modal consistency score
        dc = torch.sum(torch.pow(h_a - h_s, 2), 1)
        oc = dc / torch.sum(dc)

        # Loss components
        loss_prox_a = torch.mean(torch.log(torch.pow(oa, -1)) * dx)
        loss_hom_a = torch.mean(torch.log(torch.pow(oa, -1)) * dna)
        loss_prox_s = torch.mean(torch.log(torch.pow(os, -1)) * ds)
        loss_hom_s = torch.mean(torch.log(torch.pow(os, -1)) * dns)
        loss_c = torch.mean(torch.log(torch.pow(oc, -1)) * dc)

        loss = 0.2 * loss_prox_a + 0.2 * loss_hom_a + 0.2 * loss_prox_s + 0.2 * loss_hom_s + 0.2 * loss_c
        score = (oa + os + oc) / 3
        return score, loss


# ==================== Training Functions ====================

def _train_single_joint_done(data, epochs=100, done_hidden=64, done_num_layers=4,
                              done_dropout=0.0, cle_hidden=None, device=None, seed=42,
                              lamda1=0.5, lamda2=0.5,
                              normalize_loss=True, normalize_method='exponential_moving_average',
                              normalize_scores=True, score_norm_method='min_max',
                              use_embedding_transform=True, joint_training=True, verbose=True,
                              use_adaptive_prior=False):
    """Train a single DONE(+CLE) model on given graph.

    DONE uses dense adjacency s = to_dense_adj(edge_index) as structural input.
    Embedding for CLE = h_a (attribute encoder output).

    Parameters
    ----------
    data : PyG Data
    epochs : int
    done_hidden : int — hidden dimension
    done_num_layers : int — total layers (split between encoder/decoder)
    done_dropout : float
    cle_hidden : list[int] | None
    device : torch.device | None
    seed : int
    lamda1 : float
    lamda2 : float
    normalize_loss : bool
    normalize_method : str
    normalize_scores : bool
    score_norm_method : str
    use_embedding_transform : bool
    joint_training : bool
    verbose : bool
    use_adaptive_prior : bool
        Use embedding covariance for CLE noise; default False uses N(0, I).

    Returns
    -------
    ae_score : np.ndarray (n_nodes,)
    cle_score : np.ndarray (n_nodes,) or None
    combined_score : np.ndarray (n_nodes,)
    """
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Dense adjacency as structure input
    s = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0].to(device)

    # ---- DONE model ----
    ae_model = DONE_Base(
        x_dim=x.size(1),
        s_dim=s.size(1),
        hid_dim=done_hidden,
        num_layers=done_num_layers,
        dropout=done_dropout,
        act=F.leaky_relu
    ).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)

    # ---- CLE model (only if joint training) ----
    cle_model = None
    cle_optimizer = None
    emb_ref = None
    loss_normalizer = None

    if joint_training:
        cle_model = CLERegression(
            hidden_size=cle_hidden, epochs=300, batch_size=64,
            lr=1e-4, weight_decay=5e-4, T=400, num_bins=1, device=device
        )

        # Reference embedding: h_a
        ae_model.eval()
        with torch.no_grad():
            x0, s0, h_a0, h_s0, dna0, dns0 = ae_model(x, s, edge_index)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(h_a0))
            else:
                emb_ref = h_a0
        ae_model.train()

        if use_adaptive_prior:
            flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
            flow.fit(emb_ref)
            cle_model.noise_flow = flow.eval()

        if normalize_loss:
            loss_normalizer = LossNormalizer(method=normalize_method)

    # ---- Training loop ----
    for epoch in range(epochs):
        ae_model.train()
        if joint_training and cle_model.model is not None:
            cle_model.model.train()

        ae_optimizer.zero_grad()
        if cle_optimizer is not None:
            cle_optimizer.zero_grad()

        # AE forward
        x_, s_, h_a, h_s, dna, dns = ae_model(x, s, edge_index)
        ae_score, ae_loss = ae_model.loss_func(x, x_, s, s_, h_a, h_s, dna, dns)

        if joint_training:
            # Align embedding (h_a)
            if use_embedding_transform:
                emb_n = _normalize_cols(_center_cols(h_a))
                emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                emb_aligned = _sign_fix(emb_aligned, emb_ref)
            else:
                emb_aligned = h_a

            if cle_model.model is None:
                cle_model.model = CLEMLP(
                    [emb_aligned.shape[-1]] + cle_hidden,
                    num_bins=cle_model.num_bins
                ).to(device)
                cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)

            t = torch.randint(0, cle_model.T, (emb_aligned.shape[0],), device=device).long()
            cle_loss = cle_model.compute_loss(emb_aligned.detach(), t)

            if normalize_loss:
                ae_norm, cle_norm = loss_normalizer.normalize(ae_loss, cle_loss)
                joint_loss = ae_norm + lamda1 * cle_norm
            else:
                joint_loss = ae_loss + lamda1 * cle_loss

            joint_loss.backward()
            ae_optimizer.step()
            if cle_optimizer is not None:
                cle_optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | DONE: {:.5f} | CLE: {:.5f} | Joint: {:.5f}".format(
                    epoch, ae_loss.item(), cle_loss.item(), joint_loss.item()))
        else:
            ae_loss.backward()
            ae_optimizer.step()

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print("  Epoch {:04d} | DONE Loss: {:.5f}".format(epoch, ae_loss.item()))

    # ---- Get scores ----
    ae_model.eval()
    with torch.no_grad():
        x_, s_, h_a, h_s, dna, dns = ae_model(x, s, edge_index)
        ae_score, _ = ae_model.loss_func(x, x_, s, s_, h_a, h_s, dna, dns)
        ae_score = ae_score.detach().cpu().numpy()

        if joint_training:
            cle_model.model.eval()
            if use_embedding_transform:
                emb_eval_aligned = _align_embedding(h_a, emb_ref)
            else:
                emb_eval_aligned = h_a
            cle_score = cle_model.predict_score(emb_eval_aligned)

            if normalize_scores:
                ae_score_n = _normalize_vector(ae_score, method=score_norm_method)
                cle_score_n = _normalize_vector(cle_score, method=score_norm_method)
            else:
                ae_score_n = ae_score
                cle_score_n = cle_score

            combined_score = 1.0 * ae_score_n + lamda2 * cle_score_n
        else:
            cle_score = None
            if normalize_scores:
                combined_score = _normalize_vector(ae_score, method=score_norm_method)
            else:
                combined_score = ae_score

    return ae_score, cle_score, combined_score


def train_joint_done_cle(data, epochs=100, done_hidden=64, done_num_layers=4,
                          done_dropout=0.0, cle_hidden=None, batch_size=64, device=None,
                          normalize_loss=True, normalize_method='exponential_moving_average',
                          lamda1=0.5, lamda2=0.5, normalize_scores=True,
                          score_norm_method='min_max', joint_training=True,
                          dataset_name='unknown', use_embedding_transform=True,
                          use_adaptive_prior=False):
    """Joint training of DONE + CLE models.

    Parameters
    ----------
    data : PyG Data
    epochs : int
    done_hidden : int — hidden dimension
    done_num_layers : int — total layers
    done_dropout : float
    cle_hidden : list[int]
    batch_size : int
    device : torch.device | None
    [standard params same as other models]
    use_adaptive_prior : bool
        Use embedding covariance for CLE noise; default False uses N(0, I).

    Returns
    -------
    If joint_training:
        (ae_model, cle_model, combined_metrics_dict)
    Else:
        (ae_model, ae_metrics_dict)
    """
    if cle_hidden is None:
        cle_hidden = [256, 512, 256]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    print("Using device:", device)
    print("Loading dataset: {}".format(dataset_name))
    print("Dataset info: {} nodes, {} features, {} anomalies".format(
        data.x.shape[0], data.x.shape[1], data.y.sum().item()))
    print("Base model: DONE")

    training_start_time = time.time()
    print("Training started at: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    x = data.x.to(device)
    y = data.y.bool()
    edge_index = data.edge_index.to(device)
    s = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0].to(device)

    # Initialize DONE model
    ae_model = DONE_Base(
        x_dim=x.size(1),
        s_dim=s.size(1),
        hid_dim=done_hidden,
        num_layers=done_num_layers,
        dropout=done_dropout,
        act=F.leaky_relu
    ).to(device)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-3)

    cle_model = None
    cle_optimizer = None
    emb_ref = None
    loss_normalizer = None

    if joint_training:
        cle_model = CLERegression(
            hidden_size=cle_hidden, epochs=300, batch_size=batch_size,
            lr=1e-4, weight_decay=5e-4, T=400, num_bins=1, device=device
        )
        loss_normalizer = LossNormalizer(method=normalize_method) if normalize_loss else None

    print("\n" + "=" * 60)
    if joint_training:
        print("Phase 1: Joint Training DONE + CLE (Unsupervised)")
        print("DONE:1.0, CLE:lamda1={}".format(lamda1))
    else:
        print("Phase 1: Training DONE Only (Unsupervised)")
    if normalize_loss and joint_training:
        print("Loss Normalization: {}".format(normalize_method))
    print("=" * 60)

    # Reference embedding
    if joint_training:
        ae_model.eval()
        with torch.no_grad():
            x0, s0, h_a0, h_s0, dna0, dns0 = ae_model(x, s, edge_index)
            if use_embedding_transform:
                emb_ref = _normalize_cols(_center_cols(h_a0))
            else:
                emb_ref = h_a0
        ae_model.train()

        if use_adaptive_prior:
            flow = LinearFlowNoise(dim=emb_ref.shape[1], ridge=1e-3, device=device, dtype=emb_ref.dtype)
            flow.fit(emb_ref)
            cle_model.noise_flow = flow.eval()

    # Training loop
    epoch_times = []
    for epoch in range(epochs):
        epoch_start_time = time.time()

        ae_model.train()
        ae_optimizer.zero_grad()

        if joint_training:
            if cle_model.model is not None:
                cle_model.model.train()
            if cle_model.model is not None and cle_optimizer is None:
                cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)
            if cle_optimizer is not None:
                cle_optimizer.zero_grad()

        x_, s_, h_a, h_s, dna, dns = ae_model(x, s, edge_index)
        ae_score, ae_loss = ae_model.loss_func(x, x_, s, s_, h_a, h_s, dna, dns)

        if joint_training:
            if use_embedding_transform:
                emb_n = _normalize_cols(_center_cols(h_a))
                emb_aligned, _ = _procrustes_align(emb_n, emb_ref)
                emb_aligned = _sign_fix(emb_aligned, emb_ref)
            else:
                emb_aligned = h_a

            if cle_model.model is None:
                cle_model.model = CLEMLP(
                    [emb_aligned.shape[-1]] + cle_hidden,
                    num_bins=cle_model.num_bins
                ).to(device)
                cle_optimizer = Adam(cle_model.model.parameters(), lr=1e-4, weight_decay=5e-4)

            t = torch.randint(0, cle_model.T, (emb_aligned.shape[0],), device=device).long()
            cle_loss = cle_model.compute_loss(emb_aligned.detach(), t)

            if normalize_loss:
                ae_norm, cle_norm = loss_normalizer.normalize(ae_loss, cle_loss)
                joint_loss = ae_norm + lamda1 * cle_norm
            else:
                joint_loss = ae_loss + lamda1 * cle_loss

            joint_loss.backward()
            ae_optimizer.step()
            if cle_optimizer is not None:
                cle_optimizer.step()

            if epoch % 20 == 0 or epoch == epochs - 1:
                print("Epoch: {:04d} | DONE Loss: {:.5f} | CLE Loss: {:.5f} | Joint Loss: {:.5f}".format(
                    epoch, ae_loss.item(), cle_loss.item(), joint_loss.item()))
        else:
            ae_loss.backward()
            ae_optimizer.step()
            if epoch % 20 == 0 or epoch == epochs - 1:
                print("Epoch: {:04d} | DONE Loss: {:.5f}".format(epoch, ae_loss.item()))

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        if epoch % 20 == 0 or epoch == epochs - 1:
            ae_model.eval()
            with torch.no_grad():
                x_eval, s_eval, h_a_eval, h_s_eval, dna_eval, dns_eval = ae_model(x, s, edge_index)
                ae_score, _ = ae_model.loss_func(x, x_eval, s, s_eval, h_a_eval, h_s_eval, dna_eval, dns_eval)
                ae_score = ae_score.detach().cpu().numpy()

                if normalize_scores:
                    ae_score_disp = _normalize_vector(ae_score, method=score_norm_method)
                else:
                    ae_score_disp = ae_score

                y_np = y.cpu().numpy()
                ae_metrics = compute_all_metrics(y_np, ae_score_disp)

                if joint_training:
                    cle_model.model.eval()
                    if use_embedding_transform:
                        emb_eval_aligned = _align_embedding(h_a_eval, emb_ref)
                    else:
                        emb_eval_aligned = h_a_eval
                    cle_score_vec = cle_model.predict_score(emb_eval_aligned)
                    combined_score_vec = _compute_combined_score(
                        ae_score_disp, cle_score_vec, normalize_scores, score_norm_method, lamda2)

                    cle_metrics = compute_all_metrics(y_np, cle_score_vec)
                    combined_metrics = compute_all_metrics(y_np, combined_score_vec)

                    print("  -> DONE:     AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        ae_metrics['auc'], ae_metrics['auprc'],
                        ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
                    print("  -> CLE:      AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        cle_metrics['auc'], cle_metrics['auprc'],
                        cle_metrics['recall_at_k'], cle_metrics['precision_at_k']))
                    print("  -> Combined: AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        combined_metrics['auc'], combined_metrics['auprc'],
                        combined_metrics['recall_at_k'], combined_metrics['precision_at_k']))
                else:
                    print("  -> DONE: AUC:{:.4f} AUPRC:{:.4f} R@K:{:.4f} P@K:{:.4f}".format(
                        ae_metrics['auc'], ae_metrics['auprc'],
                        ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))

                elapsed_time = time.time() - training_start_time
                avg_epoch_time = np.mean(epoch_times[-20:]) if epoch_times else 0
                eta_seconds = avg_epoch_time * (epochs - epoch - 1)
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                print("  -> Time: {}/epoch (avg), Elapsed: {}, ETA: {}".format(
                    format_time_precise(avg_epoch_time),
                    format_time_precise(elapsed_time), eta_str))

    # Final evaluation
    print("\n" + "=" * 60)
    print("Phase 2: Final Evaluation")
    print("=" * 60)

    ae_model.eval()
    with torch.no_grad():
        x_final, s_final, h_a_final, h_s_final, dna_final, dns_final = ae_model(x, s, edge_index)
        ae_score, _ = ae_model.loss_func(x, x_final, s, s_final, h_a_final, h_s_final, dna_final, dns_final)
        ae_score = ae_score.detach().cpu().numpy()

        if normalize_scores:
            ae_score = _normalize_vector(ae_score, method=score_norm_method)

        y_np = y.cpu().numpy()

        if joint_training:
            if cle_model.model is None:
                raise ValueError("CLE model was not properly initialized during training")
            cle_model.model.eval()

            if use_embedding_transform:
                emb_eval_aligned = _align_embedding(h_a_final, emb_ref)
            else:
                emb_eval_aligned = h_a_final
            cle_score_vec = cle_model.predict_score(emb_eval_aligned)
            combined_score_vec = _compute_combined_score(
                ae_score, cle_score_vec, normalize_scores, score_norm_method, lamda2)

            ae_metrics = compute_all_metrics(y_np, ae_score)
            cle_metrics = compute_all_metrics(y_np, cle_score_vec)
            combined_metrics = compute_all_metrics(y_np, combined_score_vec)

            print("\nFinal Results:")
            print("  DONE Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                ae_metrics['auc'], ae_metrics['auprc'],
                ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
            print("  CLE Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                cle_metrics['auc'], cle_metrics['auprc'],
                cle_metrics['recall_at_k'], cle_metrics['precision_at_k']))
            print("  Combined Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                combined_metrics['auc'], combined_metrics['auprc'],
                combined_metrics['recall_at_k'], combined_metrics['precision_at_k']))
            print("  (K = {} anomalies)".format(int(y_np.sum())))

            return ae_model, cle_model, combined_metrics
        else:
            ae_metrics = compute_all_metrics(y_np, ae_score)
            print("\nFinal Results:")
            print("  DONE Model:")
            print("    AUC: {:.6f} | AUPRC: {:.6f} | Recall@K: {:.6f} | Precision@K: {:.6f}".format(
                ae_metrics['auc'], ae_metrics['auprc'],
                ae_metrics['recall_at_k'], ae_metrics['precision_at_k']))
            print("  (K = {} anomalies)".format(int(y_np.sum())))

            return ae_model, ae_metrics
