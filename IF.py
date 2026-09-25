from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn.functional as F

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from .BaseOOD import BaseOODAlg
from collections import OrderedDict


@register.ood_alg_register
class IF(BaseOODAlg):


    def __init__(self, config: Union[CommonArgs, Munch]):
        super(IF, self).__init__(config)
        self.targets = None
        self.node_repr = None
        self.att = None
        self.hard_node_mask = None
        self.soft_node_mask = None
        self.hard_edge_mask = None
        self.soft_edge_mask = None
        self.node_logit_rationale = None

        self.coef_task = config.ood.extra_param[0]
        self.coef_conn = config.ood.extra_param[1]
        
        self.coef_if = config.ood.ood_param

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions.

        """
        self.soft_edge_mask, self.hard_edge_mask, self.logit_c, self.logit_s, self.logit_full = model_output
        return self.logit_c


    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func: Accuracy}
                                   })


        Returns (Tensor):
            cross entropy loss

        """
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * self.coef_task * mask
        self.targets = targets

        return loss

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss based on GSAT algorithm

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   dataset: {num_envs: int(10)},
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns (Tensor):
            loss based on DIR algorithm

        """
        # if config.dataset.dataset_name == 'GOODHIV' and getattr(data, 'domain_id') is not None:
        #     data.env_id = data.domain_id

        # node_mask = self.node_mask * 0.99 + 0.005
        # mask_ent = - node_mask * torch.log(node_mask) - (1 - node_mask) * torch.log(1 - node_mask)

        self.spec_loss = OrderedDict()

        # influence ranking loss
        model = kwargs.get('model', None)
        if model is not None:
            self.spec_loss['IF'] = self.coef_if * stable_influence_single_loss(
                model=model,
                logit_c=self.logit_c,
                logit_s=self.logit_s,
                logit_full=self.logit_full,
                targets=self.targets,
                margin=0.2,
                hard_ratio=0.1,
                coef_task=self.coef_task,
                mask = mask
            )

        self.spec_loss['FULL'] = (config.metric.loss_func(self.logit_full, self.targets, reduction='none') * self.coef_task * mask).sum()/mask.sum()
        self.spec_loss['CONN'] = self.coef_conn * get_conn_loss(self.soft_edge_mask)
        # self.spec_loss['SIZE'] = self.coef_size * get_size_loss(self.soft_edge_mask)

        self.mean_loss = loss.sum()/mask.sum()
        loss = self.mean_loss + sum(self.spec_loss.values())
        return loss
    
def stable_influence_single_loss(
    model,
    logit_c,
    logit_s,
    logit_full,
    targets,
    margin,
    coef_task,
    mask,
    eps=1e-8,
    **kwargs
):
    """
    Stable Influence Ranking Loss (single-pair variant) with no grad calculating.
    """
    loss_full = F.cross_entropy(logit_full, targets, reduction="none") * coef_task * mask
    loss_c    = F.cross_entropy(logit_c,    targets, reduction="none") * coef_task * mask
    loss_s    = F.cross_entropy(logit_s,    targets, reduction="none") * coef_task * mask
    
    # hard-sample weight, detached
    weight = loss_full.detach()
    weight = weight / (weight.mean() + eps)
    #clipping for stability
    weight = weight.clamp(0.5, 2.0)
    # causal subgraph should outperform spurious subgraph on hard samples
    loss_hip = weight * F.relu(margin + loss_c - loss_s)

    return loss_hip.mean()
        

def stable_influence_ranking_loss(
    model,
    logit_c,
    logit_s,
    logit_full,
    targets,
    margin=0.1, # m ~= [0.2,0.5]
    hard_ratio=0.3,
    grad_norm_eps: float = 1e-6,           # skip pair if any grad norm below this
    **kwargs
):
    """
    Stable Influence Ranking Loss (semi-prototype variant).
 
    Encourage causal subgraph Gc to have stronger positive influence
    on hard samples than spurious subgraph Gs.
 
    C(a, b) = cosine(grad_encoder loss(a), grad_encoder loss(b))
 
    Subgraph end is AGGREGATED (mean over batch -> 1 backward each for c, s).
    Hard end stays PER-SAMPLE (H backwards). Total: 2 + H backward passes.
    """
    device = targets.device
    B = targets.size(0)
 
    # 1) encoder parameters ONLY (NOT predictor / head)
    encoder_params = [p for p in model.predictor.parameters() if p.requires_grad]
    if len(encoder_params) == 0:
        return torch.zeros((), device=device)
 
    # 2) per-sample CE losses (no reduction)
    loss_full = F.cross_entropy(logit_full, targets, reduction="none")
    loss_c    = F.cross_entropy(logit_c,    targets, reduction="none")
    loss_s    = F.cross_entropy(logit_s,    targets, reduction="none")
 
    # 3) hard-sample selection by full-graph loss
    num_hard = max(1, int(hard_ratio * B))
    hard_idx = torch.topk(loss_full.detach(), k=num_hard).indices  # [H]
    hard_loss_detached = loss_full[hard_idx].detach()
    hard_weight = hard_loss_detached / (hard_loss_detached.sum() + 1e-8)  # [H]
 
    # 4) AGGREGATED subgraph gradients (1 backward each, vs per-sample loop in v1)
    #    By linearity of gradient: grad(mean(loss_c)) == mean(per-sample grad_c).
    #    create_graph=True so SIR loss can propagate back into the mask predictor.
    def _flat(grads):
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, encoder_params)
        ])
 
    gc_bar_raw = torch.autograd.grad(
        loss_c.mean(), encoder_params,
        retain_graph=True, create_graph=True, allow_unused=True,
    )
    gc_bar = _flat(gc_bar_raw)
 
    gs_bar_raw = torch.autograd.grad(
        loss_s.mean(), encoder_params,
        retain_graph=True, create_graph=True, allow_unused=True,
    )
    gs_bar = _flat(gs_bar_raw)
 
    # numerical guard on aggregated subgraph gradients
    if gc_bar.norm() < grad_norm_eps or gs_bar.norm() < grad_norm_eps:
        return torch.zeros((), device=device)
 
    # 5) per-sample hard gradients (no create_graph -- hard samples are reference points)
    grads_h_dict = {}
    for j in hard_idx.tolist():
        gs_h = torch.autograd.grad(
            loss_full[j],
            encoder_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        grads_h_dict[j] = _flat(gs_h).detach()
 
    # 6) ranking loss: aggregated c/s vs per-sample hard
    total_loss = torch.zeros((), device=device)
    n_active = 0
 
    for w_j, j in zip(hard_weight, hard_idx.tolist()):
        gh_j = grads_h_dict[j]
        if gh_j.norm() < grad_norm_eps:
            continue
 
        c_score = F.cosine_similarity(gc_bar.unsqueeze(0), gh_j.unsqueeze(0),
                                      dim=-1, eps=1e-8).squeeze()
        s_score = F.cosine_similarity(gs_bar.unsqueeze(0), gh_j.unsqueeze(0),
                                      dim=-1, eps=1e-8).squeeze()
 
        pair_loss = F.relu(margin - c_score + s_score)
        total_loss = total_loss + w_j * pair_loss
        n_active += 1
 
    if n_active == 0:
        return torch.zeros((), device=device)
 
    # hard_weight already sums to 1, so total_loss is in [0, 2]. No /B needed.
    return total_loss


def stable_influence_ranking_loss_by_class(
    model,
    logit_c,
    logit_s,
    logit_full,
    targets,
    margin=0.1,  # m ~= [0.2, 0.5]
    hard_ratio=0.3,
    grad_norm_eps: float = 1e-6,
    **kwargs
):
    """
    Stable Influence Ranking Loss — class-stratified prototype variant.
 
    Same per-pair semantics as before, but the subgraph end is aggregated
    PER CLASS (instead of per batch). Different classes' causal gradients
    no longer cancel each other when batch is mixed.
 
    Total backward passes:  2 * |classes_in_batch|  +  H
    For binary tasks: 4 + H. Cheap.
    """
    device = targets.device
    B = targets.size(0)
 
    # 1) encoder parameters ONLY (NOT predictor / head)
    encoder_params = [p for p in model.predictor.parameters() if p.requires_grad]
    if len(encoder_params) == 0:
        return torch.zeros((), device=device)
 
    # 2) per-sample CE losses (no reduction)
    loss_full = F.cross_entropy(logit_full, targets, reduction="none")
    loss_c    = F.cross_entropy(logit_c,    targets, reduction="none")
    loss_s    = F.cross_entropy(logit_s,    targets, reduction="none")
 
    # 3) hard-sample selection by full-graph loss
    num_hard = max(1, int(hard_ratio * B))
    hard_idx = torch.topk(loss_full.detach(), k=num_hard).indices  # [H]
    hard_loss_detached = loss_full[hard_idx].detach()
    hard_weight = hard_loss_detached / (hard_loss_detached.sum() + 1e-8)  # [H]
 
    def _flat(grads):
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, encoder_params)
        ])
 
    # 5) per-sample hard gradients (no create_graph — hard samples are reference points)
    #    [CHANGE 1 — moved up]: compute hard grads ONCE before the class loop, reuse for every class.
    grads_h_dict = {}
    for j in hard_idx.tolist():
        gs_h = torch.autograd.grad(
            loss_full[j],
            encoder_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        grads_h_dict[j] = _flat(gs_h).detach()
 
    # [CHANGE 2]: enumerate classes present in the batch
    unique_classes = targets.unique().tolist()
 
    total_loss = torch.zeros((), device=device)
    n_active_classes = 0
 
    # [CHANGE 3]: outer loop over classes — per-class prototype gradients
    for k in unique_classes:
        class_mask = (targets == k)
        if class_mask.sum() == 0:
            continue
 
        # 4) AGGREGATED subgraph gradients PER CLASS (was: over full batch)
        gc_k_raw = torch.autograd.grad(
            loss_c[class_mask].mean(), encoder_params,
            retain_graph=True, create_graph=True, allow_unused=True,
        )
        gc_k = _flat(gc_k_raw)
 
        gs_k_raw = torch.autograd.grad(
            loss_s[class_mask].mean(), encoder_params,
            retain_graph=True, create_graph=True, allow_unused=True,
        )
        gs_k = _flat(gs_k_raw)
 
        # numerical guard
        if gc_k.norm() < grad_norm_eps or gs_k.norm() < grad_norm_eps:
            continue
 
        # 6) ranking loss: per-class aggregated c/s vs per-sample hard (shared across classes)
        class_loss = torch.zeros((), device=device)
        n_active_pairs = 0
        for w_j, j in zip(hard_weight, hard_idx.tolist()):
            gh_j = grads_h_dict[j]
            if gh_j.norm() < grad_norm_eps:
                continue
 
            c_score = F.cosine_similarity(gc_k.unsqueeze(0), gh_j.unsqueeze(0),
                                          dim=-1, eps=1e-8).squeeze()
            s_score = F.cosine_similarity(gs_k.unsqueeze(0), gh_j.unsqueeze(0),
                                          dim=-1, eps=1e-8).squeeze()
 
            pair_loss = F.relu(margin - c_score + s_score)
            class_loss = class_loss + w_j * pair_loss
            n_active_pairs += 1
 
        if n_active_pairs == 0:
            continue
 
        total_loss = total_loss + class_loss
        n_active_classes += 1
 
    if n_active_classes == 0:
        return torch.zeros((), device=device)
 
    # [CHANGE 4]: average across classes (each class' inner sum is in [0, 2] since hard_weight sums to 1)
    return total_loss / n_active_classes

def get_size_loss(edge_mask):
    """
    Compute the size loss in PGExp.
    """
    return edge_mask.sum()/edge_mask.size(0)

def get_conn_loss(edge_mask):
    """
    Compute the connectivity loss in PGExp.
    """
    edge_mask = edge_mask * 0.99 + 0.005
    mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
    
    return torch.mean(mask_ent)

def get_conti_loss_like_rnp(hard_node_mask):
    """
    Compute the continuity loss like RNP.
    Inputs:
        z -- (batch_size, sequence_length)
    """
    return torch.mean(torch.abs(hard_node_mask[1:] - hard_node_mask[:-1]))

# def get_conn_loss_by_edge(soft_node_mask, edge_index):
#     """
#     Compute the connectivity loss in PGExp by node.
#     """
#     edge_mask = torch.tensor([(soft_node_mask[edge_index[0][i]] + soft_node_mask[edge_index[1][i]])/2 for i in range(edge_index.shape[-1])]).to(soft_node_mask.device)
    

#     edge_mask_adj = torch.zeros((soft_node_mask.shape[0],soft_node_mask.shape[0]))
#     for i in range(edge_mask.shape[-1]):
#         edge_mask_adj[edge_index[0][i]][edge_index[1][i]] = edge_mask[i]

#     edge_mask = edge_mask * 0.99 + 0.005
#     mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)

#     return torch.mean(mask_ent)

