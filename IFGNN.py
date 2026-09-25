import munch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected, subgraph
from torch_sparse import transpose

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor
from .Pooling import GlobalMeanPool
from munch import munchify
from .MolEncoders import AtomEncoder, BondEncoder
from GOOD.utils.fast_pytorch_kmeans import KMeans


@register.model_register
class IFGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(IFGIN, self).__init__(config)

        # --- if environment inference ---
        config.environment_inference = False
        if config.environment_inference:
            self.env_infer_warning = f'#W#Expermental mode: environment inference phase.'
            config.dataset.num_envs = 3
        # --- Test environment inference ---

        self.config = config

        self.learn_edge_att = True
        self.top_ratio = config.ood.extra_param[2]
        self.without_embed = config.ood.extra_param[3]


        fe_kwargs = {'without_embed': True if self.without_embed else False}

        # --- Build networks ---
        self.feature_mlp = EFMLP(config, bn=True)
        self.generator = GINFeatExtractor(config, **fe_kwargs)
        #self.mlp_gen=nn.Linear(config.model.dim_hidden,2)
        #self.mlp_gen=MLP([config.model.dim_hidden,2], dropout=config.model.dropout_rate,
        #                                 config=config, bn=True)
        self.mlp_gen = ExtractorMLP(config)
        self.predictor = GINFeatExtractor(config, **fe_kwargs)
        #**kwargs – without_readout will output node features instead of graph features.

        self.pool = GlobalMeanPool()
        self.classifier = Classifier(config)
        self.edge_mask = None



    def forward(self, *args, **kwargs):
        r"""
        The LECIGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        if self.without_embed:
            data.x = self.feature_mlp(data.x, data.batch)
            kwargs['data'] = data
        node_repr_gen = self.generator.get_node_repr(*args, **kwargs)

        att = self.mlp_gen(node_repr_gen, data.edge_index, data.batch)
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                nodesize = data.x.shape[0]
                edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
        soft_edge_att = torch.sigmoid(edge_att)

        # convert soft edge mask to hard
        hard_edge_att = F.gumbel_softmax(soft_edge_att, tau=1, hard=True)

        # control sparsity
        soft_edge_att = control_sparsity(soft_edge_att, top_t = self.top_ratio)

        set_masks(soft_edge_att, self.predictor)
        repr_pre = self.predictor(*args, **kwargs)
        clear_masks(self)
        logit_c = self.classifier(repr_pre)

        # spurious/complement subgraph Gs
        set_masks(1.0 - soft_edge_att, self.predictor)
        repr_s = self.predictor(*args, **kwargs)
        clear_masks(self)
        logit_s = self.classifier(repr_s)

        # full graph G
        repr_full = self.predictor(*args, **kwargs)
        logit_full = self.classifier(repr_full)

        return soft_edge_att,hard_edge_att,logit_c, logit_s, logit_full
    
    def predict_with_mask(self, edge_mask, *args, **kwargs):
        data = kwargs.get('data')
        if self.without_embed:
            data = data.clone()
            data.x = self.feature_mlp(data.x, data.batch)
            kwargs['data'] = data

        set_masks(edge_mask, self.predictor)
        repr_pre = self.predictor(*args, **kwargs)
        clear_masks(self)
        return self.classifier(repr_pre)

    
    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att


@register.model_register
class IFvGIN(IFGIN):
    r"""
    The GIN virtual node version of LECI.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(IFvGIN, self).__init__(config)
        fe_kwargs = {'without_embed': True if self.without_embed else False}
        self.generator = vGINFeatExtractor(config, **fe_kwargs)
        self.predictor = vGINFeatExtractor(config, **fe_kwargs)


class EFMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch], bn):
        super(EFMLP, self).__init__()
        if config.dataset.dataset_type == 'mol':
            self.atom_encoder = AtomEncoder(config.model.dim_hidden, config)
            self.mlp = MLP([config.model.dim_hidden, config.model.dim_hidden, 2 * config.model.dim_hidden,
                            config.model.dim_hidden], config.model.dropout_rate, config, bn=bn)
        else:
            self.atom_encoder = nn.Identity()
            self.mlp = MLP([config.dataset.dim_node, config.model.dim_hidden, 2 * config.model.dim_hidden,
                            config.model.dim_hidden], config.model.dropout_rate, config, bn=bn)

    def forward(self, x, batch):
        return self.mlp(self.atom_encoder(x), batch)


class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = True
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p,
                                         config=config, bn=True)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p,
                                         config=config, bn=True)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch=None):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                assert batch is not None
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, config, bias=True, bn=False):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if bn:
                    m.append(nn.BatchNorm1d(channels[i]))
                else:
                    m.append(InstanceNorm(channels[i]))

                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

def control_sparsity(mask, top_t):
    r"""

    :param mask: mask that need to transform
    :param top_t: sparsity we need to control i.e. 0.7, 0.5
    :return: transformed mask where top 1 - sparsity values are set to inf.
    """
    while(mask.shape[-1] == 1):
        mask = mask.squeeze(-1)

    _, indices = torch.sort(mask, descending=True)
    mask_len = mask.shape[0]
    split_point = int(top_t * mask_len)
    important_indices = indices[: split_point]
    unimportant_indices = indices[split_point:]
    trans_mask = mask.clone()
    #trans_mask[important_indices] = 1.
    trans_mask[unimportant_indices] = 0.

    return trans_mask

def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._apply_sigmoid = False
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None

def mask_to_index(mask: Tensor) -> Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.

    Example:
        >>> mask = torch.tensor([False, True, False])
        >>> mask_to_index(mask)
        tensor([1])
    """
    return mask.nonzero(as_tuple=False).view(-1)