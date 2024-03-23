import flax.linen as nn
import functools as ft

from typing import Type
from ...nn.gnn import GNN
from ...nn.mlp import MLP
from ...nn.utils import default_nn_init
from ...utils.typing import Array, Params
from ...utils.graph import GraphsTuple


class CBFNet(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Array:
        x = self.gnn_cls()(obs, node_type=0, n_type=n_agents)
        x = self.head_cls()(x)
        x = nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x))
        return x


class CBF:

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, gnn_layers: int, dim_factor: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.dim_factor = dim_factor

        self.cbf_gnn = ft.partial(
            GNN,
            msg_dim=64 * dim_factor,
            hid_size_msg=(128 * dim_factor, 128 * dim_factor),
            hid_size_aggr=(64 * dim_factor, 64 * dim_factor),
            hid_size_update=(128 * dim_factor, 128 * dim_factor),
            out_dim=64 * dim_factor,
            n_layers=gnn_layers
        )
        self.cbf_head = ft.partial(
            MLP,
            hid_sizes=(128 * dim_factor, 128 * dim_factor),
            act=nn.relu,
            act_final=False,
            name='CBFHead'
        )
        self.net = CBFNet(
            gnn_cls=self.cbf_gnn,
            head_cls=self.cbf_head
        )

    def get_cbf(self, params: Params, obs: GraphsTuple) -> Array:
        return self.net.apply(params, obs, self.n_agents)
