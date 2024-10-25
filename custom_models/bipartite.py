from torch import Tensor, nn
import torch
import numpy as np
from tsl.nn.models import BaseModel
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from einops.layers.torch import Rearrange
from einops import rearrange
from tsl.nn.utils import get_layer_activation, maybe_cat_exog
from tsl.nn.layers.base import NodeEmbedding

from tsl.nn.layers.graph_convs import GatedGraphNetwork
        
class BiPartiteSTGraphModel(BaseModel):
    def __init__(self,
                 input_size: int,
                 input_window_size: int,
                 horizon: int,
                 n_nodes: int,
                 hidden_size: int,
                 output_size: int = None,
                 exog_size: int = 0,
                 n_aux_nodes: int = 10,
                 enc_layers: int = 1,
                 gnn_layers: int = 1,
                 activation: str = 'silu'):

        super(BiPartiteSTGraphModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size or input_size
        self.input_window_size = input_window_size

        input_size += exog_size

        self.input_encoder = nn.Sequential(
            nn.Linear(input_size * input_window_size, hidden_size), )

        self.encoder_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size),
                          get_layer_activation(activation)(),
                          nn.Linear(hidden_size, hidden_size))
            for _ in range(enc_layers)
        ])

        self.main_to_aux_edge_index = torch.zeros((n_nodes + n_aux_nodes, n_nodes + n_aux_nodes), dtype=torch.int32)

        self.aux_to_main_edge_index = torch.zeros((n_nodes + n_aux_nodes, n_nodes + n_aux_nodes), dtype=torch.int32)

        self.main_to_aux_edge_index[-n_aux_nodes:,:n_nodes] = 1

        self.aux_to_main_edge_index[:n_nodes,-n_aux_nodes] = 1

        self.main_to_aux_edge_index = self.main_to_aux_edge_index.nonzero().t().contiguous()
        self.aux_to_main_edge_index = self.aux_to_main_edge_index.nonzero().t().contiguous()

        self.n_nodes = n_nodes

        self.emb = NodeEmbedding(n_nodes=n_nodes + n_aux_nodes, emb_size=hidden_size)

        self.gcn_layers_main_to_auxillary = [
            (GatedGraphNetwork(hidden_size, hidden_size, activation=activation),
             GatedGraphNetwork(hidden_size, hidden_size, activation=activation))
            for _ in range(gnn_layers)
        ]

        self.decoder = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     get_layer_activation(activation)())

        self.readout = nn.Sequential(
            nn.Linear(hidden_size, horizon * self.output_size),
            Rearrange('b n (h f) -> b h n f', h=horizon, f=self.output_size))

    def forward(self, x, edge_index=None, u=None):
        
        x = maybe_cat_exog(x, u)
        x = rearrange(x[:, -self.input_window_size:], 'b s n f -> b n (s f)')
        
        x = self.input_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x) + x

        y = self.emb().unsqueeze(0)
        
        y = y.repeat(x.size(0), 1, 1)
        
        y[:,:self.n_nodes] += x
        
        for layer in self.gcn_layers_main_to_auxillary:
            main_to_aux = layer[0]
            y = main_to_aux(y, self.main_to_aux_edge_index)
            aux_to_main = layer[1]
            y = aux_to_main(y, self.aux_to_main_edge_index)

        x = y[:,:self.n_nodes]

        x = self.decoder(x) + x

        return self.readout(x)
    
    

        