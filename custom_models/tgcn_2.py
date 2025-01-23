import torch
import torch.nn as nn
from torch import Tensor
from tsl.nn.layers.graph_convs.graph_conv import GraphConv
from torch_geometric.typing import Adj, OptTensor
from tsl.nn.utils import get_functional_activation
from tsl.nn.models import BaseModel
from tsl.nn.layers.recurrent.base import GraphGRUCellBase
from tsl.nn.blocks.encoders.recurrent.base import RNNBase
from tsl.nn.blocks.encoders import MLP
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.utils import maybe_cat_exog
import logging
logger = logging.getLogger(name=__name__)
logging.basicConfig(filename='info.log', level=logging.INFO)



class TGCNConv(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 norm: str = 'mean',
                 k:int=2,
                 bias:bool = True,
                 root_weight: bool = False,
                 activation: str = None,
                 cached: bool = False
                 ):
        super(TGCNConv, self).__init__()
        self.layers = torch.nn.ModuleList([GraphConv(input_size=input_size if i == 0 else hidden_size,
                                        output_size=hidden_size,
                                        bias=bias,
                                        norm=norm,
                                        root_weight=root_weight,
                                        activation=activation,
                                        cached=cached) for i in range(k)])
    
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight:  OptTensor = None,
                ):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x
    
class TGCNCell(GraphGRUCellBase):
    def __init__(self,
                 input_size:int,
                 spatial_hidden_size:int,  # Hidden_size after GCN
                 temporal_hidden_size:int,
                 norm='mean',
                 k=2,
                 root_weight: bool = True,
                 activation: str = None,
                 cached: bool = False):
        
        forget_gate = MLP(spatial_hidden_size + temporal_hidden_size,
                            temporal_hidden_size
                            )
        update_gate = MLP(spatial_hidden_size + temporal_hidden_size,
                            temporal_hidden_size
                            )
        candidate_gate = MLP(spatial_hidden_size + temporal_hidden_size,
                            temporal_hidden_size
                            )
        super(TGCNCell, self).__init__(hidden_size=temporal_hidden_size,
                                      forget_gate=forget_gate,
                                       update_gate=update_gate,
                                        candidate_gate=candidate_gate )
        self.gcn_layer = TGCNConv(input_size,
                            spatial_hidden_size,
                            norm=norm,
                            k=k,
                            root_weight = root_weight,
                            activation = activation,
                            cached = cached
                            )
    def forward(self, x: Tensor, h: Tensor, *args, **kwargs) -> Tensor:
        """
        x: [batch, *, channels] # input_size in channels
        h: [batch, *, channels] # temporal hidden size in channel
        """
        embed_gcn = self.gcn_layer(x, *args, **kwargs)
        x_gates = torch.cat([embed_gcn, h], dim=-1)
        r = torch.sigmoid(self.forget_gate(x_gates))
        u = torch.sigmoid(self.update_gate(x_gates))
        x_c = torch.cat([embed_gcn, r * h], dim=-1)
        c = torch.tanh(self.candidate_gate(x_c))        
        h_new = u * h + (1. - u) * c
        return h_new

class TGCN(RNNBase):
    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 norm:str = 'mean',
                 n_layers:int=2,
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 k: int = 2,
                 root_weight: bool = True,
                 activation: str = None,
                 cached: str = None
                 ):
        
        self.input_size = input_size
        self.hidden_dize = hidden_size
        rnn_cells = [
            TGCNCell(input_size=input_size if i == 0 else hidden_size,
                     temporal_hidden_size = hidden_size,
                     spatial_hidden_size = hidden_size,
                     norm=norm,
                     k=k,
                     root_weight = root_weight,
                     activation = activation,
                     cached = cached
                     )
        for i in range(n_layers)]
        super(TGCN, self).__init__(rnn_cells, cat_states_layers, return_only_last_state)

class TGCNModel_2(BaseModel):
    def __init__(self,
                 input_size: int,
                 horizon:int,
                 n_layers:int = 1,
                 hidden_size:int = 32,
                 spatial_kernel:int=2,
                 exog_size: int = 0,
                 ff_size = 256,
                 activation: str = 'relu'
                 ):
        super(TGCNModel_2, self).__init__()
        self.tgcn = TGCN(input_size=input_size+exog_size,
                         hidden_size=hidden_size,
                         n_layers=n_layers,
                         k=spatial_kernel,
                         return_only_last_state=True
                         )
        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=ff_size,
                                  output_size=input_size,
                                  horizon=horizon,
                                  activation=activation)
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                u = None):
        x = maybe_cat_exog(x=x, u=u)
        out = self.tgcn(x,
                        edge_index,
                        edge_weight)
        out =self.readout(out)
        return out



