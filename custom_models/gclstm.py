import torch
import torch.nn as nn
from torch import Tensor
from tsl.nn.layers.graph_convs.graph_conv import GraphConv
from torch_geometric.typing import Adj, OptTensor
from tsl.nn.utils import get_functional_activation
from tsl.nn.models import BaseModel
from tsl.nn.layers.recurrent.base import GraphGRUCellBase
from tsl.nn.blocks.encoders.recurrent.base import RNNBase
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.layers.recurrent.gcrnn import GraphConvLSTMCell

import logging
logger = logging.getLogger(name=__name__)
logging.basicConfig(filename=f'infos/{__name__}__info.log', level=logging.INFO)

class GraphConvLSTMModel(BaseModel):
    def __init__(self,
                 input_size: int,
                 horizon:int,
                 n_layers:int = 4,
                 hidden_size:int = 32,
                 ff_size = 256,
                 activation: str = 'relu'
                 ):
        super(GraphConvLSTMModel, self).__init__()
        self.gclstm = GraphConvLSTM(input_size=input_size,
                         hidden_size=hidden_size,
                         n_layers=n_layers,
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
                edge_weight: OptTensor = None):
        out = self.gclstm(x,
                        edge_index,
                        edge_weight)
        out = self.readout(out)
        return out
    

class GraphConvLSTM(RNNBase):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 norm: str = 'mean',
                 n_layers: int = 2,
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 root_weight: bool = False,
                 cached: bool = False,
                 ):
        self.input_size = input_size
        self.hidden_dize = hidden_size

        rnn_cells = [
            GraphConvLSTMCell(input_size=input_size if i == 0 else hidden_size,
                              hidden_size=hidden_size,
                              norm=norm,
                              root_weight=root_weight,
                              cached=cached)
                              for i in range(n_layers)]
        super(GraphConvLSTM, self).__init__(rnn_cells, 
                                            cat_states_layers=cat_states_layers,
                                            return_only_last_state=return_only_last_state)





