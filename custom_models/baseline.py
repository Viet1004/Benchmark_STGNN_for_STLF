from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
import torch.nn as nn

class SameHour(nn.Module):
    def __init__(self,
                 input_size: int, # The number of features in input
                 window: int,       # number of steps
                 horizon: int,      # number of forecasting steps
                 n_nodes: int,      # number of time series
                 ):
        super(SameHour, self).__init__()
        self.input_size = input_size
        self.window = window
        self.horizon = horizon
        self.n_nodes = n_nodes
        
    def forward(self, 
                x: Tensor,
                edge_index: Adj,
                edge_weight:  OptTensor = None,
                u: OptTensor = None):
        return x[:,:self.horizon,:,:]
    
class LastValue(nn.Module):
    def __init__(self,
                 input_size: int, # The number of features in input
                 window: int,       # number of steps
                 horizon: int,      # number of forecasting steps
                 n_nodes: int,      # number of time series
                 ):
        super(LastValue, self).__init__()
        self.input_size = input_size
        self.window = window
        self.horizon = horizon
        self.n_nodes = n_nodes
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                u: OptTensor = None):
        return x[:,-self.horizon:, :, :]
        
