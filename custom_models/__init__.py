from .bipartite import BiPartiteSTGraphModel
from .gts import StaticGTS
from .tgcn import TGCNModel
from .tgcn_2 import TGCNModel_2
from .stegnn import STEGNN
from .gclstm import GraphConvLSTMModel
from .baseline import SameHour, LastValue
__all__ = ["BiPartiteSTGraphModel", "StaticGTS", "TGCNModel", "TGCNModel_2", "STEGNN", "GraphConvLSTMModel", "SameHour", "LastValue"]

classes = __all__