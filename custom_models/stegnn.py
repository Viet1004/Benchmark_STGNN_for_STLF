from torch import Tensor, nn
import torch
import numpy as np
import torch.nn as nn
from tsl.nn.models import BaseModel
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from einops.layers.torch import Rearrange
from einops import rearrange
from tsl.nn.utils import get_layer_activation, maybe_cat_exog
from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.blocks.encoders import DCRNN
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.layers.recurrent import GraphGRUCellBase
from lightning.pytorch import LightningModule
from tsl.metrics.torch import mae
from logging import Logger
from einops.layers.torch import Rearrange  # reshape data with Einstein notation
from pathlib import Path
import logging
logger = logging.getLogger(name=__name__)
Path("infos/").mkdir(parents=True, exist_ok=True)
# logging.basicConfig(filename=f'/infos/{__name__}__info.log', level=logging.INFO)

class STEGNN(BaseModel):
    def __init__(self,
                 input_size: int, # The number of features in input
                 window: int,       # number of steps
                 horizon: int,      # number of forecasting steps
                 n_nodes: int,      # number of time series
                 temporal_embedding_size: int = None,
                 node_emb_size: int = 64,
                 eps: float = 1,
                 topk: int = 150,
                 k = 2,
                 L = 2,  # Number of layers in readout
                 beta = 0.5 # How much to retain information in propagation. The higher the more different.
                 ):
        super(STEGNN, self).__init__()
        self.input_size = input_size
        self.window = window
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.node_emb_size = node_emb_size  
        self.source_embeddings = NodeEmbedding(n_nodes, node_emb_size)
        self.target_embeddings = NodeEmbedding(n_nodes, node_emb_size)
        self.source_node_ff = nn.Linear(node_emb_size, node_emb_size)
        self.target_node_ff = nn.Linear(node_emb_size, node_emb_size)
        self.dynamic_transform = nn.Linear(self.window, node_emb_size)
        self.linear_layer_attention = nn.Linear(2*node_emb_size, 1)
        self.latent_space_transform_layer = nn.Linear(window,window)  # Output C
        self.static_embedding_layer = nn.Linear(k * window, node_emb_size)  # Output S_static
        self.dynamic_embedding_layer = nn.Linear(k * window, node_emb_size)  # Output S_dynamic
        readout_dim = window +  2 * node_emb_size

        if temporal_embedding_size is not None:
            self.day_embeddings = nn.Embedding(7, temporal_embedding_size )  # Lookup table for day in a week
            self.timeslot_embeddings = nn.Embedding(48, temporal_embedding_size)  # Lookup table for hour in a week
            readout_dim += 2 * temporal_embedding_size
            
        self.readout = nn.ModuleList([nn.Sequential(nn.Linear(input_size * readout_dim, input_size *readout_dim), nn.ReLU(), nn.Linear(input_size*readout_dim,input_size*readout_dim)) for i in range(L)] + [nn.Linear(readout_dim, horizon)])
        self.eps = eps
        self.topk = topk
        self.k = k
        self.beta = beta

    def _static_adj(self):
        M1 = (self.eps * self.source_node_ff(self.source_embeddings())).tanh()
        M2 = (self.eps * self.target_node_ff(self.target_embeddings())).tanh()
        static_adj = nn.functional.relu((self.eps * (M1 @ M2.T - M2 @ M1.T)).tanh())
        ret = torch.topk(static_adj, k=self.topk)
        res = torch.zeros_like(static_adj)
        res.scatter_(-1, ret.indices, ret.values)
        return res
    
    def _dynamic_adj(self, x):
        logger.info(f'Shape of input {x.shape}')
        x_reshaped = x.permute(0, 2, 3, 1)  # shape becomes (32, 50, 1, 24)

        # Apply the linear transformation
        input_transformed = self.dynamic_transform(x_reshaped)  # Output shape will be (32, 50, 1, 64)

        # If needed, permute back to the original format
        input_transformed = input_transformed.permute(0, 3, 1, 2)  # shape becomes (32, 64, 50, 1)
        
        logger.info(f'Shape of input_transformed: {input_transformed.shape}')
        X_concat = torch.cat([input_transformed.unsqueeze(2).expand(-1, -1, self.n_nodes, -1, -1),  # Broadcast node i
                      input_transformed.unsqueeze(3).expand(-1, -1, -1, self.n_nodes, -1)],  # Broadcast node j
                     dim=1)  # Concatenate along the embedding dimension
        logger.info(f'Shape of input_transformed: {X_concat.shape}')
        
        batch_size, two_embedding_size, n_nodes, n_nodes, input_size = X_concat.shape

        X_concat_flattened = X_concat.view(batch_size, two_embedding_size, -1).transpose(1, 2)  # Flatten and prepare for linear layer
        
        attention_scores = self.linear_layer_attention(X_concat_flattened)  # Shape (batch_size * n_nodes * n_nodes, 1)
        logger.info(f'Shape of attention score: {attention_scores.shape}')
        attention_scores = attention_scores.view(batch_size, n_nodes, n_nodes)  # Reshape back to (batch_size, n_nodes, n_nodes, 1)
        dynamic_adj = torch.full(attention_scores.shape, 10**(-9)).to(x.device)
        ret = torch.topk(attention_scores, k=self.topk)
        dynamic_adj.scatter_(-1, ret.indices, ret.values)
        dynamic_adj = torch.softmax(dynamic_adj, dim=-1)
        logger.info(f'Shape of dynamic_adj: {dynamic_adj.shape}')
        return dynamic_adj
    
    def _moving_average(self, x):
        # The shape of input is (batch_size, window, n_nodes, input_size)
        logger.info(f'Assure that input requires_grad is {x.requires_grad}')
        batch_size, window, n_nodes, input_size = x.shape
        static_adj = self._static_adj()
        dynamic_adj = self._dynamic_adj(x) # shape of dynamic_adj is (batch_size, n_nodes, n_nodes)
        batch_size , n_nodes, _ = dynamic_adj.shape
        static_with_iden = static_adj + torch.eye(n_nodes).to(x.device)
        dynamic_with_iden = dynamic_adj + torch.eye(n_nodes).unsqueeze(0).to(x.device)
        normalized_static_adj = (static_with_iden/static_with_iden.sum(dim=-1, keepdim=True)).unsqueeze(0).expand(batch_size, -1, -1)
        normalized_dynamic_adj = dynamic_with_iden/dynamic_with_iden.sum(dim=-1, keepdim=True)
        logger.info(f'Size of normalized static adjacency matrix {normalized_static_adj.shape}')
        logger.info(f'Size of normalized static adjacency matrix {normalized_dynamic_adj.shape}')        
        H_static = x.clone()
        H_dynamic = x.clone()
        H_static_stack = []
        H_dynamic_stack = []
        for i in range(self.k):
            theta_static = torch.einsum('bwni, bjk->bwki', H_static, normalized_static_adj)
            H_static = self.beta * H_static + (1 - self.beta) * theta_static
            H_static_stack.append(H_static)
            
            theta_dynamic = torch.einsum('bwni, bjk->bwki', H_dynamic, normalized_dynamic_adj)
            H_dynamic = self.beta * H_dynamic + (1 - self.beta) * theta_dynamic
            H_dynamic_stack.append(H_dynamic)
        H_static_concat = torch.cat(H_static_stack, dim=1)
        H_dynamic_concat = torch.cat(H_dynamic_stack, dim=1)
        H_static_concat = H_static_concat.transpose(1,2).view(batch_size, n_nodes, -1)
        S_static = self.static_embedding_layer(H_static_concat)
        S_static = S_static.view(batch_size, n_nodes, -1, input_size)
        logger.info(f'S_static has dimension: {S_static.shape}') # S_static should have dimension (batch, static_embedding_dim, n_nodes)
        H_dynamic_concat = torch.cat(H_dynamic_stack, dim=1)
        H_dynamic_concat = H_dynamic_concat.transpose(1,2).view(batch_size, n_nodes, -1)
        S_dynamic = self.dynamic_embedding_layer(H_dynamic_concat)
        S_dynamic = S_dynamic.view(batch_size, n_nodes, -1, input_size)
        logger.info(f'S_dynamic has dimension: {S_dynamic.shape}') # S_dynamic should have dimension (batch, dynamic_embedding_dim, n_nodes)
        return S_static.transpose(1,2), S_dynamic.transpose(1,2)
    
    def forward(self, 
                x: Tensor,
                edge_index: Adj = None,
                edge_weight:  OptTensor = None,
                u: OptTensor = None
                ):
        temporal_embedding = None
        if (u is not None) and u.dim() == 3:
            # print(u.shape)
            # u = u[:,-1,:]
            timeslot_indices = u[:,-1,0]
            weekday_indices = u[:,-1,1]
            timeslot_embed = self.timeslot_embeddings(timeslot_indices)
            weekday_embed = self.day_embeddings(weekday_indices)
            temporal_embedding = torch.concat((timeslot_embed,weekday_embed), dim=-1).unsqueeze(-1).expand(-1,-1,self.n_nodes).unsqueeze(-1)


        batch_size, window, n_nodes, input_size = x.shape
        logger.info(f'Shape of input is: {x.shape}')
        S_static, S_dynamic = self._moving_average(x) # Shape of S_static, S_dynamic are (batch_size, emb_size, n_nodes)
        logger.info(f'Shape of S_static: {S_static.shape} and S_dynamic: {S_dynamic.shape}')
        # encoded_C = torch.einsum('bwni,wh->bhni', x, self.latent_space_transform_layer.weight.T) + self.latent_space_transform_layer.bias # Shape of encoded_C is (batch_size, emb_size, n_nodes, input_size)
        x = x.permute(0,2,3,1)
        encoded_C = self.latent_space_transform_layer(x)
        encoded_C = encoded_C.permute(0,3,1,2)
        output = torch.concat((S_static, S_dynamic, encoded_C), dim=1)
        if temporal_embedding is not None:
            output = torch.concat((output, temporal_embedding), dim=1)
        output = output.permute(0,2,3,1)
        logger.info(f'Shape of output before readout: {output.shape}')
        layers_output = len(self.readout)
        for i, layer in enumerate(self.readout):
            if i < layers_output-1:
                output = layer(output) + output
            else:
                output = layer(output)
        output = output.permute(0,3,1,2)
        logger.info(f'Shape of output: {output.shape}')
        return output        




                

        