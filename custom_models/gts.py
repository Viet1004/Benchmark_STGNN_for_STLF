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

import logging
logger = logging.getLogger(name=__name__)
# logging.basicConfig(filename=f'infos/{__name__}__info.log', level=logging.INFO)
# torch.cuda.memory._record_memory_history()

class StaticGTS(BaseModel, LightningModule):
    def __init__(self, 
                 input_size: int,   # number of features
                 window: int,       # number of steps
                 horizon: int,      # number of forecasting steps
                 n_nodes: int,      # number of time series
                 hidden_size: int,  # dimension of hidden states
                 nodes_features: torch.Tensor, 
                 encoder_layers: int = 1,
                 decoder_layers: int = 1,
                 embedding_dim_mlp: int = 32,
                 initial_decay = 10,
                 use_curriculum_learning = True,
                 loss_fn = mae
                ):
        super(StaticGTS, self).__init__()
        self.input_size = input_size
        self.window = window
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.nodes_features = nodes_features.detach()
        self.hidden_size = hidden_size 
        self.inital_decay = initial_decay
        self.use_curriculum_learning = use_curriculum_learning
        out_channel_encoder = 4
        kernel = 15
        stride = 4
        padding = 7
        print(f'Length before linear layer: {nodes_features.shape}')
        length_ff=(nodes_features.shape[-1] + 2*padding - (kernel - 1 ) - 1)//stride + 1
        length_ff = (length_ff + 2*padding - (kernel - 1) -1)//stride + 1
        # logger.info(f'Length before linear layer: {length_ff}')
        self.input_encoder = nn.Sequential(
            nn.Conv1d(input_size,out_channel_encoder,kernel, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channel_encoder),
            nn.Dropout(0.2),
            nn.Conv1d(out_channel_encoder,2*out_channel_encoder,kernel, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(2*out_channel_encoder),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(2*out_channel_encoder*length_ff,embedding_dim_mlp),    ### Need to modify. This links to length of time series
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim_mlp),
        )        
        self.automatic_optimization = False
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim_mlp * 2, embedding_dim_mlp),
            nn.ReLU(),
            nn.Linear(embedding_dim_mlp, 2)
            )

        self.rearange_encoder = Rearrange('b t n c -> b n t c')
        self.encoder = DCRNN(input_size=input_size, hidden_size=hidden_size, n_layers=encoder_layers)
        self.decoder = DCRNN(input_size=input_size, hidden_size=hidden_size, n_layers=decoder_layers)
        self.readout = MLPDecoder(input_size=hidden_size, hidden_size=8*hidden_size, output_size=input_size)
        self.loss_fn = loss_fn
        # total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.first_time = True
        # logger.info(f"Total number of parameters: {total_params}")

    def _threshold(self, batch_num):
        return self.inital_decay / (
            self.inital_decay + np.exp(batch_num / self.inital_decay))

    def forward(self, 
                x, 
                labels: torch.Tensor = None,
                u: OptTensor = None,
                batch_num: int = None) -> Tensor:

        # What we see now is (batch, window, n_nodes, input_size)
        # labels: input of shape (batch, horizon, n_nodes, input_size (number of feature))
        batch_size, window, n_nodes, input_size = x.shape
        # logger.info(f"Shape of input {x.shape}")
        # logger.info(f'Number of nodes is: {n_nodes}')
        abstract_node_features = self.input_encoder(self.nodes_features.to(x.device))
        # print(f"Abstract node feature shape: {abstract_node_features.shape}")
        first_tensor = abstract_node_features.unsqueeze(1).expand(-1,n_nodes,-1)
        second_tensor = abstract_node_features.unsqueeze(0).expand(n_nodes, -1,-1)
        adj = torch.concat((first_tensor, second_tensor), dim = -1)
        adj = self.mlp(adj)
        # adj = abstract_node_features @ abstract_node_features.T
        adj = nn.functional.gumbel_softmax(adj, tau=1, hard=True)[:,:,0].clone().view(self.n_nodes, -1)
        
        # torch.cuda.memory._snapshot()
        print(f'Before gumbel: {adj.shape}')
        # torch.cuda.memory._dump_snapshot(filename='debug/dump_snapshot.pickle')
        rows, cols = torch.nonzero(adj, as_tuple=True)
        edge_index = torch.vstack((rows,cols))
        edge_weight = adj[rows, cols]
        
        # encoder_hidden_state = torch.zeros(self.n_nodes, self.hidden_size, self.input_size, dtype=x.dtype)
        encoder_hidden_state = torch.zeros(x.size(0),
                                            x.size(-2),
                                            self.hidden_size,
                                            dtype=x.dtype,
                                            device=x.device)
        # x = self.rearange_encoder(x)
        batch_size, window, n_nodes, input_size = x.shape
        # logger.info(f"Shape of input after rearange {x.shape}")
        # logger.info(f"Shape of hidden state is: {encoder_hidden_state.shape}")
        for t in range(self.window):
            _, encoder_hidden_state = self.encoder(x, edge_index, edge_weight, h=encoder_hidden_state)
        decoder_hidden_state = encoder_hidden_state[-1]
        # logger.info(f'Decoder hidden state shape: {decoder_hidden_state.shape}')
        decoder_input = torch.zeros(batch_size,window, n_nodes, input_size).to(x.device)
        output_list = []
        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = self.decoder(decoder_input, edge_index, edge_weight, h = decoder_hidden_state)
            readout_output = self.readout(decoder_output)
            decoder_input = readout_output
            if self.training and self.use_curriculum_learning:   
                c = np.random.uniform()
                if c < self._threshold(batch_num=batch_num):
                    decoder_input = labels[:,t,:,:] 
            readout_output = self.readout(decoder_output)
            output_list.append(readout_output)
            # logger.info(f'Step: {t}')                
        if self.first_time:
            logger.info(f"Shape of outputs: {readout_output.shape}")
            # logger.info(f"Shape of {}")
            self.first_time = False
        outputs = torch.concat(output_list, dim=1)
        # logger.info(f'Shape of outputs: {outputs.shape}')
        return outputs

    def training_step(self, batch, batch_idx):
        # logger.info(f'In batch: {batch_idx}')
        inputs, labels = batch
        opt = self.optimizers()
        if self.training:
            if self.use_curriculum_learning:
                outputs = self.forward(inputs, labels)
                loss = self.loss_fn(outputs, labels)
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
            else:
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, labels)
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
    
# class GTSSupervisor:
#     def __init__(self, dataloader, optimizer, **model_kwargs):
#         input_size = int(model_kwargs.get("input_size"))
#         window = int(model_kwargs.get("window"))
#         horizon = int(model_kwargs.get("horizon"))
#         hidden_size = int(model_kwargs.get("hidden_size"))
#         n_nodes = int(model_kwargs.get("n_nodes"))
#         nodes_features = torch.Tensor(model_kwargs.get("nodes_features"))
#         self.dataloader = dataloader
#         self.optimizer = optimizer
#         self.model = StaticGTS(input_size=input_size,
#                                window=window,
#                                horizon=horizon,
#                                n_nodes=n_nodes,
#                                hidden_size=hidden_size,
#                                nodes_features=nodes_features)
        
#     def train(self, config):
#         epochs = config["epochs"]
#         threshold = config["threshold"]
#         for i, (inputs, labels) in enumerate(self.dataloader):
#             output = self.model(inputs, labels= labels)
             


#     def save_model(self, path=None):
#         if path == None:
#             torch.save(self.model.state_dict(), "saved_models/gts.pth")
#         else:
#             torch.save(self.model.state_dict(), path)

#     def _save_model_checkpoint(self, epoch, loss, PATH = "saved_models/model_gts.pth"):
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss': loss,
#         },
#         PATH )
#     def load_model(self, path):
#         return torch.load(path)


