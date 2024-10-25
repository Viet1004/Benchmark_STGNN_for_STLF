
# %%
import os
import torch

print(torch.__version__)

import tsl
import copy
import torch
import scipy
import numpy as np
import pandas as pd
import scipy.special
from dtw import dtw
import scipy.spatial.distance as sd
from tsl.data import SpatioTemporalDataset
from tsl.metrics.torch import MaskedMAE, MaskedMAPE
from tsl.engines import Predictor
from tsl.data.datamodule import (SpatioTemporalDataModule,
                                 TemporalSplitter)
from custom_models import BiPartiteSTGraphModel, StaticGTS, STEGNN, TGCNModel, GraphConvLSTMModel, SameHour, LastValue
from tsl.nn.utils import get_layer_activation, maybe_cat_exog

from tsl.data.preprocessing import StandardScaler
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from tsl.nn.models import BaseModel
from tsl.nn.utils import get_layer_activation, maybe_cat_exog
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tsl.nn.models.stgn.graph_wavenet_model import GraphWaveNetModel
from tsl.nn.models.stgn.gru_gcn_model import GRUGCNModel
from tsl.nn.models.stgn.agcrn_model import AGCRNModel
from tsl.nn.models.temporal.linear_models import VARModel
from tsl.nn.models.temporal.rnn_model import RNNModel


print(f"tsl version  : {tsl.__version__}")
print(f"torch version: {torch.__version__}")

pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(edgeitems=3, precision=3)
torch.set_printoptions(edgeitems=2, precision=3)

# Utility functions ################
def print_matrix(matrix):
    return pd.DataFrame(matrix)

def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

class SaveAdjMatrix(pl.callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Check if the current epoch is a multiple of 5
        # if (trainer.current_epoch + 1) % 5 == 0:
            print(f"Epoch {trainer.current_epoch + 1} has ended.")
            # Execute custom logic
            self.save_adj_matrix(trainer, pl_module)
    
    def save_adj_matrix(self, trainer, pl_module):
        print(pl_module.__class__.__name__)
        # Check if the model has the 'get_adj_matrix' method
        if hasattr(pl_module.model, 'get_learned_adj') and callable(getattr(pl_module.model, 'get_learned_adj')):
            # Execute the 'get_adj_matrix' method
            adj_matrix = pl_module.model.get_learned_adj()
            print(adj_matrix[:5,:5])
            # print("Adjacency matrix obtained from get_learned_adj method:", adj_matrix)
        else:
            print("The model does not have a callable 'get_learned_adj' method.")



class STGraph:
    def __init__(self, models: list, datamodules: list[SpatioTemporalDataModule], loss_fn = None, metrics = None):
        self.models = models
        self.datamodules = datamodules
        
        if loss_fn:
            self.loss_fn = loss_fn
        else: 
            self.loss_fn = MaskedMAE()
        if metrics:
            self.metrics =  metrics
        else: 
            self.metrics = {'mae': MaskedMAE(),
                'mape': MaskedMAPE(),
                # 'mae_at_15': MaskedMAE(at=2),  # '2' indicates the third time step,
                                                # which correspond to 15 minutes ahead
                'mae_at_30': MaskedMAE(at=5),
                # 'mae_at_60': MaskedMAE(at=11)
                }
    def add_model(self, model):
        self.models.append(model)

    def run(self, config):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)


        if torch.cuda.is_available():
            accelerator = 'gpu'
        # elif torch.backends.mps.is_available():
        #     accelerator = 'mps'
        else:
            accelerator = 'cpu'

        max_epochs = config['max_epochs']
        devices = config['devices']
        limit_val_batches = config['limit_val_batches']

        for datamodule in self.datamodules:
            for model in self.models:
                logger = TensorBoardLogger(save_dir="logs", name=model.__class__.__name__, version=0)
                
                checkpoint_callback = ModelCheckpoint(
                    dirpath=f'checkpoint/{model.__class__.__name__}',
                    save_top_k=1,
                    monitor='val_mae',
                    mode='min',
                )
                            
                save_adj = SaveAdjMatrix()
                
                early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)

                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    logger=logger,
                    accelerator=accelerator,
                    devices=devices,
                    limit_train_batches=limit_val_batches,
                    check_val_every_n_epoch=5,
                    callbacks = [checkpoint_callback, save_adj, early_stopping],
                    # callbacks = [checkpoint_callback]
                )
                predictor = Predictor(
                    model=model,
                    optim_class=torch.optim.Adam,  # specify optimizer to be used...
                    optim_kwargs={'lr': 0.001},    # ...and parameters for its initialization
                    loss_fn=self.loss_fn,               # which loss function to be used
                    metrics=self.metrics                # metrics to be logged during train/val/test
                )
                try:
                    trainer.fit(predictor, datamodule)
                except ValueError as e:
                    pass
                predictor.freeze()
                trainer.test(predictor, datamodule=datamodule)


def load_data():
    # This is for load data LCL_12month.h5
    # df = pd.read_hdf('data/LCL_12month.h5').iloc[:,:50]
    # df = pd.read_hdf('data/LCL_12month.h5')

    # df_numpy = copy.deepcopy(df)
    # print(df_numpy[:5])
    # df.index = pd.to_datetime(df.index)


    ## This is for load data LCL_228houses.csv
    df = pd.read_csv('data/DataLCL_228houses_with_timeslot_temperature.csv')
    df.index = pd.to_datetime(df.index)

    time_index = df[['time_slot', 'weekday']].values
    temperature = df['Temperature'].values
    covariates = {'u': time_index}
    df = df.loc[:, ~df.columns.isin(['ds','time_slot', 'Temperature', 'weekday'])]
    



    df_down_sampling = df.resample('1D').sum()

    numpy_df = df_down_sampling.to_numpy().T
    
    def dtw_distance(ts_1, ts_2):
        return dtw(ts_1, ts_2, keep_internals=True).normalizedDistance

    distance_table = sd.squareform(sd.pdist(numpy_df, dtw_distance))

    adj = scipy.special.softmax(-distance_table, axis=1)
    print(adj.shape)
    # adj = np.zeros((100,100))
    # for i in range(100):
    #     adj[i,i] = 0
    rows, cols = np.nonzero(adj)
    # edge_index = np.vstack((rows, cols))
    edge_index = torch.vstack((torch.tensor(rows, dtype=torch.int64), torch.tensor(cols, dtype=torch.int64)))

    edge_weights = adj[rows, cols]


    torch_dataset = SpatioTemporalDataset(target=df,
                                            connectivity=(edge_index, edge_weights),
                                            # connectivity=(None, None),
                                            # covariates=covariates,
                                            horizon=24,
                                            window=48,
                                            stride=1)


    scalers = {'target': StandardScaler(axis=(0, 1))}

    # Split data sequentially:
    #   |------------ dataset -----------|
    #   |--- train ---|- val -|-- test --|
    splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=32,
    )
    
    metadata = {"n_channels": torch_dataset.n_channels,
                "horizon": torch_dataset.horizon,
                "n_nodes": torch_dataset.n_nodes,
                "window": torch_dataset.window}

    
    return df_down_sampling, df, dm, metadata

def main():
    df_down_sampling, df_numpy, dm, metadata = load_data()

    gw = GraphWaveNetModel(input_size=metadata['n_channels'],
                       horizon=metadata['horizon'],
                       output_size=metadata['n_channels'],
                       n_nodes=metadata['n_nodes']
                       )
    
    bipartite = BiPartiteSTGraphModel(
        input_size = metadata['n_channels'],
        horizon=metadata['horizon'],
        output_size=metadata['n_channels'],
        n_nodes=metadata['n_nodes'],
        input_window_size=metadata['window'],
        hidden_size = 128
    )

    agcrnn_model = AGCRNModel(input_size=metadata['n_channels'],
                              output_size=metadata['n_channels'],
                              horizon=metadata['horizon'],
                              n_nodes=metadata['n_nodes']
                              )
    
    grugcn_model = GRUGCNModel(
        input_size=metadata['n_channels'],
        hidden_size=128,
        output_size=metadata['n_channels'],
        horizon=metadata['horizon'],
        exog_size=0,
        enc_layers=1,
        gcn_layers=1
    )

    static_gts = StaticGTS(
        input_size=metadata['n_channels'],
        window=metadata['window'],
        horizon=metadata['horizon'],
        n_nodes=metadata['n_nodes'],
        hidden_size=64,
        nodes_features=torch.Tensor(np.array(df_down_sampling.T)).unsqueeze(1)
    )
    
    tgcnModel = TGCNModel(input_size=metadata['n_channels'],
                     horizon=metadata['horizon']
                     )

    gclstm = GraphConvLSTMModel(input_size=metadata['n_channels'],
                                horizon=metadata['horizon'],
                                )

    stegnn = STEGNN(input_size=metadata['n_channels'],
                    window=metadata['window'],
                    horizon=metadata['horizon'],
                    n_nodes=metadata['n_nodes'],
                    temporal_embedding_size=32)

    same_hour = SameHour(input_size=metadata['n_channels'],
                         window=metadata['window'],
                         horizon=metadata['horizon'],
                         n_nodes=metadata['n_nodes'])
    
    last_value = LastValue(input_size=metadata['n_channels'],
                         window=metadata['window'],
                         horizon=metadata['horizon'],
                         n_nodes=metadata['n_nodes'])

    var = VARModel(input_size=metadata['n_channels'],
                   temporal_order=4,
                   output_size=metadata['n_channels'],
                   horizon=metadata['horizon'],
                   n_nodes=metadata['n_nodes'])
    
    rnn = RNNModel(input_size=metadata['n_channels'],
                   output_size=metadata['n_channels'],
                   horizon=metadata['horizon'])

    stgraph = STGraph(models=[grugcn_model, rnn], datamodules=[dm])
    stgraph.run(config='config.yaml')

main()


