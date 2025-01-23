
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
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
from tsl.metrics.torch.metric_base import MaskedMetric
from torchmetrics import MeanSquaredLogError
from tsl.engines import Predictor
from tsl.data.datamodule import (SpatioTemporalDataModule,
                                 TemporalSplitter,
                                 AtTimeStepSplitter)
from tsl.metrics.torch.metric_base import convert_to_masked_metric

from custom_models import BiPartiteSTGraphModel, StaticGTS, STEGNN, TGCNModel, GraphConvLSTMModel, SameHour, LastValue, TGCNModel_2
from graph_generation import AdjacencyMatrixGenerator
from tsl.nn.utils import get_layer_activation, maybe_cat_exog
import argparse
from tsl.data.preprocessing import StandardScaler, MinMaxScaler
import yaml
import torch.nn as nn
from typing import Any
from datetime import datetime

from einops import rearrange
from einops.layers.torch import Rearrange


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from tsl.nn.models import BaseModel
from tsl.nn.utils import get_layer_activation, maybe_cat_exog
from tsl.nn.models.stgn.graph_wavenet_model import GraphWaveNetModel
from tsl.nn.models.stgn.gated_gn_model import GatedGraphNetworkModel
from tsl.nn.models.stgn.gru_gcn_model import GRUGCNModel
from tsl.nn.models.stgn.agcrn_model import AGCRNModel
from tsl.nn.models.temporal.linear_models import VARModel
from tsl.nn.models.temporal.rnn_model import RNNModel
from tsl.nn.models.temporal.transformer_model import TransformerModel 
import matplotlib.pyplot as plt
from tools import plot_time_series, plot_adj_heatmap

# Define a TorchTrainer without hyper-parameters for Tuner

parser = argparse.ArgumentParser(description='Model to use in an experiment.')
parser.add_argument('model', help='Model')
parser.add_argument('experiment_id', help='Id of experiment')
parser.add_argument('--method', default=None, help='Model with default parameter in the script.')
parser.add_argument('--window', default=48, type=int,  help='Window as input for model.')
parser.add_argument('--hidden_dimension', default=64, type=int, help='hidden dimension fore neural network model.')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate for training.')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size of data')
args = parser.parse_args()


W_window = args.window
hidden_dimension = args.hidden_dimension
learning_rate = args.learning_rate
batch_size = args.batch_size

print(f"tsl version  : {tsl.__version__}")
print(f"torch version: {torch.__version__}")

pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(edgeitems=3, precision=3)
torch.set_printoptions(edgeitems=2, precision=3)


config = 'config.yaml'
###########


with open(config, 'r') as file:
    config = yaml.safe_load(file)

exogeneous_added = config['exogeneous']
max_epochs = config['max_epochs']
devices = config['devices']
limit_val_batches = config['limit_val_batches']
fold = config['fold']        
# batch_size = config['batch_size']



# Utility functions ################
def print_matrix(matrix):
    return pd.DataFrame(matrix)

def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

class MaskedMeanSquaredLogError(MaskedMetric):
    """Mean Absolute Error Metric.

    Args:
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
         time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, mask_nans=False, mask_inf=False, at=None, **kwargs: Any):
        super(MaskedMeanSquaredLogError, self).__init__(
            metric_fn=MeanSquaredLogError(),
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            metric_fn_kwargs={'reduction': 'none'},
            at=at,
            **kwargs,
        )


class SaveAdjMatrix(pl.callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Check if the current epoch is a multiple of 5
        # if (trainer.current_epoch + 1) % 5 == 0:
            # print(f"Epoch {trainer.current_epoch + 1} has ended.")
            # Execute custom logic
            self.save_adj_matrix(trainer, pl_module)
    
    def save_adj_matrix(self, trainer, pl_module):
        print(pl_module.__class__.__name__)
        # Check if the model has the 'get_adj_matrix' method
        if hasattr(pl_module.model, 'get_learned_adj') and callable(getattr(pl_module.model, 'get_learned_adj')):
            # Execute the 'get_adj_matrix' method
            adj_matrix = pl_module.model.get_learned_adj()
            # print("Adjacency matrix obtained from get_learned_adj method:", adj_matrix)
        else:
            pass
            # print("The model does not have a callable 'get_learned_adj' method.")



class STGraph:
    def __init__(self, model: BaseModel, datamodule: SpatioTemporalDataModule, loss_fn = None, metrics = None):
        self.model = model
        self.datamodule = datamodule
        
        if loss_fn:
            self.loss_fn = loss_fn
        else: 
            self.loss_fn = MaskedMAE()
            # msle = convert_to_masked_metric(MeanSquaredLogError)
            # self.loss_fn = MaskedMeanSquaredLogError()
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

    # def tune():
    
    def run(self, experiment_id, method = None):

        ########### Change
        # scheduler = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)



        if torch.cuda.is_available():
            accelerator = 'gpu'
        # elif torch.backends.mps.is_available():
        #     accelerator = 'mps'
        else:
            accelerator = 'cpu'


        logger = TensorBoardLogger(save_dir="logs", name=self.model.__class__.__name__, version=0)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoint/{self.model.__class__.__name__}',
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
            limit_val_batches=limit_val_batches,
            check_val_every_n_epoch=5,
            # strategy=RayDDPStrategy(),
            # strategy='ddp',
            callbacks = [checkpoint_callback, save_adj, early_stopping],
            # callbacks = [checkpoint_callback, save_adj, early_stopping, RayTrainReportCallback()],
            enable_progress_bar=False
            # callbacks = [checkpoint_callback]
        )

        ######## Change
        # trainer = prepare_trainer(trainer)
        ########

        predictor = Predictor(
            model=self.model,
            optim_class=torch.optim.Adam,  # specify optimizer to be used...
            optim_kwargs={'lr': learning_rate},    # ...and parameters for its initialization
            loss_fn=self.loss_fn,               # which loss function to be used
            metrics=self.metrics                # metrics to be logged during train/val/test
        )
        ########## Change
        # def train_func():
        ##########
        try:
            trainer.fit(predictor, self.datamodule)
        except ValueError as e:
            print(e)
        
        # predictor.load_model(checkpoint_callback.best_model_path)
        validation_result = trainer.validate(predictor, self.datamodule, ckpt_path='best')
        # print(validation_result)
        validation_mae = validation_result[0]["val_mae"]

        predictor.freeze()
        
        def record(predictions):
            # predictions = trainer.predict(predictor, dataloaders=self.datamodule.test_dataloader())
            y_hat_col = []
            y_col = []
            for i, prediction in enumerate(predictions):
                y_hat = rearrange(prediction['y_hat'], 'b t n f -> (b t) (n f)')
                y = rearrange(prediction['y'], 'b t n f -> (b t) (n f)')
                y_hat_col.append(y_hat)
                y_col.append(y)
            y_hat_flat = torch.cat(y_hat_col, dim=0)
            y_flat = torch.cat(y_col, dim=0)
            return y_hat_flat, y_flat

        try:
            trainer.test(predictor, datamodule=self.datamodule, ckpt_path='best')
            predictions = trainer.predict(predictor, dataloaders=self.datamodule.test_dataloader(), ckpt_path='best')
            
        except ValueError as e:
            print("Model has no trainable parameters")
            trainer.test(predictor, datamodule=self.datamodule)
            predictions = trainer.predict(predictor, dataloaders=self.datamodule.test_dataloader())
        y_hat, y = record(predictions)
        y_hat, y = y_hat.cpu().numpy(), y.cpu().numpy()
        plot_time_series(y_hat[:96,0], y[:96,0], self.model.__class__.__name__)
        if method != None:
            os.makedirs(f'save_inference_result/{experiment_id}/{method}', exist_ok=True)
            np.save(f'save_inference_result/{experiment_id}/{method}/y_hat_{self.model.__class__.__name__}.npy', y_hat)
            np.save(f'save_inference_result/{experiment_id}/{method}/y.npy', y)
        else:
            os.makedirs(f'save_inference_result/{experiment_id}/', exist_ok=True)
            np.save(f'save_inference_result/{experiment_id}/y_hat_{self.model.__class__.__name__}.npy', y_hat)
            np.save(f'save_inference_result/{experiment_id}/y.npy', y)
                    
        return validation_mae




def load_data(method):

    df = pd.read_csv('data/DataLCL_228houses_with_timeslot_temperature.csv')
    
    # df.index = pd.to_datetime(df['ds'])

    # time_index = df[['time_slot', 'weekday']].values
    
    # temperature = np.expand_dims(temperature, -1)

    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S')
    # df = df.loc[df['ds'] < datetime(2013,12,30,23,59,59)]
    df.set_index('ds', inplace=True)
    temperature = pd.DataFrame(df['Temperature'])
    df = df.loc[:, ~df.columns.isin(['time_slot', 'Temperature', 'weekday'])]
    n_nodes = df.shape[1]
    if exogeneous_added:
        covariates = {'u': temperature}
    else:
        covariates = None    
    # df = df.iloc[:,:20]
    #This is for 1000 household
    # df = pd.read_parquet('/home/users/qnguyen/Graph/data/dataframe_not_anonymized.parquet')
    # df = df.pivot(index='ds', columns='unique_id', values='y')
    
    # df.index = pd.to_datetime(df['ds'])

    edge_index = edge_index = np.vstack((np.arange(n_nodes), np.arange(n_nodes)))
    edge_weights = np.ones(n_nodes)

    if method != None:

        adj_gen = AdjacencyMatrixGenerator(df)

        adj = adj_gen.calculate(method=method)

        plot_adj_heatmap(adj, f'visualization/adj_heatmap/{method}.pdf')

        rows, cols = np.nonzero(adj)
        edge_index = np.vstack((rows, cols))
        edge_weights = adj[rows, cols]


    torch_dataset = SpatioTemporalDataset(target=df,
                                            connectivity=(edge_index, edge_weights),
                                            # connectivity=(None, None),
                                            covariates=covariates,
                                            horizon=48,
                                            window=W_window,
                                            stride=48)


    scalers = {'target': StandardScaler(axis=(0, 1))}

    # Split data sequentially:
    #   |------------ dataset -----------|
    #   |--- train ---|- val -|-- test --|
    # splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

    timestamps_config = {
        'fold_1': {
            'first_val': datetime(2013,7,1,0,0,0),
            'last_val': datetime(2013,8,1,0,0,0),
            'first_test': datetime(2013,8,1,0,0,0),
            'last_test': datetime(2013,8,31,23,59,59)
        },
        'fold_2': {
            'first_val': datetime(2013,9,1,0,0,0),
            'last_val': datetime(2013,10,1,0,0,0),
            'first_test': datetime(2013,10,1,0,0,0),
            'last_test': datetime(2013,10,31,23,59,59)
        },
        'fold_3': {
            'first_val': datetime(2013,11,1,0,0,0),
            'last_val': datetime(2013,12,1,0,0,0),
            'first_test': datetime(2013,12,1,0,0,0),
            'last_test': datetime(2013,12,31,23,59,59)
        }
    }
    

    splitter =  AtTimeStepSplitter(
        first_val_ts = timestamps_config[fold]['first_val'], 
        first_test_ts = timestamps_config[fold]['first_test'],
        last_val_ts = timestamps_config[fold]['last_val'],
        last_test_ts = timestamps_config[fold]['last_test']
    )


    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
    )
    
    metadata = {"n_channels": torch_dataset.n_channels,
                "horizon": torch_dataset.horizon,
                "n_nodes": torch_dataset.n_nodes,
                "window": torch_dataset.window}

    
    return df, dm, metadata



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chosen_model = args.model
    chosen_graph_creation_method = args.method
    experiment_id = args.experiment_id
    print(f'The chosen model is {chosen_model}')
    df_numpy, dm, metadata = load_data(chosen_graph_creation_method)
    # exogeneous_added = config['exogeneous']
    if exogeneous_added: 
        exog_size = 1
    else: 
        exog_size = 0
    gw_model = GraphWaveNetModel(input_size=metadata['n_channels'],
                       horizon=metadata['horizon'],
                       output_size=metadata['n_channels'],
                       n_nodes=metadata['n_nodes']
                       )
    
    bipartite_model = BiPartiteSTGraphModel(
        input_size = metadata['n_channels'],
        horizon=metadata['horizon'],
        output_size=metadata['n_channels'],
        n_nodes=metadata['n_nodes'],
        input_window_size=metadata['window'],
        hidden_size = hidden_dimension,
        exog_size=exog_size
    )

    gg_network_model = GatedGraphNetworkModel(
        input_size=metadata['n_channels'],
        input_window_size=metadata['window'],
        horizon=metadata['horizon'],
        n_nodes=metadata['n_nodes'],
        hidden_size=hidden_dimension,
        exog_size=exog_size
    )

    agcrnn_model = AGCRNModel(input_size=metadata['n_channels'],
                              output_size=metadata['n_channels'],
                              horizon=metadata['horizon'],
                              n_nodes=metadata['n_nodes']
                              )
    
    grugcn_model = GRUGCNModel(
        input_size=metadata['n_channels'],
        hidden_size=hidden_dimension,
        output_size=metadata['n_channels'],
        horizon=metadata['horizon'],
        exog_size=exog_size,
        enc_layers=1,
        gcn_layers=1
    )
    
    tgcn_model = TGCNModel(input_size=metadata['n_channels'],
                     horizon=metadata['horizon'],
                     exog_size=exog_size
                     )

    tgcn_model_2 = TGCNModel_2(input_size=metadata['n_channels'],
                               horizon=metadata['horizon'],
                               exog_size=exog_size)


    stegnn_model = STEGNN(input_size=metadata['n_channels'],
                    window=metadata['window'],
                    horizon=metadata['horizon'],
                    n_nodes=metadata['n_nodes'],
                    temporal_embedding_size=32
                    )

    same_hour_model = SameHour(input_size=metadata['n_channels'],
                         window=metadata['window'],
                         horizon=metadata['horizon'],
                         n_nodes=metadata['n_nodes'])
    
    last_value_model = LastValue(input_size=metadata['n_channels'],
                         window=metadata['window'],
                         horizon=metadata['horizon'],
                         n_nodes=metadata['n_nodes'])

    var_model = VARModel(input_size=metadata['n_channels'],
                   temporal_order=4,
                   output_size=metadata['n_channels'],
                   horizon=metadata['horizon'],
                   n_nodes=metadata['n_nodes'])
    
    rnn_model = RNNModel(input_size=metadata['n_channels'],
                   output_size=metadata['n_channels'],
                   horizon=metadata['horizon'],
                   hidden_size=hidden_dimension,
                   rec_layers=3)

    tf_model = TransformerModel(input_size=metadata['n_channels'],
                   output_size=metadata['n_channels'],
                   horizon=metadata['horizon'])

    model_dict = {'gw_model': gw_model, 
                  'bipartite_model': bipartite_model,   # 5m to run 
                  'rnn_model': rnn_model,           
                  'agcrnn_model': agcrnn_model,      # 1 hour to train
                  'grugcn_model': grugcn_model,      # 15m to run
                #   'static_gts_model': static_gts_model, 
                  'tgcn_model': tgcn_model,  
                  'tgcn_model_2': tgcn_model_2,           
                  'stegnn_model': stegnn_model,         # 10m to run
                  'same_hour_model': same_hour_model,     
                  'last_value_model': last_value_model, 
                  'var_model': var_model, 
                  'rnn_model': rnn_model,                  # 5m to run
                  'tf_model': tf_model,
                  'gg_network_model': gg_network_model
                }
    _model_ = model_dict[chosen_model]

    stgraph = STGraph(model=_model_, datamodule=dm)
    validation_mae = stgraph.run(experiment_id=experiment_id, method=chosen_graph_creation_method)
    print(f"MAE at validation dataset is: {validation_mae}")


main()


