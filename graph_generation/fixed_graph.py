from dtw import dtw
import scipy.spatial.distance as sd
import pandas as pd
from typing import Union
import numpy as np
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
import tsl
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def dtw_distance(ts_1, ts_2):
        return dtw(ts_1, ts_2, keep_internals=True).normalizedDistance

class AdjacencyMatrix():
    eps = 1e-8

    def __init__(self, data_array: Union[pd.DataFrame, np.ndarray], resample_rate='1D'):
        """
        data_array: time series of shape times * sample
        """
        
        if type(data_array) == pd.DataFrame:
            df_down_sampling = data_array.resample(resample_rate).sum()
            numpy_df = np.array(df_down_sampling)
        
        elif type(data_array) == np.ndarray:
            numpy_df = data_array
        
        else:
            raise TypeError(f"The data_array should be of form pd.DataFrame or np.ndarray, but receive {type(data_array)}")
        self.data_array = numpy_df
        
    def calculate(self, method='dtw', config = None):
        
        if method == "dtw":
            distance_table = sd.squareform(sd.pdist(self.data_array, dtw_distance))
            adj = scipy.special.softmax(-distance_table, axis=1)

        if method == "euclidean":
            adj = scipy.special.softmax(-euclidean_distances(self.numpy_df), axis=1)

        if method == "correntropy":
            if config != None:
                period = config["period"]
            else:
                period = 7
            adj = AdjacencyMatrix._correntropy(self.data_array, period=period)

        if method == "pearson":
            adj = AdjacencyMatrix._person_correlation(self.data_array)

        logger.info(f"Shape of adjacency matrix is: {adj.shape}")

        rows, cols = np.nonzero(adj)
        edge_index = np.vstack((rows, cols))
        edge_weights = adj[rows, cols]

        return edge_index, edge_weights

    @staticmethod
    def _correntropy(x, period, gamma = 0.05):
        if mask is None:
            mask = 1 - np.isnan(x, dtype='uint8')
            mask = mask[..., None]

        sim = np.zeros((x.shape[1], x.shape[1]))
        tot = np.zeros_like(sim)
        for i in range(period, len(x), period):
            xi = x[i - period:i].T
            m = mask[i - period:i].min(0)
            si = rbf_kernel(xi, gamma=gamma)
            m = m * m.T
            si = si * m
            sim += si
            tot += m
        return sim / (tot + AdjacencyMatrix.eps)
    
    @staticmethod
    def _person_correlation(data_array):
        norms = np.linalg.norm(data_array, axis=1)
        normalized_data_array = data_array - data_array.mean(1, keepdims=True) / norms
        n_samples = data_array.shape[0]
        adj = np.zeros(shape=(n_samples, n_samples))
        for i in range(n_samples):
            adj[i,i+1:] = normalized_data_array[i] @ normalized_data_array[i+1:] \
                    / (normalized_data_array[i] * normalized_data_array[i+1:] + AdjacencyMatrix.eps)
        return adj
