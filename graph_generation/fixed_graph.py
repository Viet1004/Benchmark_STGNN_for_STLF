from dtw import dtw
import scipy.spatial.distance as sd
import pandas as pd
from typing import Union
import numpy as np
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from pyinform.transferentropy import transfer_entropy
import tsl
import logging
import os


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def dtw_distance(ts_1, ts_2):
        return dtw(ts_1, ts_2, keep_internals=True).normalizedDistance

def save_or_load_adj_from_file(method_name):
    """Decorator to check if adjacency matrix exists in a file; if so, load it, otherwise save it after calculation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Define the filename based on the method name
            filename = f"graph_generation/cache/{method_name}_adjacency_matrix.npy"
            # Check if the file already exists
            if os.path.exists(filename):
                # Load the adjacency matrix from the file
                adj = np.load(filename)
                print(f"Loaded adjacency matrix from {filename}")
            else:
                # Call the original function to calculate the adjacency matrix
                adj = func(*args, **kwargs)
                # Save the adjacency matrix to a file
                np.save(filename, adj)
                print(f"Calculated and saved adjacency matrix to {filename}")
            return adj
        return wrapper
    return decorator

class AdjacencyMatrixGenerator():
    eps = 1e-8

    def __init__(self, data_array: Union[pd.DataFrame, np.ndarray], resample_rate='1D'):
        """
        data_array: time series of shape times * sample
        """
        
        if type(data_array) == pd.DataFrame:
            
            df_down_sampling = data_array.resample(resample_rate).sum()
            # print(f'Shape of data array before: {df_down_sampling.shape}')
            numpy_df = np.array(df_down_sampling)
            # print(f'Shape of data array after: {numpy_df.shape}')
        
        elif type(data_array) == np.ndarray:
            numpy_df = data_array
        
        else:
            raise TypeError(f"The data_array should be of form pd.DataFrame or np.ndarray, but receive {type(data_array)}")
        self.data_array = numpy_df
        # print(f'Shape of data array: {self.data_array.shape}')
        
    def calculate(self, method='dtw', config = None):
        
        if method == "dtw":
            adj = AdjacencyMatrixGenerator._dtw(self.data_array)

        if method == "euclidean":
            adj = AdjacencyMatrixGenerator._euclidean(self.data_array)

        if method == "correntropy":
            adj = AdjacencyMatrixGenerator._correntropy(self.data_array)

        if method == "pearson":
            adj = AdjacencyMatrixGenerator._person_correlation(self.data_array)

        if method == 'transfer_entropy':
            adj = AdjacencyMatrixGenerator._transfer_entropy(self.data_array)

        logger.info(f"Shape of adjacency matrix is: {adj.shape}")

        return adj

    @staticmethod
    @save_or_load_adj_from_file("dtw")
    def _dtw(data_array):
        distance_table = sd.squareform(sd.pdist(data_array.T, dtw_distance))
        adj = np.where(distance_table < 3, np.exp(-distance_table/3), 0)
        
        return adj

    @staticmethod
    @save_or_load_adj_from_file("euclidean")
    def _euclidean(data_array):
        distance_table = euclidean_distances(data_array.T)
        adj = np.where(distance_table < 200, np.exp(-distance_table/200), 0)
        return adj

    @staticmethod
    @save_or_load_adj_from_file("correntropy")
    def _correntropy(data_array):

        # sim = np.zeros((data_array.shape[1], data_array.shape[1]))
        # tot = np.zeros_like(sim)
        # for i in range(period, len(data_array), period):
        #     xi = data_array[i - period:i].T
        #     m = mask[i - period:i].min(0)
        #     si = rbf_kernel(xi, gamma=gamma)
        #     m = m * m.T
        #     si = si * m
        #     sim += si
        #     tot += m

        n_samples = data_array.shape[1]
        sigma = 2
        def correntropy(x, y, sigma):
            return np.mean(np.exp(-((x - y) ** 2) / (2 * sigma ** 2)))

        # Calculate correntropy matrix for every pair of houses
        adj = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):  # Use symmetry to reduce computation
                corr_value = correntropy(data_array[:, i], data_array[:, j], sigma)
                adj[i, j] = corr_value
                adj[j, i] = corr_value  # Symmetric value
                # Since adj[i, j] = adj[j, i] by symmetry, make it symmetric
        adj = np.where(adj < 0.2, 0, 1)
        return adj
            
    @staticmethod
    @save_or_load_adj_from_file("pearson")
    def _person_correlation(data_array):
        adj = np.corrcoef(data_array.T)
        adj = np.where(adj < 0.2, 0, adj)
        return adj
    
    @staticmethod
    @save_or_load_adj_from_file("transfer_entropy")
    def _transfer_entropy(data_array):
        print(f'Shape of data array:{data_array.shape}')
        n_samples = data_array.shape[1]
        adj = np.zeros(shape=(n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                adj[i,j] = transfer_entropy(data_array[i], data_array[j], k = 2)
        adj = np.where(adj < 0.2, 0, 1)
        return adj 
            

