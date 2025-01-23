# Benchmark of Spatiotemporal Graph Neural Networks for Short-Term Load Forecasting

## Introduction
Short-term load forecasting (STLF) plays a critical role in efficient energy management and grid operations. Spatiotemporal Graph Neural Networks (STGNNs) have shown promise in capturing spatial and temporal dependencies within power grid systems.  
This benchmark evaluates the performance of various STGNN architectures on STLF tasks using a subset of [Low Carbon London dataset](https://data.london.gov.uk/dataset/low-carbon-london-electric-vehicle-load-profiles) and metrics. The aim is to establish a foundation for comparing models and guiding future research.

---

## Data
### Datasets
We use the following datasets for the benchmark:
**Electricity Load Dataset**  
   Description: Hourly energy consumption data for different nodes in a power grid.  
   Source: [Low Carbon London dataset](https://data.london.gov.uk/dataset/low-carbon-london-electric-vehicle-load-profiles).
   Data_directory: `DataLCL_228houses_with_timeslot_temperature.csv`

   Format: CSV with columns: `time`, `$smart_meter_id` (228 values) 
### Preprocessing
- **Normalization**: Load and weather data normalized using `Z-score Normalization` scaling.  
- **Temporal Binning**: Aggregate data into 15-minute or hourly bins as required.  
- **Graph Construction**:  
  - Nodes: Each household.  
  - Edges: Based on correlation threshold or learnable parameters during training.  

---

## Models to Examine

### Overview:
| Models         | Predefined Graph | Learnable Graph | TTS          | T&S          |
|----------------|------------------|-----------------|--------------|--------------|
| [GRUGCN](https://arxiv.org/abs/2103.07016)     | ✅               |                 | ✅           |              |
| [GCGRU](https://www.sciencedirect.com/science/article/pii/S0952197622003761)      | ✅               |                 |              | ✅           |
| [T-GCN](https://www.sciencedirect.com/science/article/pii/S0142061522006470)      | ✅               |                 |              | ✅           |
| [AGCRN](ttps://proceedings.neurips.cc/paper/2020/hash/ce1aad92b939420fc17005e5461e6f48-Abstract.html)      |                  | ✅              |              | ✅           |
| [GraphWavenet](https://ieeexplore-ieee-org.proxy.bnl.lu/document/9468666?arnumber=9468666)|                  | ✅              | ✅           |              |
|  [FC-GNN](https://arxiv.org/abs/2203.03423)   |                  | ✅              | ✅           |              |
| [BP-GNN](https://arxiv.org/abs/2203.03423)  |                  | ✅              | ✅           |              |

##### **TTS**: Time-then-Space
##### **T&S**: Time-and-Space
---

## Benchmarking Process
### Evaluation Metrics
1. **Mean Absolute Error (MAE)**  
2. **Root Mean Squared Error (RMSE)**  
3. **Mean Absolute Percentage Error (MAPE)**  

### Baselines
- **SeasonalNaive**: Uses the value of the previous day (same hour) as the forecast for the next day.  
- **VAR**: Auto-Regressive Integrated Moving Average.
- **GRU**: Gated Recurrent Units
- **Transformer**: Transformer
---

## Command Line for Training Models
Below are example placeholders for training commands:

### General Training Command
```bash
python SpatioTemporal_TS_with_Graph.py <MODEL_NAME> \
                       <EXP>  \  \# experiment_id to save forecast on test
                      --method <METHOD> \  \# Method to generate graph from similarity function, could be either euclidean, dtw, pearson, correntropy \
                      --window <WINDOW> \   \# Window of historical result
                      --hidden_dimension <HIDDEN_DIMENSION>  \     \# Number of hidden dimension for neural network (Look at model architecture in \custome_models)
                      --learning_rate <LEARNING_RATE> \
                      --batch_size <BATCH_SIZE> \

```

### Hyperparamter tuning
```bash
python hyperparameter_tuning.py <MODEL>   \# see help for possible parameter for <MODEL>
```


                      