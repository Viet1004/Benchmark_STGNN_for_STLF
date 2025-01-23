import optuna
import subprocess
import argparse

def objective(trial):
    parser = argparse.ArgumentParser()

    list_of_models = ['last_value_model', 
                  'bipartite_model',   # 5m to run 
                  'rnn_model'           
                  'tf_model',
                  'agcrnn_model',      # 1 hour to train
                  'grugcn_model',      # 15m to run
                  'gcgru_model',  
                  'tgcn_model',           
                  'stegnn_model',         
                  'var_model', 
                  'rnn_model',                  # 5m to run
                  'gg_network_model',
                  'gw_model',]

    parser.add_argument("model", help=f"Model shoud be one of {list_of_models}")
    args = parser.parse_args()
    model = args.model

    assert model in list_of_models, f"Model should only be one of the models {list_of_models}"

    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16,32,64])
    model = 'bipartite_model'
    # model = 'rnn_model'
    exp = 'exp_11'
    window = trial.suggest_categorical('window', [48,96,144])
    hidden_dimension = trial.suggest_categorical('hidden_dimension', [64,80])
 
    cmd = [
        'python', 
        'SpatioTemporal_TS_with_Graph.py',
        model,
        exp,
        '--window', str(window),
        '--hidden_dimension', str(hidden_dimension),
        '--learning_rate', str(lr),
        '--batch_size', str(batch_size)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if 'MAE at validation dataset is' in line:
            mae_error = float(line.split(":")[-1].strip())
            return mae_error
    raise RuntimeError("MAE Error is not found")


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters:", study.best_params)
    print("Best validation Error:", study.best_value)