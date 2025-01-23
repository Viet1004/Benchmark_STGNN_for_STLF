#!/bin/bash
#SBATCH --job-name=my_gpu-job           # Job name
#SBATCH --output=/home/users/qnguyen/Graph/output/output_%j.txt      # Standard output and error log (%j expands to jobID)
#SBATCH --error=/home/users/qnguyen/Graph/error/error_%j.txt        # Error log (%j expands to jobID)
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00     # Set job duration to 15 hours


# Activate a virtual environment if needed
source ~/environments/tsl/bin/activate


models_1=("last_value_model" "tf_model" "var_model" "rnn_model" "bipartite_model" "agcrnn_model" "gg_network_model" "gw_model" )
# models=("gg_network_model")
# models_1=()
methods=("correntropy" "pearson" "dtw" "euclidean")
# methods=("correntropy" "pearson" "dtw")
models_2=("tgcn_model_2" "tgcn_model" "grugcn_model")
experiment_ids=("exp_06_04" "exp_06_05")
# experiment_ids=("exp_03_02")
#### Note on naming convention of exp ####
# exp_05 experiment with fold one with 
# exp_06 experiment with fold two
# exp_07 experiment with fold three
# exp_11 with temperature

# Check if methods array is defined and non-empty
if [ -z "${methods+x}" ] || [ ${#methods[@]} -eq 0 ]; then
    method_flag=false
else
    method_flag=true
fi

# Loop through each experiment ID
for experiment_id in "${experiment_ids[@]}"; do 
    for model in "${models_1[@]}"; do
        echo "Running model: $model"
        
        # Skip methods loop if no methods are defined
            # Run without the --method argument
        srun --exclusive -N 1 -n 1 python SpatioTemporal_TS_with_Graph.py "$model" "$experiment_id" || \
        echo "Model $model without method failed"
    
    done
    
    for model in "${models_2[@]}"; do
        echo "Running model: $model"
        
        # Skip methods loop if no methods are defined
        if [ "$method_flag" = true ]; then
            for method in "${methods[@]}"; do
                srun --exclusive -N 1 -n 1 python SpatioTemporal_TS_with_Graph.py "$model" "$experiment_id" --method "$method" || \
                echo "Model $model with $method failed"
            done
        else
            # Run without the --method argument
            srun --exclusive -N 1 -n 1 python SpatioTemporal_TS_with_Graph.py "$model" "$experiment_id" || \
            echo "Model $model without method failed"
        fi
    done

done
# Run your script
# srun python SpatioTemporal_TS_with_Graph.py

deactivate

# Print completion message
echo "Job finished at $(date)"
