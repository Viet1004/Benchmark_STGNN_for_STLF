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

# Activate a virtual environment if needed
source ~/environments/tsl/bin/activate

# Run your script
srun python SpatioTemporal_TS_with_Graph.py

deactivate

# Print completion message
echo "Job finished at $(date)"
