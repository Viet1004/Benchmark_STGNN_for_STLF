#!/bin/bash
#SBATCH --job-name=my_job           # Job name
#SBATCH --output=/home/users/qnguyen/Graph/output/output_%j.txt      # Standard output and error log (%j expands to jobID)
#SBATCH --error=/home/users/qnguyen/Graph/error/error_%j.txt        # Error log (%j expands to jobID)


#SBATCH -N 1
#SBATCH --ntasks-per-node 1  # Optimized for 1 full node of aion
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G


# Activate a virtual environment if needed
source ~/environments/tsl/bin/activate

# Run your script
srun -c 16 python /home/users/qnguyen/Graph/SpatioTemporal_TS_with_Graph.py

deactivate

# Print completion message
echo "Job finished at $(date)"
