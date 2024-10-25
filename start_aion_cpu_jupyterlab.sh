#!/usr/bin/bash --login
#SBATCH --job-name=Jupyter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128 # Change accordingly, note that ~1.7GB RAM is proivisioned per core
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --output=/dev/null #%x_%j.out  # Print messages to 'Jupyter_<job id>.out
#SBATCH --error=/dev/null #%x_%j.err   # Print debug messages to 'Jupyter_<job id>.err
#SBATCH --time=0-10:00:00   # Change maximum allowable jupyter server uptime here

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"

# Load the default Python 3 module
module load lang/Python
source "${HOME}/environments/jupyter_env/bin/activate"

declare port="8888"
declare connection_instructions="connection_instructions.log"

jupyter lab --ip=$(hostname -i) --port=${port} --no-browser &
declare lab_pid=$!

# Add connection instruction
echo "# Connection instructions" > "${connection_instructions}"
echo "" >> "${connection_instructions}"
echo "To access the jupyter notebook execute on your personal machine:" >> "${connection_instructions}"
echo "ssh -J ${USER}@access-${ULHPC_CLUSTER}.uni.lu:8022 -L ${port}:$(hostname -i):${port} ${USER}@$(hostname -i)" >> "${connection_instructions}"
echo "" >> "${connection_instructions}"
echo "To access the jupyter notebook if you have setup a key to connect to cluster nodes execute on your personal machine:" >> "${connection_instructions}"
echo "ssh -i ~/.ssh/hpc_id_ed25519 -J ${USER}@access-${ULHPC_CLUSTER}.uni.lu:8022 -L ${port}:$(hostname -i):${port} ${USER}@$(hostname -i)" >> "${connection_instructions}"
echo "" >> "${connection_instructions}"
echo "Then navigate to:" >> "${connection_instructions}"

# Wait for the server to start
sleep 2s
# Wait and check that the landing page is available
curl \
    --connect-timeout 10 \
    --retry 5 \
    --retry-delay 1 \
    --retry-connrefused \
    --silent --show-error --fail \
    "http://$(hostname -i):${port}" > /dev/null
# Note down the URL
jupyter lab list 2>&1 \
    | grep -E '\?token=' \
    | awk 'BEGIN {FS="::"} {gsub("[ \t]*","",$1); print $1}' \
    | sed -r 's/([0-9]{1,3}\.){3}[0-9]{1,3}/127\.0\.0\.1/g' \
    >> "${connection_instructions}"

# Save some debug information

echo -e '\n===\n'

echo "AVAILABLE LABS"
echo ""
jupyter lab list

echo -e '\n===\n'

echo "CONFIGURATION PATHS"
echo ""
jupyter --paths

echo -e '\n===\n'

echo "KERNEL SPECIFICATIONS"
echo ""
jupyter kernelspec list

# Wait for the user to terminate the lab

wait ${lab_pid}

