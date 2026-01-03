#!/bin/bash
#SBATCH --job-name=notebook_vscode_jbaur
#SBATCH -D /cluster/raid/home/jbaur/projects/mdlm/
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jebaur@ethz.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx3090:4
#SBATCH --time=06:00:00
#SBATCH --output=notebook_%j.log
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/data/jbaur

# Load Anaconda
export PATH="/cluster/software/anaconda3/bin:$PATH"

source /cluster/software/anaconda3/etc/profile.d/conda.sh

# Activate your Conda environment
source activate /cluster/raid/home/jbaur/.conda/envs/mdlm # att312

# Random port
PORT=$(shuf -i 8000-9999 -n 1)

echo "Allocated GPUs:"
nvidia-smi
which python

echo "--------------------------------------------------------"
echo " Job running on: $(hostname)"
echo " Forward this port from your local machine:"
echo " ssh -L ${PORT}:localhost:${PORT} jbaur@grewemaster1.ethz.ch"
echo "--------------------------------------------------------"
echo "Then connect VS Code to: http://localhost:${PORT}"
echo "--------------------------------------------------------"

jupyter notebook --no-browser --port=${PORT} --ip=0.0.0.0
