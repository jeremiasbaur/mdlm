#!/bin/bash
#SBATCH -J T_mdlm                     # Job name
#SBATCH -D /cluster/raid/home/jbaur/projects/mdlm/
#SBATCH --get-user-env
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jdframes0@gmail.com
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3090:4
#SBATCH --time=4-23:59:59
#SBATCH --output=notebook_%j.log
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/data/jbaur

checkpoint_path=/cluster/raid/home/jbaur/projects/mdlm/outputs/openwebtext-train/2025.12.11/215109/checkpoints/5-102000.ckpt

export HYDRA_FULL_ERROR=1

# Load Anaconda
source /cluster/software/anaconda3/etc/profile.d/conda.sh
# Activate your Conda environment
conda activate /cluster/raid/home/jbaur/.conda/envs/mdlm

echo "Running on node: $(hostname)"
echo "Allocated GPUs:"
nvidia-smi
which python

for T in 0 1000; do
  echo "$T"
  srun python -u -m main \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    mode=ppl_eval \
    model=tiny \
    model.init.type=symmetric \
    model.init.tie_weights=False \
    model.init.PE=RoPE \
    training.clipped_sampling=True \
    training.clip_beta=0.1 \
    training.clip_omega=0.2 \
    data=wikitext2 \
    parameterization=subs \
    model.length=1024 \
    eval.compute_generative_perplexity=True \
    T="$T" \
    eval.checkpoint_path=$checkpoint_path \
    +wandb.offline=true
done