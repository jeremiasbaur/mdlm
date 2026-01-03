#!/bin/bash
#SBATCH --job-name=train_mdlm
#SBATCH -D /cluster/raid/home/jbaur/projects/mdlm/
#SBATCH --get-user-env
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jdframes0@gmail.com
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3090:4
#SBATCH --time=6-23:59:59
#SBATCH --output=notebook_%j.log
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/data/jbaur

export HYDRA_FULL_ERROR=1

# Load Anaconda
source /cluster/software/anaconda3/etc/profile.d/conda.sh
# Activate your Conda environment
conda activate /cluster/raid/home/jbaur/.conda/envs/mdlm

echo "Running on node: $(hostname)"
echo "Allocated GPUs:"
nvidia-smi
which python
# 119378
python -u -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  trainer.accumulate_grad_batches=8 \
  model=tiny \
  model.init.type=unitary_default \
  model.init.tie_weights=False \
  model.init.PE=RoPE \
  training.clipped_sampling=True \
  training.clip_beta=0.1 \
  training.clip_omega=0.2 \
  data=openwebtext-split \
  data.cache_dir=/cluster/raid/home/jbaur/projects/mdlm/data_cache \
  wandb.name=mdlm-owt-tiny-init-unitary_default-RoPE-expanding \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 \
  trainer.max_steps=300000 \
  lr_scheduler=warmup_const_cosine_decay \
  lr_scheduler.first_decay_steps=60000 \
  lr_scheduler.final_factor=1.0
