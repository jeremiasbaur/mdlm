#!/bin/bash
#SBATCH -J eval_mdlm                     # Job name
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

export HYDRA_FULL_ERROR=1

# Load Anaconda
source /cluster/software/anaconda3/etc/profile.d/conda.sh
# Activate your Conda environment
conda activate /cluster/raid/home/jbaur/.conda/envs/mdlm

echo "Running on node: $(hostname)"
echo "Allocated GPUs:"
nvidia-smi
which python

SAVE_DIR="/cluster/raid/home/jbaur/projects/attention-geometry-extended/_results/custom-models/diffusion-text/validation_runs"

for EVAL_DATASET in "openwebtext-split"; do #"lm1b-gpt2" "wikitext103" "scientific_papers_arxiv" "scientific_papers_pubmed"; do
    echo "Evaluating on dataset: $EVAL_DATASET"
    for CKPT in "11-200000.ckpt"; do #"5-102000.ckpt" "8-150000.ckpt" "10-175000.ckpt" "11-200000.ckpt"; do
        echo "Evaluating checkpoint: $CKPT"

        # mdlm-tiny-init-default:
        python -u -m automatic_validation_script \
            --model_dir "/data/jbaur/openwebtext-train/2025.11.03/164805" \
            --ckpt "$CKPT" \
            --eval_dataset "$EVAL_DATASET" \
            --save_dir "$SAVE_DIR"

        # mdlm-tiny-init-symm:
        python -u -m automatic_validation_script \
            --model_dir "/data/jbaur/openwebtext-train/2025.11.03/164407" \
            --ckpt "$CKPT" \
            --eval_dataset "$EVAL_DATASET" \
            --save_dir "$SAVE_DIR"
            
        # # default init, CL, LR:
        python -u -m automatic_validation_script \
            --model_dir "./outputs/openwebtext-train/2025.12.11/215109" \
            --ckpt "$CKPT" \
            --eval_dataset "$EVAL_DATASET" \
            --save_dir "$SAVE_DIR"

        # symmetric init, CL, LR:
        python -u -m automatic_validation_script \
            --model_dir "./outputs/openwebtext-train/2025.12.11/215139" \
            --ckpt "$CKPT" \
            --eval_dataset "$EVAL_DATASET" \
            --save_dir "$SAVE_DIR"  
        
        # default, WT, LPE
        python -u -m automatic_validation_script \
            --model_dir "./outputs/openwebtext-train/2025.12.17/233015" \
            --ckpt "$CKPT" \
            --eval_dataset "$EVAL_DATASET" \
            --save_dir "$SAVE_DIR" 
        
        # symmetric, WT, LPE
        python -u -m automatic_validation_script \
            --model_dir "./outputs/openwebtext-train/2025.12.17/233050" \
            --ckpt "$CKPT" \
            --eval_dataset "$EVAL_DATASET" \
            --save_dir "$SAVE_DIR" 

        # unitary default, RoPE
        python -u -m automatic_validation_script \
            --model_dir "./outputs/openwebtext-train/2025.12.25/105952" \
            --ckpt "$CKPT" \
            --eval_dataset "$EVAL_DATASET" \
            --save_dir "$SAVE_DIR" 
        
        # # unitary symmetric, RoPE
        python -u -m automatic_validation_script \
            --model_dir "./outputs/openwebtext-train/2025.12.25/110356" \
            --ckpt "$CKPT" \
            --eval_dataset "$EVAL_DATASET" \
            --save_dir "$SAVE_DIR" 
        
        # mdlm-owt-tiny-init-symm-qk_tied-RoPE
        python -u -m automatic_validation_script \
            --model_dir "./outputs/openwebtext-train/2025.12.25/094115" \
            --ckpt "$CKPT" \
            --eval_dataset "$EVAL_DATASET" \
            --save_dir "$SAVE_DIR" 
    done
done