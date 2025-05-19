#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:turing:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nfs/robertjanbruin/bulk-home/out/run-%j.out
#SBATCH --mail-type=END,FAIL

echo "Input to sbatch: " $@

source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate vit-shap

export WANDB_API_KEY="4dd7e3d25b8b902100d6615a7012e2bd4c72babc"

# For multi-GPU, don't use P2P as it hangs
export NCCL_P2P_DISABLE=1

cd ~/bulk-home/code/vit-position-info/toy_experiments
srun "$@"