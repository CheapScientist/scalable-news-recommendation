#!/bin/bash
#SBATCH -A <YOUR-ALLOCATION>       # e.g. GT-shared or your PI’s allocation
#SBATCH -p coe-gpu                 # H100 partition
#SBATCH --gres=gpu:h100:1          # request exactly 1 H100 GPU
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J em_gpu_h100
#SBATCH -o slurm-em-gpu-h100-%j.out

module load anaconda3/2023.03
module load openmpi/4.1.5
conda activate paratopic

cd $SLURM_SUBMIT_DIR

python src/em_gpu.py \
    --doc_term data/processed/doc_term.npz \
    --vocab data/processed/vocab.txt \
    --K 20 \
    --max_iter 30 \
    --device cuda
