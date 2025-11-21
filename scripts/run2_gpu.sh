#!/bin/bash
#SBATCH --job-name=em_gpu_test
#SBATCH --partition=ice-gpu          # GPU partition (or ice-gpu / coc-gpu) 
#SBATCH --gres=gpu:1             # request 1 A100 GPU
#SBATCH --time=00:30:00               # wall time
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=em_gpu_%j_test1.out        # output file
# Optional, if needed:
# #SBATCH -A GT-shared

module load anaconda3/2023.03
module load openmpi/4.1.5
conda activate paratopic

cd $SLURM_SUBMIT_DIR/..

echo "Running EM on GPU..."
python src/em_gpu.py \
    --doc_term data/processed/doc_term.npz \
    --vocab data/processed/vocab.txt \
    --K 20 \
    --max_iter 30 \
    --device cuda

