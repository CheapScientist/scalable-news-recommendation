#!/bin/bash
#SBATCH --job-name=weak_1gpu
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=weak_1gpu_%j.out

module load anaconda3/2023.03
module load openmpi/4.1.5
conda activate paratopic

cd $SLURM_SUBMIT_DIR/..

echo "Running weak scaling: 1 GPU (13k docs)"
mpirun -np 1 python src/em_mpi_gpu.py \
    --doc_term data/processed/doc_term_13k.npz \
    --vocab data/processed/vocab.txt \
    --K 20 \
    --max_iter 30 \
    --partition_strategy balanced \
    --seed 0
