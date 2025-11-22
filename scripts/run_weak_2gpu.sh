#!/bin/bash
#SBATCH --job-name=weak_2gpu
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=01:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=weak_2gpu_%j.out

module load anaconda3/2023.03
module load openmpi/4.1.5
conda activate paratopic

cd $SLURM_SUBMIT_DIR/..

echo "Running weak scaling: 2 GPUs (26k docs)"
mpirun -np 2 python src/em_mpi_gpu.py \
    --doc_term data/processed/doc_term_26k.npz \
    --vocab data/processed/vocab.txt \
    --K 20 \
    --max_iter=30 \
    --partition_strategy balanced \
    --seed 0
