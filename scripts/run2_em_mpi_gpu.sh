#!/bin/bash
#SBATCH --job-name=em_mpi_gpu
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 4 MPI ranks
#SBATCH --gres=gpu:4            # 4 GPUs (or l40s:4)
#SBATCH --time=01:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=em_mpi_gpu_%j.out

module load anaconda3/2023.03
module load openmpi/4.1.5
conda activate paratopic

cd $SLURM_SUBMIT_DIR/..

echo "Running MPI+GPU EM..."
mpirun -np 4 python src/em_mpi_gpu.py \
    --doc_term data/processed/doc_term.npz \
    --vocab data/processed/vocab.txt \
    --K 20 \
    --max_iter 30 \
    --partition_strategy balanced \
    --seed 0
