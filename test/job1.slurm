#!/bin/bash
#SBATCH --job-name=lorenz63_deriv
#SBATCH --array=0-3
#SBATCH --output=logs/lorenz63_%A_%a.out
#SBATCH --error=logs/lorenz63_%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --partition=sapphire
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module purge
# module load GCCcore/11.3.0
# module load Python/3.10.4
# source ~/venvs/venv-3.10.4/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs

NSRS=(0.01 0.1 0.5 1.0)
NSR=${NSRS[$SLURM_ARRAY_TASK_ID]}

echo "Job started on $(date)"
echo "Running on node: $HOSTNAME"
echo "Python version: $(python --version)"
echo "Using $OMP_NUM_THREADS threads"
echo "Working directory before cd: $PWD"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "NSR=$NSR"
free -h
echo "SLURM mem allocation: ${SLURM_MEM_PER_NODE:-Unknown}"

cd ~/code/RKHS4DD/test

stdbuf -o0 -e0 python run_lorenz63_derivative.py --nsr ${NSR}

echo "Job finished at $(date)"