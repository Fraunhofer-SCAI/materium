#!/bin/bash
#SBATCH --mem=32gb                    # Total memory limit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-mix
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00               # Time limit 2-hrs:min:sec days


export CUDA_VISIBLE_DEVICES=0

module load CUDA/11.8.0
cd ~/materium
source ./llm_env/bin/activate
cd ~/materium/src/materium/

mkdir -p ./llm_train
srun python train.py > "llm_train/run_$SLURM_JOB_ID.out" 
