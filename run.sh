#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT_NAME
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS
module purge
eval "$(conda shell.bash hook)"
conda activate vllm_env

export PYTHONPATH=$PWD:$PYTHONPATH
python fine_tuning.py
