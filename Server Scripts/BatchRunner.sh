#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH -t 0-12:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
module purge
module load anaconda/py3
source activate Crypto-Thesis
python3 LinearRegressionWorking.py $1 $2 $3 $4 $5 >> out.txt
