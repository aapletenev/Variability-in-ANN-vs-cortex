#!/bin/bash
#SBATCH --job-name=gaus_noise
#SBATCH --partition=gpu
#SBATCH --mem=250GB
#SBATCH --output=main_%j.out
#SBATCH --mail-user=jfeld2@u.rochester.edu
#SBATCH --mail-type=ALL

source /software/miniconda3/4.9.2/etc/profile.d/conda.sh
conda activate myenv
python main.py
