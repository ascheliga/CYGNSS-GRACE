#!/bin/bash
#SBATCH --job-name=CYGNSS_daily_area_glen_canyon_att0
#SBATCH --account=fc_ecohydrology
#SBATCH --partition=savio2_htc
#SBATCH --time=00:11:15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

## Command(s) to run:

eval "$(conda shell.bash hook)"

conda activate rioxarray_env
python CYGNSS_daily_area_nonparallel.py
