#!/bin/bash
#SBATCH --job-name=CYGNSS_HUC_daily_area_attempt27
#SBATCH --account=fc_ecohydrology
#SBATCH --partition=savio2_htc
#SBATCH --time=00:11:15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

## Command(s) to run:

eval "$(conda shell.bash hook)"

conda activate rioxarray_env
python CYGNSS_daily_area_nonparallel_nonGRanD.py
