#!/bin/bash
#SBATCH --job-name=download_era5_met_attempt17
#SBATCH --account=fc_ecohydrology
#SBATCH --partition=savio2_htc
#SBATCH --time=00:30:35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

## Command(s) to run:

echo "hello world"
module load python

eval "$(conda shell.bash hook)"

conda activate rio_keras
#conda activate /global/home/users/ann_scheliga/.conda/envs/rio_keras
# python era5_monthly_data_downoad.py

ipython era5_preciptype_download.py
# ipython era5_precip_download.py
# ipython era5_temp_download.py
