#!/bin/bash
#SBATCH --job-name=download_era5_met_systematic_attempt08
#SBATCH --account=fc_ecohydrology
#SBATCH --partition=savio2_htc
#SBATCH --time=05:02:35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

## Command(s) to run:

echo "hello world"
module load python

eval "$(conda shell.bash hook)"

conda activate rio_keras
#conda activate /global/home/users/ann_scheliga/.conda/envs/rio_keras
# python era5_monthly_data_downoad.py

export start_year=2015
export end_year_ex=2024
# ipython era5_preciptype_download.py
# ipython era5_precip_download.py
ipython era5_temp_download.py
