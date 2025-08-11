#!/bin/bash
#SBATCH --job-name=download_era5_met_2000s_mppool_att00
#SBATCH --account=fc_ecohydrology
#SBATCH --partition=savio2_htc
#SBATCH --time=01:00:35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3

## Command(s) to run:

eval "$(conda shell.bash hook)"

conda activate rio_keras
#conda activate /global/home/users/ann_scheliga/.conda/envs/rio_keras
# python era5_monthly_data_downoad.py

export start_year=2000
export end_year_ex=2010
python era5_preciptype_download.py
python era5_precip_download.py
python era5_temp_download.py
