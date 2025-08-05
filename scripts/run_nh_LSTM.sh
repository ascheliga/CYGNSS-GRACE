#!/bin/bash
#SBATCH --job-name=run_nh_LSTM_EXP00_att00
#SBATCH --account=fc_ecohydrology
#SBATCH --partition=savio2_htc
#SBATCH --time=01:01:15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2



eval "$(conda shell.bash hook)"
conda activate rioxarray_env

# pre-processing
python LSTM_preprocessing.py


conda activate neuralhydrology

nh-run train --config-file /global/home/users/ann_scheliga/neuralhydrology/cygnss_models/EXP00_1basin_sw/colorado_1basin_wi_sw.yml

nh-run train --config-file /global/home/users/ann_scheliga/neuralhydrology/cygnss_models/EXP00_1basin_sw/colorado_1basin_no_sw.yml