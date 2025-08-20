#!/bin/bash
#SBATCH --job-name=write_topo_att02
#SBATCH --account=fc_ecohydrology
#SBATCH --partition=savio2_htc
#SBATCH --time=00:02:15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

## Command(s) to run:

eval "$(conda shell.bash hook)"

conda activate rioxarray_env
python create_basin_attributes.py