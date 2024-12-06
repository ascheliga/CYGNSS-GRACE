#!/bin/bash
#SBATCH --job-name=CYGNSS_powell_daily_area_attempt3
#SBATCH --account=fc_ecohydrology
#SBATCH --partition=savio2_htc
#SBATCH --time=00:00:15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

## Command(s) to run:

echo "hello world"


eval "$(conda shell.bash hook)"

conda activate rio_keras
module load python
python -m codebase.CYGNSS_daily_area_nonparallel