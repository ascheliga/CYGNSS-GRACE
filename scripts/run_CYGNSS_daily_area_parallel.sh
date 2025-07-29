#!/bin/bash
#SBATCH --job-name=CYGNSS_powell_daily_area_mp_att9
#SBATCH --account=fc_ecohydrology
#SBATCH --partition=savio2_htc
#SBATCH --time=00:05:15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

## Command(s) to run:

echo "hello world"


eval "$(conda shell.bash hook)"

conda activate rioxarray_env
# module load python
python CYGNSS_daily_area_parallel_mppool.py


## NOTES
# could not run parallel with rio_keras environment. Ran into: ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29'