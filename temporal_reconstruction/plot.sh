#!/bin/bash -l
#SBATCH -t 01:00:00
#SBATCH -J reconstructions
#SBATCH -M kale,ukko
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
#SBATCH --mem=8G
#SBATCH --output=/home/leeviloi/Fluxrope-Reconstruction/%x_%A_%a.out

source /appl/easybuild/opt/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate vir_env2

cd ~/Fluxrope-Reconstruction

T_REF=1360
MAX_OFFSET=20

#start_times=()
#end_times=()

#for ((i=MAX_OFFSET; i>=0; i--)); do
#    start_times+=($((T_REF - i)))
#    end_times+=($((T_REF + i)))
#done

#START=${start_times[$SLURM_ARRAY_TASK_ID]}
#END=${end_times[$SLURM_ARRAY_TASK_ID]}

START=1340
END=1372
echo "Start time = $START"
echo "End time = $END"

python ./temporal_reconstruction/run.py --start_time "${START}" --end_time "${END}" 