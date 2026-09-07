#!/bin/bash -l
#SBATCH -t 01:00:00
#SBATCH -J reconstructions
#SBATCH -M kale,ukko
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
#SBATCH --mem=8G
#SBATCH --array=0-12
#SBATCH --output=/home/leeviloi/fluxrope_thesis/timeseries_tail/dataset_constrained/centered_at_1360_flipped_z=0.5/radius_wass/logs/%x_%A_%a.out

source /appl/easybuild/opt/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate vir_env2

cd ~/Fluxrope-Reconstruction

T_REF=1360
MAX_OFFSET=20

#tart_times=()
#end_times=()

#for ((i=MAX_OFFSET; i>=0; i--)); do
#    start_times+=($((T_REF - i)))
#    end_times+=($((T_REF + i)))
#done

#START=${start_times[$SLURM_ARRAY_TASK_ID]}
#END=${end_times[$SLURM_ARRAY_TASK_ID]}

#START=1400
#END=1500
#echo "Start time = $START"
#echo "End time = $END"
radii=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6)
RADIUS=${radii[$SLURM_ARRAY_TASK_ID]}

python ./temporal_reconstruction/run.py --radius "${RADIUS}" 