#!/bin/bash -l
#SBATCH -t 01:00:00
#SBATCH -J reconstructions
#SBATCH -M kale,ukko
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
#SBATCH --mem=8G
#SBATCH --array=0-6
#SBATCH --output=/home/leeviloi/fluxrope_thesis/timeseries_tail/missing_sc_flipped_z=0.5/logs/%x_%A_%a.out

source /appl/easybuild/opt/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate vir_env2

cd ~/Fluxrope-Reconstruction

T_REF=1360

START=1330
END=1380
echo "Start time = $START"
echo "End time = $END"

SC_NAMES=(sc1 sc2 sc3 sc4 sc5 sc6 sc7)
MISSING_SC=${SC_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Missing spacecraft = $MISSING_SC"

python ./temporal_reconstruction/run.py --start_time "${START}" --end_time "${END}" \
    --missing_sc "${MISSING_SC}"