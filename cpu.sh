#!/bin/bash
#SBATCH --job-name=torch-test%j  # create a short name for your job
#SBATCH --array=1-500
#SBATCH --nodes=1                # node count
#SBATCH --cpus-per-task=10       # cpu-cores per task
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p defq                  # partition name
#SBATCH --output job%j.%N.out    # output file with job ID and node name
#SBATCH --error job%j.%N.err     # error file with job ID and node name
#SBATCH --account rc_general     # add your PI's account name

###########Load modules and enter code below

module load python3/anaconda/3.12
 
source /work/sm222/envis/python_env/bin/activate
 

# ── sanity checks (printed to your .out log) ───────────────────────────────────
#echo "=== which python ==="
#which python
#echo "=== python version ==="
#python --version
#echo "=== GPU check ==="
#nvidia-smi
#python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
 
# ── run script ────────────────────────────────────────────────────────────


START_TIME=$(date +%s)
echo "=== Job ${SLURM_ARRAY_TASK_ID} started at $(date) ==="
 
# ── run script ────────────────────────────────────────────────────────────────
python --version
cd /work/sm222
python data_generator.py $SLURM_ARRAY_TASK_ID
 
# ── end timing ────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
 
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))
 
echo "=== Job ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
echo "=== Elapsed time: ${HOURS}h ${MINS}m ${SECS}s (${ELAPSED}s total) ==="
