#!/bin/bash
#$ -S /bin/bash
#$ -N onn_3d_8pm
#$ -cwd
#$ -j y
#$ -o /data.nst/ysinha/projects/ONN/logs/
#$ -t 1

#Remember to change:
# 1. the name of the job
# 2. the output directory for logs

# Some diagnostic messages for the output
echo "Started: $(date)"
echo "Running on: $(hostname)"
echo "------------------"

# Check if SGE_TASK_ID is set (this is provided by the scheduler)
if [ -z "${SGE_TASK_ID}" ]; then
  echo "Error: SGE_TASK_ID is not set."
  exit 1
fi

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#conda activate psl_env
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate onn_env

exe="/ysinha/projects/ONN/fig3d.py"
 
#CHANGE SIMULATION NAME & LOG Directory!!!!!!!

echo "Running task $SGE_TASK_ID"

# Run the Python script with the appropriate parameters
#python "$exe" --an $an --ap $ap --b $b --c $c --sweep_folder "$sweep_folder" --task_id "$SGE_TASK_ID"
python "$exe" 

exit_code=$?

# Check if the Python script ran successfully
if [ $exit_code -ne 0 ]; then
    echo "Error: Task with task_id=$SGE_TASK_ID failed with exit code $exit_code" >&2
    exit $exit_code
fi

echo "Task with task_id=$SGE_TASK_ID completed successfully"
echo "------------------"


