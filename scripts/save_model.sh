#!/bin/bash
#SBATCH --job-name save-musicgen
#SBATCH --output log/out/%j.txt
#SBATCH --error log/err/%j.txt
#SBATCH --mail-user spanio@dei.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --partition allgroups
#SBATCH --mem 32G

# cd $WORKING_DIR
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
srun ~/miniconda3/bin/conda run \
        -n musicgen \
	python scripts/save_model.py -d artifacts $1
