#!/bin/bash
#SBATCH --job-name musicgen
#SBATCH --output log/out/%j.txt
#SBATCH --error log/err/%j.txt
#SBATCH --mail-user myemail
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 2
#SBATCH --partition allgroups
#SBATCH --mem 20G
#SBATCH --gres=gpu:rtx

# cd $WORKING_DIR
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORHC_DISTTRIBUTED_DEBUG=INFO
export PATH="$PATH:$HOME/ffmpeg:$HOME/.local/bin"

srun ~/miniconda3/bin/conda run \
        -n musicgen \
        dora -P audiocraft run \
                solver=musicgen/musicgen_base_32khz \
                model/lm/model_scale=small \
                continue_from=//pretrained/facebook/musicgen-small \
                conditioner=text2music \
                dset=audio/train \
                logging.log_wandb=true \
                wandb.project=musicgen2 \
                dataset.num_workers=1 \
                dataset.valid.num_samples=1 \
                dataset.batch_size=4 \
                schedule.cosine.warmup=8 \
                optim.lr=1e-4 \
                optim.epochs=30 \
                optim.updates_per_epoch=2000 \
                optim.adam.weight_decay=0.01 \
                generate.lm.prompted_samples=False \
                generate.lm.gen_gt_samples=True \
                slurm.gpus=1 \
                slurm.mem_per_gpu=24 \
                slurm.time=172800 \
                slurm.partition=allgroups \
                optim.optimizer=adamw
