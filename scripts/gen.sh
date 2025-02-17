#!/bin/bash
#SBATCH --job-name infer-musicgen
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user myemail
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 2
#SBATCH --partition allgroups
#SBATCH --mem 20G
#SBATCH --gres=gpu:rtx

# cd $WORKING_DIR
# prompt:
# - sour
# - salty
# - sweet
# - bitter
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export PATH="$PATH:$HOME/ffmpeg:$HOME/.local/bin"
srun ~/miniconda3/bin/conda run \
        -n musicgen \
	python scripts/generate.py \
                --length=15 \
                --prompt="sour, high-pitched, sharp, dissonant music, with 90 BPM. Use bright, crisp instruments like violins, flutes, and electronic tones to evoke a sour, sharp taste. Ambient for fine restaurant.;salty, mid-pitched, rhythmic, and grounded music, with 140 BPM and subtle harmonic tension. Use percussion, electric bass, and horns for a solid, balanced sound that conveys a salty flavor. Ambient for fine restaurant.;sweet, high-pitched, soft, and flowing music, with 70 BPM, consonant harmony, and legato articulation. Use gentle instruments like harp, piano, and strings to evoke a warm, sweet sensation.Ambient for fine restaurant.;bitter, low-pitched, deep, and reflective music, with 60 BPM and dissonant harmony. Use bass, cello, and deep brass for a dark, heavy sound that reflects the bitterness of taste.Ambient for fine restaurant." \
                --wandb-log \
                '/home/spanio/jobs/finetune-musicgen/artifacts/finetuned/'
                # 'facebook/musicgen-small'

                # --prompt="Create music that is energetic, tense, and sharp, with high-pitched notes, fast tempo, dissonant harmony, staccato articulation, using violins, flutes, and electronic tones. The dynamics should be loud and intense, evoking arousal and heightened intensity.;Create music that feels grounded, moderate, and rhythmic, with mid-pitched notes, steady tempo, and subtle harmonic tension. Use percussion, electric bass, and horns. Articulation should be short and steady. Dynamics should alternate between moderate and soft, evoking a balance of tension and relaxation.;Create music that is gentle, flowing, and soothing, with high-pitched notes, slow tempo, consonant harmony, and legato articulation. Use instruments like harp, piano, and strings. Dynamics should be soft, evoking warmth and pleasantness.;Create music that is dark, deep, and reflective, with low-pitched notes, slow tempo, dissonant harmony, and long, sustained articulations. Use bass, cello, and deep brass instruments. Dynamics should be soft to moderate, evoking a sense of heaviness and melancholy" \
