import argparse as ap
from dataclasses import dataclass
import logging
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    wandb_log: bool
    length: int
    model_path: str
    prompt: str
    destination: str
    times: int


def get_args() -> Args:
    parser = ap.ArgumentParser(
        "musicgen",
        description="Script to generate audio from a pretrained MusicGen model.",
    )
    parser.add_argument(
        "-l", "--length", help="Length of the generated sample.", default=30, type=int
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help='The prompt used to condition the output. You can write more than one separating them with ";"',
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dest",
        help="The destination folder where to save the generated files. This is set to the current directory by default.",
        default=".",
    )
    parser.add_argument("-t", "--times", help="Specify the number of times to generate each prompt. 1 by default.", default=1, type=int)
    parser.add_argument(
        "-w", "--wandb-log", action="store_true", help="Enable wandb logging."
    )
    parser.add_argument(
        "path",
        help='The path to the model weights. To use the standard one try "facebook/musicgen-small."',
    )

    args = parser.parse_args()

    return Args(
        wandb_log=args.wandb_log,
        model_path=args.path,
        length=args.length,
        prompt=args.prompt,
        destination=args.dest,
        times=args.times,
    )


def main() -> None:
    args = get_args()

    if args.wandb_log:
        wandb.init(
            project="musicgen2",
            job_type="generate",
        )

    musicgen = MusicGen.get_pretrained(args.model_path)
    musicgen.set_generation_params(duration=args.length)
    logger.info(f"Loaded model {args.model_path}")

    prompts = args.prompt.split(";")
    prompts = args.times * prompts

    logger.info(f"Generating audio for prompts: {prompts}")
    wavs = musicgen.generate(prompts)

    for idx, (wav, prompt) in enumerate(zip(wavs, prompts)):
        generated_audio = audio_write(
            f"{args.destination}/{idx}_{prompt.replace(" ", "_").replace(".", "_")}",
            wav.cpu(),
            musicgen.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )
        logging.info(f"Generated audio for prompt: {prompt}")
        if args.wandb_log:
            wandb.log(
                {
                    f"generated_audio_{idx}": wandb.Audio(
                        str(generated_audio),
                        caption=prompt,
                        sample_rate=musicgen.sample_rate,
                    )
                }
            )


if __name__ == "__main__":
    main()
