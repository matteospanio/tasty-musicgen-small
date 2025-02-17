import argparse as ap
import pathlib as pt
from audiocraft.utils import export
from audiocraft import train

def main():
    parser = ap.ArgumentParser(
        'saveModel',
        description='Run this script to save the model weights.',
    )

    parser.add_argument('signature', help="The run ID.")
    parser.add_argument('-d', '--dest', help='Where to save the model files.', default='checkpoints')

    args = parser.parse_args()

    checkpoint_folder = pt.Path(f'{args.dest}/{args.signature}')
    checkpoint_folder.mkdir(exist_ok=True, parents=True)

    xp = train.main.get_xp_from_sig(args.signature)
    export.export_lm(xp.folder / 'checkpoint.th', )
    export.export_pretrained_compression_model('facebook/encodec_32khz', checkpoint_folder / 'compression_state_dict.bin')

if __name__ == '__main__':
    main()