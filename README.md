# tasty-musicgen-small

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Description

This repository contains the code to finetune [MusicGEN](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) on a small multimodal dataset of music that can be induce gustatory sensations. The dataset is available [here](https://github.com/matteospanio/taste-music-dataset).

## Usage

### Fine-tune

To fine-tune the model on the dataset, run the following command:

```bash
bash run.sh
```

### Inference

The model has been released in the [Hugging Face Model Hub](https://huggingface.co/matteospanio/tasty-musicgen-small) and can be used with the following code:

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-to-audio", model="csc-unipd/tasty-musicgen-small")
```

```python
# or Load model directly
from transformers import AutoTokenizer, AutoModelForTextToWaveform

tokenizer = AutoTokenizer.from_pretrained("csc-unipd/tasty-musicgen-small")
model = AutoModelForTextToWaveform.from_pretrained("csc-unipd/tasty-musicgen-small")
```

## Installation

Create a conda environment with the required dependencies:

```bash
conda env create -n tasty-musicgen-small -f requirements.txt
```

If you don't use conda remember to install ffmpeg.
This program has been tested with the latest version of Python 3.10, more recent versions may not work, for further information see the [official audiocraft documentation](https://github.com/facebookresearch/audiocraft/).

## Authors

- [Matteo Spanio](https://matteospanio.github.io/)

## Citation

If you use this code in your research, please cite the following article:

```
@misc{spanio2025multimodalsymphonyintegratingtaste,
      title={A Multimodal Symphony: Integrating Taste and Sound through Generative AI}, 
      author={Matteo Spanio and Massimiliano Zampini and Antonio Rodà and Franco Pierucci},
      year={2025},
      eprint={2503.02823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.02823}, 
}
```

## License

This repository is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) License.
