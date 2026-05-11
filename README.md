# CMFST

This repository provides the implementation of **CMFST**, a cross-modal speech-text fusion method for non-invasive Alzheimer's disease identification from spontaneous speech.

CMFST integrates acoustic and linguistic cues by jointly modeling audio and text modalities. It is designed to address the challenges 
of long-speech modeling, uneven distribution of discriminative cues, and insufficient exploitation of cross-modal complementary information in speech-based Alzheimer's disease identification.

## Overview

CMFST mainly consists of the following components:

- **Audio feature extraction**: segment-level speech representations are extracted from long speech recordings.
- **Text feature extraction**: local segment-level textual representations and global utterance-level textual representations are obtained from transcripts.
- **Local-Global Semantic-Guided Audio Aggregation (LGSGA)**: local and global textual semantics are used to guide the aggregation of segment-level audio representations.
- **Audio-Text Collaborative Representation (ATCR)**: complementary information between audio and text modalities is further modeled for final prediction.
- **Classification module**: fused multimodal representations are used for Alzheimer's disease identification.

## Repository Structure

```text
CMFST/
├── model.py                  # Model definition
├── dataset_gate.py            # Dataset loading and preprocessing
├── train_5fold_cv.py          # Training and cross-validation script
├── gate_pool.py               # Gating and pooling modules
├── requirements.txt           # Python dependencies
├── README.md                  # Project description
└── .gitignore                 # Files ignored by Git
```

The actual file names may vary depending on the implementation.

## Requirements

The code was developed using Python and PyTorch. The recommended environment is:

```text
Python >= 3.10
PyTorch >= 2.0
Transformers
NumPy
Pandas
Scikit-learn
Tqdm
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, the commonly used packages can be installed manually:

```bash
pip install torch transformers numpy pandas scikit-learn tqdm
```

## Datasets

The experiments are conducted on publicly available research datasets, including:

- **NCMMSC**
- **ADReSSo**

Due to data usage restrictions, the original datasets are **not included** in this repository. 
Please obtain the datasets from their official sources and organize them according to the data loading script.

A recommended data structure is:

```text
data/
├── NCMMSC/
│   ├── audio/
│   ├── transcripts/
│   └── labels.csv
└── ADReSSo/
    ├── audio/
    ├── transcripts/
    └── labels.csv
```

Please modify the dataset paths in the training and evaluation scripts according to your local environment.

## Usage

### Training

An example command for training is:

```bash
python train.py
```

If the script supports command-line arguments, the dataset and output path can be specified as follows:

```bash
python train_5fold_cv.py 
```

Please adjust the arguments according to your implementation.

### Evaluation

After training, the saved model can be evaluated using the corresponding evaluation script. For example:

```bash
python eval.py 
```

If no independent evaluation script is provided, evaluation may be included in the training or cross-validation script.

## Notes

- This repository does not include raw datasets, pretrained checkpoints, or private experimental files.
- Please do not upload sensitive data, original speech recordings, or dataset files to a public repository.
- Large files such as model checkpoints should be excluded from GitHub or released separately if needed.






```
