# CMFST

This repository provides the implementation of **CMFST**, a cross-modal speech-text fusion method for non-invasive Alzheimer's disease identification from spontaneous speech.

CMFST jointly models acoustic and linguistic cues from speech recordings and transcripts. It is designed to address long-speech modeling, uneven distribution of discriminative cues, and insufficient use of complementary information between audio and text modalities.

## File Structure

```text
CMFST/
├── model.py                    # Main model architecture
├── gate_pool.py                # Gating and pooling modules
├── dataset_gate.py             # Dataset loading and feature organization
├── train.py                    # Training script
├── train_5fold_cv.py           # Five-fold cross-validation training script
├── eval_ensemble.py            # Ensemble evaluation script
├── eval_ensemble_en.py         # Evaluation script for the English dataset setting
├── extract_pooled_feats.py     # Feature extraction / pooled feature generation
├── explain_model.py            # Model explanation and visualization utilities
├── Modality_ablation1.py       # Modality ablation experiment script
├── Modality_ablation2.py       # Modality ablation experiment script
├── Modality_ablation3.py       # Modality ablation experiment script
├── Modality_ablation4.py       # Modality ablation experiment script
├── requirements.txt            # Python dependencies
└── README.md                   # Project description
```

## Requirements

The code was developed with Python and PyTorch. The main dependencies include:

```text
Python >= 3.10
PyTorch
Transformers
NumPy
Pandas
Scikit-learn
Tqdm
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

If some packages are missing, install them manually according to the error messages.

## Datasets

The experiments are conducted on publicly available research datasets, including:

- **NCMMSC**
- **ADReSSo**

Due to data usage restrictions, the original datasets are **not included** in this repository. Please obtain the datasets from their official sources and organize the audio, transcript, label, and feature files according to the paths defined in `dataset_gate.py` and the corresponding training scripts.

A recommended data organization is:

```text
data/
├── NCMMSC/
│   ├── audio/
│   ├── transcripts/
│   ├── features/
│   └── labels.csv
└── ADReSSo/
    ├── audio/
    ├── transcripts/
    ├── features/
    └── labels.csv
```

Before training or evaluation, please check and modify the dataset paths in the scripts according to your local environment.

## Usage

### 1. Feature Preparation

If pooled features need to be generated before training, run:

```bash
python extract_pooled_feats.py
```

Please make sure that the input paths and output paths are correctly specified in the script.

### 2. Model Training

For standard training, run:

```bash
python train.py
```

For five-fold cross-validation, run:

```bash
python train_5fold_cv.py
```

Please modify the dataset path, feature path, model saving path, and task setting in the corresponding script before running.

### 3. Evaluation

For ensemble evaluation, run:

```bash
python eval_ensemble.py
```

For the English dataset setting, run:

```bash
python eval_ensemble_en.py
```

If a pretrained checkpoint is used, make sure the checkpoint path in the evaluation script matches the actual file location.

### 4. Modality Ablation Experiments

The modality ablation experiments can be conducted using:

```bash
python Modality_ablation1.py
python Modality_ablation2.py
python Modality_ablation3.py
python Modality_ablation4.py
```

These scripts are used to evaluate the contribution of different modality settings. Please check the corresponding script comments or path settings before running.

### 5. Model Explanation

To generate model explanation or visualization results, run:

```bash
python explain_model.py
```

The generated outputs depend on the configuration in the script, such as checkpoint paths, feature paths, and output directories.

## Notes

- Raw datasets are not included in this repository.
- Please do not upload private data, original speech recordings, or restricted-access datasets to a public repository.
- Model checkpoints can be large. If needed, they may be released separately.
- File paths in the scripts may need to be modified before running on a new machine.

