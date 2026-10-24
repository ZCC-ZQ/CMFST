# Experimental configurations and reproducibility notes

The source code for the main CMFST model, training, evaluation and ablation experiments is at the repository root. This page records the settings associated with the supporting results and points to the saved fold data. The original datasets require approval from their respective providers; this repository does not redistribute recordings, complete transcripts or extracted feature arrays.

## Preprocessing and feature extraction

Audio was standardized for sampling rate and amplitude, then segmented with a 6 s window and a 1.5 s overlap (25%). Chinese speech was transcribed using the iFLYTEK Chinese speech recognition platform; English speech was transcribed using Whisper-large-v3. Segments without valid speech/text were excluded to maintain audio–text alignment. Transcripts were normalized for full-/half-width characters and whitespace, and transcription-generated placeholders, duplicate punctuation and non-linguistic symbols were removed. The saved feature preparation used WavLM-base for audio, Chinese RoBERTa (`chinese-roberta-wwm-ext`) for Chinese text and English RoBERTa (`roberta-base`) for English text; the saved embeddings had 768 dimensions. Provider-hosted ASR model versions and service parameters that were not archived are not assigned retrospectively.

## Five-fold experiments

| Dataset | Training seed | Learning rate | Recurrent hidden dim. | Gate hidden dim. | Classifier hidden dim. | Factors | Epoch limit | Batch size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NCMMSC | 32 | 0.003 | 256 | 256 | 96 | 12 | 45 | 8 |
| ADReSSo | 32 | 0.005 | 256 | 256 | 96 | 12 | 45 | 8 |
| ADReSS | 40 | 0.004 | 256 | 256 | 96 | 8 | 45 | 8 |

These are the archived settings associated with the reported five-fold results. The dropout settings for the NCMMSC and ADReSSo five-fold runs were not recorded in the archived result files and are therefore not supplied as verified settings. [Fold assignments](data/fold_assignments.csv) list each recording ID, class and validation-fold number; [fold-wise results](tables/five_fold_results.md) give accuracy and macro-averaged F1-score. For these reported cross-validation figures, checkpoint selection and performance reporting used the same validation fold, so the figures should be read as model-selection cross-validation results rather than independent outer-test estimates.

## Official-split checkpoints

The already published `best_model/best_zh.pt` and `best_model/best_model_en1.pt` correspond to the NCMMSC and ADReSSo official-split results. `best_model/best_model_adress.pt` adds the ADReSS official-split checkpoint. Their checkpoint architectures use 768-dimensional input features, a recurrent hidden size of 256, a gate hidden size of 256 and a classifier hidden size of 96. The collaborative factor count is 12 for these three official-split checkpoints; the corresponding output class counts are 3, 2 and 2, respectively. The archived ADReSS five-fold settings above are distinct from its official-split checkpoint settings.

## Ablation results

[Individual run accuracies](tables/ablation_run_accuracies.md) and [paired comparisons](tables/ablation_paired_comparisons.md) document the five saved evaluations and statistical calculations. For Tables III–V, the archived checkpoints carry run identifiers 38–42. The Table VI identifiers 1–5 are archive run identifiers; the exact random-seed mapping for these historical runs was not recovered. The full-model row in Table VI reuses the separate 38–42 series and was not paired with the 1–5 series in the reported statistical tests. No seed value was inferred from a run identifier.
