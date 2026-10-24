# Five-Fold Cross-Validation Results for All Three Datasets

Accuracy and macro-averaged F1-score are percentages. SD is the population standard deviation across five folds (`ddof=0`).

| Dataset | Fold | Training n | Validation n | Accuracy (%) | F1-score (%) |
| --- | --- | --- | --- | --- | --- |
| NCMMSC | 1 | 319 | 80 | 92.50 | 92.14 |
| NCMMSC | 2 | 319 | 80 | 88.75 | 88.24 |
| NCMMSC | 3 | 319 | 80 | 88.75 | 89.02 |
| NCMMSC | 4 | 319 | 80 | 92.50 | 92.32 |
| NCMMSC | 5 | 320 | 79 | 89.87 | 89.65 |
| NCMMSC | Mean ± SD | — | — | 90.47 ± 1.70 | 90.27 ± 1.66 |
| ADReSSo | 1 | 189 | 48 | 95.83 | 95.83 |
| ADReSSo | 2 | 189 | 48 | 85.42 | 85.36 |
| ADReSSo | 3 | 190 | 47 | 85.11 | 85.00 |
| ADReSSo | 4 | 190 | 47 | 87.23 | 87.18 |
| ADReSSo | 5 | 190 | 47 | 87.23 | 87.23 |
| ADReSSo | Mean ± SD | — | — | 88.16 ± 3.94 | 88.12 ± 3.96 |
| ADReSS | 1 | 124 | 32 | 87.50 | 87.45 |
| ADReSS | 2 | 125 | 31 | 83.87 | 83.80 |
| ADReSS | 3 | 125 | 31 | 90.32 | 90.28 |
| ADReSS | 4 | 125 | 31 | 83.87 | 83.87 |
| ADReSS | 5 | 125 | 31 | 87.10 | 86.97 |
| ADReSS | Mean ± SD | — | — | 86.53 ± 2.44 | 86.48 ± 2.43 |

[Fixed fold assignments](../data/fold_assignments.csv) and [experimental configurations](../experimental_configurations.md) are also provided.

The checkpoint was selected on the same validation fold on which performance was reported. These are model-selection cross-validation results, not independent outer-test estimates. The reported training seeds and learning rates differ by dataset; see the configurations rather than assuming a single setting for every experiment. Recorded fold IDs are disjoint at recording level; this alone is not a separate audit of all possible participant relationships.
