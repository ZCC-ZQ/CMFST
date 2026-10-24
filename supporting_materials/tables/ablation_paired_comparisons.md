# Complete Paired Statistical Comparisons for Ablation Experiments

| Dataset | Main-text table | Comparison | Mean difference (pp) | Unadjusted 95% CI (pp) | Raw p | Holm-adjusted p | Family size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NCMMSC | III | Full vs Audio | +7.23 | [+5.33, +9.12] | 0.000450651 | 0.000901302 | 2 |
| NCMMSC | III | Full vs Text | +15.29 | [+11.16, +19.43] | 0.000506717 | 0.000901302 | 2 |
| ADReSSo | III | Full vs Audio | +11.27 | [+1.30, +21.24] | 0.0349197 | 0.0698394 | 2 |
| ADReSSo | III | Full vs Text | +1.69 | [-1.44, +4.82] | 0.208 | 0.208 | 2 |
| NCMMSC | IV | LGSGA vs Mean | +2.69 | [-0.72, +6.10] | 0.0939566 | 0.245664 | 6 |
| NCMMSC | IV | LGSGA vs Max | +2.35 | [+0.49, +4.22] | 0.0248962 | 0.124481 | 6 |
| NCMMSC | IV | LGSGA vs Last | +7.90 | [+3.95, +11.84] | 0.00512888 | 0.0307733 | 6 |
| NCMMSC | IV | LGSGA vs Self-attention | +2.69 | [-0.21, +5.58] | 0.061416 | 0.245664 | 6 |
| NCMMSC | IV | LGSGA vs GSGA | +3.03 | [-1.44, +7.49] | 0.133012 | 0.245664 | 6 |
| NCMMSC | IV | LGSGA vs LSGA | +2.86 | [-0.41, +6.12] | 0.0720921 | 0.245664 | 6 |
| ADReSSo | IV | LGSGA vs Mean | +2.25 | [+0.26, +4.25] | 0.0349197 | 0.174599 | 6 |
| ADReSSo | IV | LGSGA vs Max | +2.25 | [-1.58, +6.08] | 0.177808 | 0.711231 | 6 |
| ADReSSo | IV | LGSGA vs Last | +5.07 | [+1.24, +8.90] | 0.0213116 | 0.12787 | 6 |
| ADReSSo | IV | LGSGA vs Self-attention | +1.13 | [-2.86, +5.11] | 0.476621 | 0.891229 | 6 |
| ADReSSo | IV | LGSGA vs GSGA | +1.69 | [-1.44, +4.82] | 0.208 | 0.711231 | 6 |
| ADReSSo | IV | LGSGA vs LSGA | +1.41 | [-3.22, +6.04] | 0.445614 | 0.891229 | 6 |
| NCMMSC | V | ATCR-Full vs ATCR-Text | +4.54 | [+0.07, +9.00] | 0.0477008 | 0.190803 | 4 |
| NCMMSC | V | ATCR-Full vs ATCR-Audio | +2.35 | [-0.14, +4.84] | 0.0586491 | 0.190803 | 4 |
| NCMMSC | V | ATCR-Full vs w/o ATCR | +2.35 | [-1.06, +5.77] | 0.128148 | 0.247714 | 4 |
| NCMMSC | V | ATCR-Full vs w/o ATCR-Scale | +2.86 | [-1.22, +6.94] | 0.123857 | 0.247714 | 4 |
| ADReSSo | V | ATCR-Full vs ATCR-Text | +1.69 | [+0.23, +3.15] | 0.0326779 | 0.0784353 | 4 |
| ADReSSo | V | ATCR-Full vs ATCR-Audio | +11.83 | [+3.86, +19.81] | 0.0146311 | 0.0585242 | 4 |
| ADReSSo | V | ATCR-Full vs w/o ATCR | +6.76 | [+1.31, +12.21] | 0.0261451 | 0.0784353 | 4 |
| ADReSSo | V | ATCR-Full vs w/o ATCR-Scale | +5.35 | [+0.19, +10.51] | 0.0449781 | 0.0784353 | 4 |
| NCMMSC | VI | Base+LGSGA vs Base | +1.01 | [-2.40, +4.42] | 0.458112 | 1 | 3 |
| NCMMSC | VI | Base+ATCR vs Base | +1.51 | [-4.08, +7.10] | 0.494255 | 1 | 3 |
| NCMMSC | VI | Base+ATCR vs Base+LGSGA | +0.50 | [-3.30, +4.31] | 0.7316 | 1 | 3 |
| ADReSSo | VI | Base+LGSGA vs Base | +0.85 | [-1.50, +3.19] | 0.373901 | 0.373901 | 3 |
| ADReSSo | VI | Base+ATCR vs Base | +8.17 | [+4.18, +12.16] | 0.00471962 | 0.0141589 | 3 |
| ADReSSo | VI | Base+ATCR vs Base+LGSGA | +7.32 | [+3.74, +10.91] | 0.00476113 | 0.0141589 | 3 |

Two-sided paired t-tests use five pairs (df=4), with differences computed as the first method minus the second. Confidence intervals use the sample SD of paired differences (`ddof=1`) and are not adjusted for multiple comparisons. The significance threshold is 0.05 after Holm correction. Correction is performed within each dataset × source-table family: 2, 6, 4 and 3 tests for Tables III, IV, V and VI. All 30 conducted comparisons are shown, including those omitted from main-text Table VII; this is not an exhaustive all-pairs analysis. Pairing follows archived run identifiers. The reused Full row is excluded from Table VI comparisons because it belongs to a different run series.

A confidence interval excluding zero can coexist with a nonsignificant Holm-adjusted p value. These within-study ablation tests do not establish statistical superiority over published results from other studies.
