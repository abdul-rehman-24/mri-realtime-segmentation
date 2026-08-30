# CHAOS Spleen/Organ Segmentation -- Results Report

Patient-wise split (seed=42)

Train: 14 patients | Val: 3 patients | Test: 3 patients


## Model Comparison -- Test Set Per-Organ Dice

| Organ | Baseline U-Net | Attention U-Net | Diff |
|---|---|---|---|
| Background | 0.9951 | 0.9954 | +0.0004 |
| Liver | 0.8688 | 0.8921 | +0.0233 |
| Right Kidney | 0.8623 | 0.7129 | -0.1494 |
| Left Kidney | 0.8609 | 0.7189 | -0.1420 |
| Spleen | 0.7796 | 0.7218 | -0.0579 |
| **Mean (organs)** | **0.8429** | **0.7614** | **-0.0815** |

## Model Size

- Baseline U-Net: 7,762,597 params
- Attention U-Net: 7,851,329 params (+88,732, +1.1%)

## Per-Patient Robustness (Test Set)

| Patient | Baseline | Attention | Diff |
|---|---|---|---|
| 1 | 0.8714 | 0.8154 | -0.0559 |
| 10 | 0.7902 | 0.8359 | +0.0457 |
| 37 | 0.8408 | 0.5965 | -0.2444 |

Baseline: mean=0.8341, std=0.0335

Attention: mean=0.7493, std=0.1084

Attention improved 1/3 patients

## Training Config

- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=5)
- Loss: Dice + CrossEntropy (0.5/0.5)
- Epochs: 30, Batch size: 8
- Baseline best Val Dice: 0.8373
- Attention best Val Dice: 0.8461

## Known Limitations

- Test set n=3 patients -- trends only, not statistically significant
- Val set (n=3) shows high metric variance due to small size
- Only T1DUAL InPhase axial slices used; multi-orientation robustness not evaluated

## Multi-Seed Statistical Validation (5 seeds: 42, 123, 456, 789, 999)

| Model | Mean Test Dice | Std Dev | Seeds Won |
|---|---|---|---|
| Baseline U-Net | 0.8204 | 0.0278 | 3/5 |
| Attention U-Net | 0.8014 | 0.0232 | 2/5 |

**Paired difference**: +0.019 (Baseline higher on average)
**Paired t-test**: t=0.86, df=4 (critical t=2.78 at p<0.05) — **not statistically significant**

### Conclusion

Across 5 independent training runs with different random seeds (identical data split, 
architecture configuration, hyperparameters, and training protocol), Baseline U-Net and 
Attention U-Net show **no statistically significant difference** in test-set Dice score 
on this dataset (n=14 training patients). While Baseline U-Net had a marginally higher 
mean (0.8204 vs 0.8014), the difference is well within the range explained by 
training randomness (paired t=0.86, p≈0.44).

This is a valid and informative finding: it suggests that on datasets of this size, the 
added architectural complexity of attention gates does not reliably translate into 
measurable segmentation improvement, and that single-run comparisons (as in early 
iterations of this experiment) are unreliable due to high run-to-run variance — 
underscoring the necessity of multi-seed evaluation in small-dataset medical imaging research.
