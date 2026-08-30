# Real-Time 2D MRI Organ Segmentation (Attention U-Net vs Baseline U-Net)

Research project: statistically rigorous comparison of Attention U-Net vs Baseline U-Net
for real-time abdominal organ segmentation on 2D MRI, with real-time inference validation.

## Key Finding (5-seed statistical validation)

| Model | Mean Test Dice | Std Dev |
|---|---|---|
| Baseline U-Net | 0.8204 | ± 0.0278 |
| Attention U-Net | 0.8014 | ± 0.0232 |

**No statistically significant difference** (paired t=0.86, p≈0.44) across 5 independent
seeds. This finding — reached only after moving from single-run to multi-seed evaluation —
highlights the importance of statistical rigor in small-dataset medical imaging research,
where single-run comparisons can be misleading due to high variance.

Full results, per-seed breakdown, and error analysis: [`results/metrics_report.md`](results/metrics_report.md).

## Real-Time Inference Latency (Attention U-Net, Colab T4 GPU)
| Configuration | Mean (ms) | Meets <150ms target |
|---|---|---|
| PyTorch Eager | ~12 | Yes |
| TorchScript | ~11 | Yes |

## Dataset
[CHAOS](https://chaos.grand-challenge.org/) — 20 patients, MRI T1DUAL InPhase,
liver/kidneys/spleen. Patient-wise split (14/3/3), leak-free.

## Project Structure
