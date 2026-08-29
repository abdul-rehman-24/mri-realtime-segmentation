# Real-Time 2D MRI Organ Segmentation (Attention U-Net vs Baseline U-Net)

Portfolio research project: lightweight Attention U-Net for real-time abdominal organ
segmentation on 2D MRI, benchmarked against a plain U-Net baseline, with real-time
inference latency validation (<150ms/frame target).

## Key Results (Test Set, n=3 held-out patients)

| Organ | Baseline U-Net | Attention U-Net | Diff |
|---|---|---|---|
| Liver | 0.8969 | 0.9117 | +0.0149 |
| Right Kidney | 0.7979 | 0.8078 | +0.0098 |
| Left Kidney | 0.7619 | 0.8092 | +0.0472 |
| Spleen | 0.8155 | 0.8065 | -0.0090 |
| **Mean (organs)** | **0.8181** | **0.8338** | **+0.0157** |

Full results, per-patient robustness analysis, and error analysis: see
[`results/metrics_report.md`](results/metrics_report.md).

## Real-Time Inference Latency (Attention U-Net, Colab T4 GPU)
| Configuration | Mean (ms) | P95 (ms) | Meets <150ms target |
|---|---|---|---|
| PyTorch Eager | 13.49 | 20.00 | Yes |
| TorchScript | 10.79 | 11.10 | Yes |

## Dataset
[CHAOS](https://chaos.grand-challenge.org/) — MRI T1DUAL InPhase, 20 patients,
liver/kidneys/spleen. Patient-wise split (14/3/3) to prevent data leakage.
Not included in this repo (requires registration).

## Project Structure
