# Real-Time 2D MRI Organ Segmentation (Attention U-Net, Statistically Validated & Multi-Sequence Robust)

Research-portfolio project: real-time abdominal organ segmentation on 2D MRI (CHAOS dataset),
with statistically rigorous multi-seed evaluation, a documented cross-sequence generalization
failure and fix, and real-time inference latency validation (<150ms/frame target).

## Key Finding 1 — Statistical Rigor (5-seed validation)

| Model | Mean Test Dice | Std Dev |
|---|---|---|
| Baseline U-Net | 0.8204 | ± 0.0278 |
| Attention U-Net | 0.8014 | ± 0.0232 |

**No statistically significant difference** (paired t=0.86, p≈0.44) across 5 independent seeds.
This finding — reached only after moving from single-run to multi-seed evaluation — highlights
the importance of statistical rigor in small-dataset medical imaging research, where single-run
comparisons can be misleading due to high variance.

## Key Finding 2 — Cross-Sequence Generalization Failure & Fix

| Model | T1DUAL Dice | T2SPIR Dice |
|---|---|---|
| Attention U-Net (T1-only) | 0.8338 | 0.0001 (complete collapse) |
| Attention U-Net (Multi-Sequence) | 0.8273 | 0.8132 |

Training only on T1DUAL, the Attention U-Net achieved strong same-sequence Dice but completely
failed on the same patients' T2SPIR scans — confirmed via confidence analysis to be
confidently-wrong, not merely uncertain (mean softmax at true-liver pixels: 0.98 background vs
0.01 liver). Root cause: T1 and T2SPIR (fat-suppressed) have near-inverted tissue contrast.
Retraining on combined T1DUAL+T2SPIR data resolved this at negligible cost to T1DUAL performance
(-0.0065), confirming the failure was a data-distribution gap, not an architectural limitation.

Full results, per-seed breakdown, and error analysis: results/metrics_report.md

## Real-Time Inference Latency (Attention U-Net, Colab T4 GPU)

| Configuration | Mean (ms) | Meets <150ms target |
|---|---|---|
| PyTorch Eager | ~12 | Yes |
| TorchScript | ~11 | Yes |

## Dataset

CHAOS (https://chaos.grand-challenge.org/) — MRI T1DUAL InPhase + T2SPIR, 20 patients,
liver/kidneys/spleen. Patient-wise split (14/3/3), leak-free.

## Project Structure

- src/ — datasets, preprocessing, models, losses, metrics, train.py
- inference/ — latency benchmarking
- results/ — metrics_report.md (full results + error analysis), figures/

## Known Limitations

- Test/val sets are small (n=3 patients each) — mitigated for the architecture comparison via
  5-seed validation, but cross-sequence numbers are single-run
- Only axial orientation evaluated (CHAOS provides axial only)
- ONNX Runtime GPU acceleration not benchmarked (CPU only)

## Future Work

- Multi-orientation evaluation (sagittal/coronal via multi-planar reformatting)
- Sub-pixel boundary refinement
- Extension to chest organs (requires additional dataset, e.g. ACDC)

## License

Research/portfolio use. CHAOS dataset license per chaos.grand-challenge.org terms.
