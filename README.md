# Real-Time 2D MRI Organ Segmentation — Statistically Validated, Multi-Sequence & Multi-Orientation Robust Attention U-Net

Research-portfolio project: real-time abdominal organ segmentation on 2D MRI (CHAOS dataset),
built around three research questions — architecture comparison rigor, cross-sequence
generalization, and cross-orientation generalization — each investigated, diagnosed, and
(where possible) fixed with quantified results. Real-time inference latency validated
against a <150ms/frame constraint.

## Key Finding 1 — Statistical Rigor (5-seed validation)

| Model | Mean Test Dice | Std Dev |
|---|---|---|
| Baseline U-Net | 0.8204 | ± 0.0278 |
| Attention U-Net | 0.8014 | ± 0.0232 |

No statistically significant difference (paired t=0.86, p≈0.44) across 5 independent seeds —
a finding only visible after moving from single-run to multi-seed evaluation, highlighting how
misleading single-run comparisons can be on small medical imaging datasets.

## Key Finding 2 — Cross-Sequence Generalization: Diagnosed and Fixed

| Model | T1DUAL Dice | T2SPIR Dice |
|---|---|---|
| Attention U-Net (T1-only) | 0.8338 | 0.0001 (complete collapse) |
| Attention U-Net (multi-sequence) | 0.8533 | 0.8762 |

Training only on T1DUAL, the model completely failed on the same patients' T2SPIR scans —
confirmed via confidence analysis to be confidently-wrong (0.98 background probability at
true-liver pixels), not merely uncertain. Root cause: T1 and T2SPIR (fat-suppressed) have
near-inverted tissue contrast. Retraining on combined T1DUAL+T2SPIR data resolved this fully,
with T1DUAL performance also improving slightly.

## Key Finding 3 — Cross-Orientation Generalization: Diagnosed and Partially Fixed

CHAOS provides only axial acquisitions. Sagittal and coronal views were derived via
multi-planar reformatting (stacking axial slices into a 3D volume and resampling along
each plane) — disclosed here as derived, not native, acquisitions.

| View | Axial-only model | Multi-view model |
|---|---|---|
| Sagittal (derived) | 0.0022 | 0.4770 |
| Coronal (derived) | ~0.0000 | 0.4663 |

Adding derived sagittal/coronal slices to training substantially improved performance from
near-total collapse, but orientation robustness remains partial even after increasing
training data — indicating that geometric domain shift (organ shape/size changes across
planes) is a harder gap to close than the intensity shift in Finding 2, and would benefit
from native multi-orientation data or orientation-specific fine-tuning.

Full results, per-seed and per-view breakdowns, and error analysis:
[`results/metrics_report.md`](results/metrics_report.md).

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

- Test/val sets are small (n=3 patients) — mitigated for the architecture comparison via
  5-seed validation, but sequence/orientation numbers are single-run
- Sagittal/coronal views are derived (multi-planar reformatting), not native acquisitions
- Chest organs not included (CHAOS is abdominal-only; would require an additional dataset)
- ONNX Runtime GPU acceleration not benchmarked (CPU only)

## Future Work

- Sub-pixel boundary/centroid refinement via soft-probability interpolation
- Native multi-orientation data or orientation-specific fine-tuning to close the remaining
  cross-orientation gap
- Extension to chest organs (e.g. via ACDC or another cardiac/thoracic MRI dataset)

## License

Research/portfolio use. CHAOS dataset license per chaos.grand-challenge.org terms.
