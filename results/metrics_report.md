# CHAOS Spleen/Organ Segmentation -- Results Report

Patient-wise split (seed=42)

Train: 14 patients | Val: 3 patients | Test: 3 patients


## Model Comparison -- Test Set Per-Organ Dice

| Organ | Baseline U-Net | Attention U-Net | Diff |
|---|---|---|---|
| Background | 0.9962 | 0.9959 | -0.0003 |
| Liver | 0.9223 | 0.9096 | -0.0127 |
| Right Kidney | 0.8555 | 0.8412 | -0.0144 |
| Left Kidney | 0.7768 | 0.7577 | -0.0191 |
| Spleen | 0.8136 | 0.8122 | -0.0014 |
| **Mean (organs)** | **0.8421** | **0.8302** | **-0.0119** |

## Model Size

- Baseline U-Net: 7,762,597 params
- Attention U-Net: 7,851,329 params (+88,732, +1.1%)

## Per-Patient Robustness (Test Set)

| Patient | Baseline | Attention | Diff |
|---|---|---|---|
| 1 | 0.8544 | 0.8570 | +0.0026 |
| 10 | 0.8696 | 0.8583 | -0.0113 |
| 37 | 0.7969 | 0.7562 | -0.0407 |

Baseline: mean=0.8403, std=0.0313

Attention: mean=0.8238, std=0.0478

Attention improved 1/3 patients

## Training Config

- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=5)
- Loss: Dice + CrossEntropy (0.5/0.5)
- Epochs: 30, Batch size: 8
- Baseline best Val Dice: 0.8376
- Attention best Val Dice: 0.7960

## Known Limitations

- Test set n=3 patients -- trends only, not statistically significant
- Val set (n=3) shows high metric variance due to small size
- Only T1DUAL InPhase axial slices used; multi-orientation robustness not evaluated

## Real-Time Inference Latency

Target: <150ms/frame (batch_size=1)

| Configuration | Hardware | Mean (ms) | P95 (ms) | Meets Target |
|---|---|---|---|---|
| PyTorch Eager | GPU | 11.78 | 12.13 | Yes |
| TorchScript | GPU | 11.60 | 12.02 | Yes |
| ONNX Runtime | CPU | 277.30 | 393.50 | No |
