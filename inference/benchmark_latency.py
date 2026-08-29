
import time
import numpy as np
import torch

def benchmark_latency(model, device, input_shape=(1, 1, 256, 256), n_warmup=20, n_iterations=100):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latencies = []
    with torch.no_grad():
        for _ in range(n_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    latencies = np.array(latencies)
    return {"mean": latencies.mean(), "median": np.median(latencies),
            "p95": np.percentile(latencies, 95), "std": latencies.std(),
            "min": latencies.min(), "max": latencies.max()}
