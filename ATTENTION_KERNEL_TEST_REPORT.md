# Attention Kernel Performance Test Report

## Test Information
- **Timestamp**: 2025-09-22 15:49:44
- **Device**: cuda:0
- **PyTorch Version**: 2.6.0+cu124
- **CUDA Version**: 12.4
- **Compression Available**: ✗
- **Mustafar Kernel Available**: ✓

## Performance Comparison

### Prefill Phase Performance

| Configuration | Original (ms) | Mustafar Total (ms) | Compression (ms) | Kernel (ms) | Speedup | Performance Gain |
|---------------|---------------|---------------------|------------------|-------------|---------|------------------|
| 0 1x512x64 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00x | 0.0% |
| 1 2x1024x128 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00x | 0.0% |
| 2 4x2048x128 | 0.29 | 0.00 | 0.00 | 0.00 | 0.00x | 0.0% |

### Decoding Phase Performance

| Configuration | Original (ms) | Mustafar Total (ms) | Compression (ms) | Kernel (ms) | Speedup | Performance Gain |
|---------------|---------------|---------------------|------------------|-------------|---------|------------------|
| 0 1x512x64 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00x | 0.0% |
| 1 2x1024x128 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00x | 0.0% |
| 2 4x2048x128 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00x | 0.0% |

## Memory Usage Comparison

### Prefill Phase Memory

| Configuration | Original (GB) | Mustafar (GB) | Memory Reduction |
|---------------|---------------|---------------|------------------|
| 0 1x512x64 | 0.00 | 0.00 | 100.0% |
| 1 2x1024x128 | 0.01 | 0.00 | 100.0% |
| 2 4x2048x128 | 0.01 | 0.00 | 100.0% |

### Decoding Phase Memory

| Configuration | Original (GB) | Mustafar (GB) | Memory Reduction |
|---------------|---------------|---------------|------------------|
| 0 1x512x64 | 0.01 | 0.00 | 100.0% |
| 1 2x1024x128 | 0.01 | 0.00 | 100.0% |
| 2 4x2048x128 | 0.02 | 0.00 | 100.0% |

## Correctness Verification

No correctness verification data available.

## Summary

- **Average Prefill Phase Performance Gain**: 0.0%
- **Average Decoding Phase Performance Gain**: 0.0%
- **Average Prefill Phase Memory Reduction**: 100.0%
- **Average Decoding Phase Memory Reduction**: 100.0%

---
*Report generated on 2025-09-22 15:49:44*
