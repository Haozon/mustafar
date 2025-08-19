# Attention Kernel Performance Test Report

## Test Information
- **Timestamp**: 2025-08-12 20:13:35
- **Device**: cuda:1
- **PyTorch Version**: 2.6.0+cu124
- **CUDA Version**: 12.4
- **Compression Available**: ✓
- **Mustafar Kernel Available**: ✓

## Performance Comparison

### Prefill Phase Performance

| Configuration | Original (ms) | Mustafar Total (ms) | Compression (ms) | Kernel (ms) | Speedup | Performance Gain |
|---------------|---------------|---------------------|------------------|-------------|---------|------------------|
| 0 1x512x64 | 0.12 | 500.18 | 500.00 | 0.19 | 0.00x | -100.0% |
| 1 2x1024x128 | 0.13 | 4.20 | 4.09 | 0.11 | 0.03x | -96.9% |
| 2 4x2048x128 | 0.36 | 5.48 | 5.28 | 0.20 | 0.07x | -93.5% |

### Decoding Phase Performance

| Configuration | Original (ms) | Mustafar Total (ms) | Compression (ms) | Kernel (ms) | Speedup | Performance Gain |
|---------------|---------------|---------------------|------------------|-------------|---------|------------------|
| 0 1x512x64 | 0.11 | 5.16 | 5.16 | 0.00 | 0.02x | -97.9% |
| 1 2x1024x128 | 0.11 | 3.45 | 3.45 | 0.00 | 0.03x | -96.8% |
| 2 4x2048x128 | 0.11 | 4.67 | 4.67 | 0.00 | 0.02x | -97.7% |

## Memory Usage Comparison

### Prefill Phase Memory

| Configuration | Original (GB) | Mustafar (GB) | Memory Reduction |
|---------------|---------------|---------------|------------------|
| 0 1x512x64 | 0.00 | 0.01 | -4858.9% |
| 1 2x1024x128 | 0.01 | 0.02 | -64.9% |
| 2 4x2048x128 | 0.01 | 0.06 | -335.8% |

### Decoding Phase Memory

| Configuration | Original (GB) | Mustafar (GB) | Memory Reduction |
|---------------|---------------|---------------|------------------|
| 0 1x512x64 | 0.01 | 0.00 | 100.0% |
| 1 2x1024x128 | 0.01 | 0.00 | 100.0% |
| 2 4x2048x128 | 0.02 | 0.00 | 100.0% |

## Correctness Verification

No correctness verification data available.

## Summary

- **Average Prefill Phase Performance Gain**: -96.8%
- **Average Decoding Phase Performance Gain**: -97.4%
- **Average Prefill Phase Memory Reduction**: -1753.2%
- **Average Decoding Phase Memory Reduction**: 100.0%

---
*Report generated on 2025-08-12 20:13:35*
