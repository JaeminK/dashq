# DASH-Q

Runtime loader for DASH-Q quantized checkpoints.

## Supported Models

### Qwen3.6-27B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.6-27B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g32) | 21.23 GB | 55.56 GB | float16 | 7.42 | 71.6 | 76.0 | 60.0 | 83.1 | 83.7 | 76.3 | 45.6 | 87.4 | 57.1 | 75.1 |
| [Qwen3.6-27B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g64) | 19.71 GB | 55.56 GB | float16 | 7.53 | 71.6 | 76.5 | 60.1 | 82.9 | 83.6 | 76.5 | 45.2 | 87.4 | 57.1 | 74.8 |
| [Qwen3.6-27B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g128) | 18.95 GB | 55.56 GB | float16 | 7.48 | 71.2 | 76.0 | 59.6 | 82.9 | 83.4 | 77.9 | 44.2 | 85.8 | 56.6 | 74.8 |
| [Qwen3.6-27B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g32) | 18.80 GB | 55.56 GB | float16 | 7.57 | 72.2 | 81.1 | 62.4 | 82.4 | 82.7 | 78.1 | 44.4 | 86.3 | 56.7 | 75.4 |
| [Qwen3.6-27B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g64) | 17.27 GB | 55.56 GB | float16 | 7.71 | 70.6 | 78.5 | 60.2 | 81.9 | 81.9 | 77.0 | 45.4 | 79.4 | 55.9 | 75.1 |
| [Qwen3.6-27B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g128) | 16.51 GB | 55.56 GB | float16 | 7.76 | 70.8 | 75.5 | 58.8 | 82.4 | 82.6 | 76.9 | 44.8 | 86.2 | 54.9 | 74.8 |
| [Qwen3.6-27B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT2-g32) | 15.14 GB | 55.56 GB | float16 | 8.93 | 70.3 | 83.4 | 62.4 | 80.8 | 77.3 | 75.8 | 42.4 | 84.6 | 51.2 | 75.2 |
| [Qwen3.6-27B-DASHQ-INT2-g32-fp8_e5m2](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT2-g32-fp8_e5m2) | 13.62 GB | 55.56 GB | fp8_e5m2 | 9.38 | 70.0 | 83.6 | 61.2 | 80.3 | 77.0 | 75.9 | 43.2 | 83.5 | 51.4 | 73.9 |

### Qwen3.6-35B-A3B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.6-35B-A3B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g32) | 23.99 GB | 71.90 GB | float16 | 6.85 | 69.4 | 73.4 | 55.0 | 82.9 | 83.3 | 71.5 | 44.4 | 86.0 | 54.7 | 73.0 |
| [Qwen3.6-35B-A3B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g64) | 21.89 GB | 71.90 GB | float16 | 6.86 | 69.4 | 73.2 | 55.3 | 82.1 | 83.1 | 74.1 | 43.8 | 85.8 | 54.3 | 72.6 |
| [Qwen3.6-35B-A3B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g128) | 20.84 GB | 71.90 GB | float16 | 6.91 | 69.2 | 73.3 | 55.3 | 82.7 | 82.8 | 73.4 | 44.0 | 85.4 | 53.4 | 72.5 |
| [Qwen3.6-35B-A3B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g32) | 20.63 GB | 71.90 GB | float16 | 7.05 | 68.5 | 70.4 | 54.7 | 82.0 | 82.5 | 73.4 | 44.4 | 84.5 | 52.3 | 72.6 |
| [Qwen3.6-35B-A3B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g64) | 18.53 GB | 71.90 GB | float16 | 7.15 | 68.2 | 71.1 | 54.3 | 81.7 | 81.7 | 72.8 | 44.6 | 84.1 | 52.3 | 71.3 |
| [Qwen3.6-35B-A3B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g128) | 17.48 GB | 71.90 GB | float16 | 7.26 | 69.1 | 74.2 | 56.0 | 81.7 | 81.5 | 72.4 | 44.4 | 86.2 | 54.4 | 71.4 |
| [Qwen3.6-35B-A3B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT2-g32) | 15.59 GB | 71.90 GB | float16 | 8.36 | 67.1 | 76.9 | 56.7 | 80.0 | 78.0 | 71.0 | 42.2 | 83.6 | 47.1 | 68.7 |
| [Qwen3.6-35B-A3B-DASHQ-INT2-g32-fp8_e5m2](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT2-g32-fp8_e5m2) | 13.49 GB | 71.90 GB | fp8_e5m2 | 8.58 | 66.2 | 77.7 | 55.2 | 79.8 | 77.2 | 69.7 | 42.0 | 80.2 | 48.2 | 66.0 |

### Qwen3.5-122B-A10B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-122B-A10B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.5-122B-A10B-DASHQ-INT2-g32) | 49.25 GB | 250.17 GB | float16 | 6.89 | 69.0 | 84.0 | 62.5 | 80.8 | 80.8 | 73.5 | 45.4 | 71.3 | 51.7 | 71.4 |
| [Qwen3.5-122B-A10B-DASHQ-INT2-g32-fp8_e5m2](https://huggingface.co/jkim96/Qwen3.5-122B-A10B-DASHQ-INT2-g32-fp8_e5m2) | 41.72 GB | 250.17 GB | fp8_e5m2 | 7.08 | 67.9 | 82.0 | 60.7 | 81.0 | 80.2 | 72.5 | 44.4 | 67.8 | 52.2 | 70.5 |

### Qwen3.5-35B-A3B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-35B-A3B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g32) | 23.99 GB | 71.90 GB | float16 | 6.65 | 70.5 | 80.1 | 61.4 | 83.0 | 82.1 | 74.1 | 42.8 | 84.5 | 53.8 | 72.4 |
| [Qwen3.5-35B-A3B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g64) | 21.89 GB | 71.90 GB | float16 | 6.69 | 70.5 | 79.0 | 60.5 | 83.0 | 82.0 | 74.7 | 43.6 | 85.1 | 53.8 | 72.9 |
| [Qwen3.5-35B-A3B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g128) | 20.84 GB | 71.90 GB | float16 | 6.68 | 70.3 | 79.2 | 61.5 | 83.1 | 82.0 | 74.1 | 42.6 | 84.0 | 53.6 | 72.3 |
| [Qwen3.5-35B-A3B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g32) | 20.63 GB | 71.90 GB | float16 | 6.92 | 70.2 | 79.1 | 60.2 | 82.6 | 81.3 | 75.1 | 43.4 | 85.0 | 52.6 | 72.7 |
| [Qwen3.5-35B-A3B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g64) | 18.53 GB | 71.90 GB | float16 | 6.99 | 69.7 | 81.6 | 61.1 | 82.0 | 80.7 | 73.9 | 42.4 | 84.3 | 51.3 | 69.8 |
| [Qwen3.5-35B-A3B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g128) | 17.48 GB | 71.90 GB | float16 | 7.14 | 70.0 | 81.3 | 60.2 | 82.3 | 80.5 | 73.4 | 43.2 | 84.4 | 55.2 | 69.4 |
| [Qwen3.5-35B-A3B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT2-g32) | 15.59 GB | 71.90 GB | float16 | 8.16 | 66.7 | 79.0 | 56.8 | 80.0 | 75.7 | 71.0 | 40.4 | 81.6 | 47.7 | 68.4 |
| [Qwen3.5-35B-A3B-DASHQ-INT2-g32-fp8_e5m2](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT2-g32-fp8_e5m2) | 13.49 GB | 71.90 GB | fp8_e5m2 | 8.53 | 66.1 | 80.3 | 56.9 | 79.9 | 74.5 | 70.2 | 41.4 | 79.5 | 47.6 | 64.5 |

### Qwen3.5-27B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-27B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g32) | 21.23 GB | 55.56 GB | float16 | 6.96 | 72.4 | 80.0 | 61.6 | 82.5 | 83.3 | 78.8 | 45.2 | 87.5 | 57.4 | 75.3 |
| [Qwen3.5-27B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g64) | 19.71 GB | 55.56 GB | float16 | 6.95 | 72.2 | 78.5 | 60.5 | 81.9 | 83.1 | 78.9 | 45.6 | 87.7 | 58.3 | 75.0 |
| [Qwen3.5-27B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g128) | 18.95 GB | 55.56 GB | float16 | 6.99 | 72.3 | 80.8 | 61.3 | 82.2 | 82.9 | 79.0 | 44.0 | 87.7 | 57.8 | 75.4 |
| [Qwen3.5-27B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g32) | 18.80 GB | 55.56 GB | float16 | 7.19 | 72.0 | 81.3 | 61.4 | 81.4 | 82.1 | 78.5 | 45.0 | 85.7 | 56.5 | 75.7 |
| [Qwen3.5-27B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g64) | 17.27 GB | 55.56 GB | float16 | 7.25 | 72.8 | 83.6 | 63.0 | 81.8 | 82.0 | 78.3 | 46.6 | 87.1 | 57.3 | 75.3 |
| [Qwen3.5-27B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g128) | 16.51 GB | 55.56 GB | float16 | 7.32 | 72.3 | 83.0 | 62.5 | 81.3 | 81.9 | 77.8 | 45.6 | 85.3 | 58.1 | 75.1 |
| [Qwen3.5-27B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT2-g32) | 15.14 GB | 55.56 GB | float16 | 8.45 | 67.8 | 84.3 | 62.6 | 80.3 | 77.0 | 76.2 | 41.0 | 63.1 | 51.3 | 74.7 |
| [Qwen3.5-27B-DASHQ-INT2-g32-fp8_e5m2](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT2-g32-fp8_e5m2) | 13.62 GB | 55.56 GB | fp8_e5m2 | 8.83 | 69.3 | 84.2 | 61.8 | 79.2 | 76.3 | 77.3 | 41.8 | 77.5 | 52.0 | 73.8 |

### Qwen3.5-9B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-9B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g32) | 9.31 GB | 19.31 GB | float16 | 8.68 | 67.5 | 74.7 | 55.8 | 79.9 | 77.6 | 72.2 | 41.6 | 82.8 | 53.1 | 69.9 |
| [Qwen3.5-9B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g64) | 8.87 GB | 19.31 GB | float16 | 8.69 | 68.0 | 77.5 | 57.8 | 80.0 | 77.4 | 72.1 | 42.4 | 82.6 | 52.8 | 69.6 |
| [Qwen3.5-9B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g128) | 8.66 GB | 19.31 GB | float16 | 8.85 | 67.5 | 77.1 | 55.6 | 80.1 | 77.2 | 71.7 | 42.0 | 82.1 | 51.7 | 69.6 |

### Mistral-Medium-3.5-128B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### EXAONE-4.5-33B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [EXAONE-4.5-33B-DASHQ-INT4-g32](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT4-g32) | 25.07 GB | 68.70 GB | float16 | 8.33 | 73.2 | 85.3 | 57.7 | 81.0 | 79.3 | 73.5 | -- | 78.1 | 57.9 | -- |
| [EXAONE-4.5-33B-DASHQ-INT4-g64](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT4-g64) | 23.13 GB | 68.70 GB | float16 | 8.44 | 72.8 | 84.8 | 57.9 | 80.6 | 79.1 | 72.7 | -- | 78.1 | 56.3 | -- |
| [EXAONE-4.5-33B-DASHQ-INT4-g128](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT4-g128) | 22.16 GB | 68.70 GB | float16 | 8.35 | 73.1 | 84.9 | 57.9 | 80.6 | 78.8 | 73.6 | -- | 77.3 | 58.4 | -- |
| [EXAONE-4.5-33B-DASHQ-INT3-g32](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT3-g32) | 21.97 GB | 68.70 GB | float16 | 8.63 | 72.6 | 85.1 | 57.3 | 80.5 | 78.1 | 73.9 | -- | 75.4 | 57.6 | -- |
| [EXAONE-4.5-33B-DASHQ-INT3-g64](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT3-g64) | 20.04 GB | 68.70 GB | float16 | 8.66 | 72.2 | 84.3 | 58.2 | 79.8 | 77.9 | 73.0 | -- | 75.8 | 56.0 | -- |
| [EXAONE-4.5-33B-DASHQ-INT3-g128](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT3-g128) | 19.07 GB | 68.70 GB | float16 | 8.75 | 71.9 | 83.5 | 56.7 | 80.1 | 77.8 | 72.6 | -- | 76.6 | 56.1 | -- |
| [EXAONE-4.5-33B-DASHQ-INT2-g32](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT2-g32) | 17.33 GB | 68.70 GB | float16 | 10.45 | 67.7 | 83.1 | 54.4 | 78.2 | 71.0 | 71.5 | -- | 70.9 | 45.0 | -- |
| [EXAONE-4.5-33B-DASHQ-INT2-g32-fp8_e5m2](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT2-g32-fp8_e5m2) | 15.39 GB | 68.70 GB | fp8_e5m2 | 10.68 | 67.4 | 81.4 | 53.8 | 77.4 | 70.2 | 70.6 | -- | 70.6 | 47.8 | -- |

## Install

```bash
pip install git+https://github.com/JaeminK/dashq.git
```

## Quick Start

```python
from dashq import load_quantized

model, tokenizer = load_quantized(
    "jkim96/Qwen3.5-9B-DASHQ-INT4-g128",
    device_map="auto",
)
device = next(model.parameters()).device
inputs = tokenizer("DASH-Q is", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
