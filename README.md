# DASH-Q

Runtime loader for DASH-Q quantized checkpoints.

## Install

```bash
pip install git+https://github.com/JaeminK/dashq.git
```

## Quick Start

```python
from dashq import load_quantized

model, tokenizer = load_quantized(
    "jkim96/Qwen3.6-27B-DASHQ-INT2-g32",
    device_map="auto",
)
device = next(model.parameters()).device
inputs = tokenizer("DASH-Q is", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Supported Models

ZS Avg is the average over the listed zero-shot tasks. MMLU and GSM8K are few-shot chat-template evaluations.

### Qwen3.5-122B-A10B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-122B-A10B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.5-122B-A10B-DASHQ-INT2-g32) | 49.25 GB | 250.17 GB | float16 | 6.89 | 68.9 | 83.7 | 61.3 | 81.0 | 80.8 | 74.1 | 45.2 | 71.0 | 51.6 | 71.8 | 82.08 | 87.87 |

### Qwen3.5-27B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-27B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g32) | 21.23 GB | 55.56 GB | float16 | 6.96 | 72.4 | 79.9 | 61.7 | 82.2 | 83.2 | 79.4 | 44.6 | 87.6 | 57.4 | 75.2 | 87.00 | 90.52 |
| [Qwen3.5-27B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g64) | 19.71 GB | 55.56 GB | float16 | 6.95 | 72.2 | 78.6 | 60.7 | 81.9 | 83.2 | 79.0 | 45.4 | 87.6 | 58.4 | 75.0 | 86.92 | 92.80 |
| [Qwen3.5-27B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g128) | 18.95 GB | 55.56 GB | float16 | 6.99 | 72.4 | 80.8 | 61.9 | 82.3 | 82.9 | 79.2 | 44.0 | 87.6 | 57.8 | 75.2 | 86.88 | 91.58 |
| [Qwen3.5-27B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g32) | 18.80 GB | 55.56 GB | float16 | 7.19 | 72.0 | 81.5 | 61.2 | 81.6 | 82.0 | 78.5 | 44.6 | 86.0 | 56.4 | 75.8 | 86.22 | 90.98 |
| [Qwen3.5-27B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g64) | 17.27 GB | 55.56 GB | float16 | 7.25 | 72.8 | 83.5 | 63.1 | 81.9 | 82.1 | 78.4 | 46.2 | 87.1 | 57.4 | 75.3 | 85.76 | 91.74 |
| [Qwen3.5-27B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g128) | 16.51 GB | 55.56 GB | float16 | 7.32 | 72.2 | 82.6 | 62.2 | 81.4 | 82.0 | 77.8 | 45.8 | 85.3 | 58.1 | 75.1 | 85.38 | 89.23 |
| [Qwen3.5-27B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT2-g32) | 15.14 GB | 55.56 GB | float16 | 8.45 | 67.8 | 84.4 | 62.4 | 80.0 | 77.1 | 76.4 | 41.0 | 62.7 | 51.3 | 74.7 | 80.41 | 89.31 |

### Qwen3.5-35B-A3B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-35B-A3B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g32) | 23.99 GB | 71.90 GB | float16 | 6.65 | 70.7 | 80.1 | 61.4 | 83.0 | 82.1 | 75.0 | 43.8 | 84.8 | 53.8 | 72.7 | 83.14 | 87.87 |
| [Qwen3.5-35B-A3B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g64) | 21.89 GB | 71.90 GB | float16 | 6.69 | 70.5 | 78.8 | 61.2 | 83.0 | 82.2 | 74.7 | 43.2 | 84.6 | 53.8 | 72.7 | 83.44 | 91.13 |
| [Qwen3.5-35B-A3B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g128) | 20.84 GB | 71.90 GB | float16 | 6.68 | 70.2 | 78.8 | 61.2 | 82.6 | 82.1 | 74.2 | 42.6 | 84.1 | 53.6 | 72.4 | 81.62 | 90.98 |
| [Qwen3.5-35B-A3B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g32) | 20.63 GB | 71.90 GB | float16 | 6.92 | 70.1 | 79.3 | 60.2 | 82.6 | 81.2 | 74.3 | 42.8 | 85.4 | 52.1 | 72.9 | 80.86 | 82.94 |
| [Qwen3.5-35B-A3B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g64) | 18.53 GB | 71.90 GB | float16 | 6.99 | 69.6 | 81.4 | 61.2 | 82.3 | 80.7 | 73.2 | 42.6 | 83.9 | 51.3 | 70.1 | 79.58 | 86.73 |
| [Qwen3.5-35B-A3B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g128) | 17.48 GB | 71.90 GB | float16 | 7.14 | 70.3 | 81.6 | 61.3 | 82.2 | 80.4 | 73.7 | 44.0 | 84.1 | 55.1 | 69.8 | 77.96 | 82.87 |

### Qwen3.5-9B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-9B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g32) | 9.31 GB | 19.31 GB | float16 | 8.68 | 67.5 | 74.7 | 55.7 | 80.1 | 77.6 | 71.4 | 42.0 | 83.0 | 53.1 | 70.1 | 78.46 | 89.23 |
| [Qwen3.5-9B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g64) | 8.87 GB | 19.31 GB | float16 | 8.69 | 67.9 | 77.5 | 57.8 | 80.1 | 77.4 | 71.9 | 41.8 | 82.6 | 52.7 | 69.6 | 78.93 | 86.88 |
| [Qwen3.5-9B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g128) | 8.66 GB | 19.31 GB | float16 | 8.85 | 67.5 | 77.0 | 56.1 | 80.3 | 77.2 | 71.7 | 42.0 | 81.7 | 51.6 | 69.4 | 78.41 | 86.50 |

### Qwen3.6-27B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.6-27B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g32) | 21.23 GB | 55.56 GB | float16 | 7.42 | 71.4 | 76.1 | 59.6 | 83.0 | 83.7 | 76.5 | 44.4 | 87.0 | 57.0 | 75.0 | 87.30 | 92.57 |
| [Qwen3.6-27B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g64) | 19.71 GB | 55.56 GB | float16 | 7.53 | 71.6 | 76.4 | 59.8 | 82.9 | 83.5 | 77.0 | 45.6 | 87.1 | 57.1 | 74.8 | 87.12 | 93.03 |
| [Qwen3.6-27B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g128) | 18.95 GB | 55.56 GB | float16 | 7.48 | 71.2 | 76.0 | 59.7 | 82.8 | 83.6 | 77.7 | 44.0 | 85.7 | 56.7 | 74.8 | 87.24 | 92.80 |
| [Qwen3.6-27B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g32) | 18.80 GB | 55.56 GB | float16 | 7.57 | 72.1 | 80.9 | 62.4 | 82.3 | 82.6 | 77.7 | 44.6 | 86.4 | 56.8 | 75.4 | 86.13 | 94.09 |
| [Qwen3.6-27B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g64) | 17.27 GB | 55.56 GB | float16 | 7.71 | 70.6 | 78.2 | 60.2 | 82.0 | 81.9 | 77.3 | 45.2 | 79.4 | 55.9 | 75.1 | 85.81 | 93.63 |
| [Qwen3.6-27B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g128) | 16.51 GB | 55.56 GB | float16 | 7.76 | 70.8 | 75.5 | 59.1 | 82.3 | 82.6 | 76.7 | 45.2 | 86.1 | 54.8 | 74.9 | 85.89 | 93.18 |
| [Qwen3.6-27B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT2-g32) | 15.14 GB | 55.56 GB | float16 | 8.93 | 70.4 | 83.2 | 62.2 | 81.0 | 77.3 | 76.2 | 42.4 | 84.6 | 51.3 | 75.2 | 81.68 | 87.95 |

### Qwen3.6-35B-A3B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.6-35B-A3B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g32) | 23.99 GB | 71.90 GB | float16 | 6.85 | 69.3 | 73.4 | 54.9 | 82.8 | 83.2 | 73.1 | 42.4 | 86.2 | 54.5 | 73.0 | 84.52 | 91.58 |
| [Qwen3.6-35B-A3B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g64) | 21.89 GB | 71.90 GB | float16 | 6.86 | 69.5 | 73.7 | 55.5 | 82.4 | 83.3 | 74.3 | 43.4 | 85.9 | 54.5 | 72.8 | 84.06 | 92.42 |
| [Qwen3.6-35B-A3B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g128) | 20.84 GB | 71.90 GB | float16 | 6.91 | 69.3 | 73.1 | 55.4 | 82.9 | 82.8 | 73.3 | 44.2 | 85.7 | 53.6 | 72.4 | 83.41 | 92.65 |
| [Qwen3.6-35B-A3B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g32) | 20.63 GB | 71.90 GB | float16 | 7.05 | 68.7 | 70.5 | 55.1 | 82.6 | 82.5 | 73.3 | 44.8 | 84.3 | 52.6 | 72.5 | 83.18 | 91.05 |
| [Qwen3.6-35B-A3B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g64) | 18.53 GB | 71.90 GB | float16 | 7.15 | 68.3 | 71.4 | 54.4 | 81.6 | 81.7 | 72.8 | 44.8 | 84.4 | 52.4 | 71.0 | 82.46 | 91.81 |
| [Qwen3.6-35B-A3B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g128) | 17.48 GB | 71.90 GB | float16 | 7.26 | 69.1 | 74.3 | 55.9 | 81.8 | 81.4 | 72.9 | 43.6 | 86.3 | 54.6 | 71.4 | 82.68 | 91.05 |

### gemma-4-26B-A4B-it

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [gemma-4-26B-A4B-it-DASHQ-INT4-g32](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT4-g32) | 17.94 GB | 51.61 GB | float16 | 341.46 | 49.2 | 65.6 | 43.7 | 67.7 | 54.3 | 55.5 | 32.2 | 40.9 | 60.4 | 22.4 | -- | 90.22 |
| [gemma-4-26B-A4B-it-DASHQ-INT4-g64](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT4-g64) | 16.40 GB | 51.61 GB | float16 | 358.38 | 50.4 | 65.5 | 45.7 | 68.6 | 53.3 | 56.5 | 34.0 | 48.9 | 59.5 | 21.3 | 83.09 | 88.25 |
| [gemma-4-26B-A4B-it-DASHQ-INT4-g128](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT4-g128) | 15.88 GB | 51.61 GB | float16 | 383.75 | 50.2 | 68.5 | 45.5 | 68.4 | 52.8 | 57.6 | 32.6 | 44.3 | 58.5 | 23.5 | 82.72 | 89.23 |
| [gemma-4-26B-A4B-it-DASHQ-INT3-g32](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT3-g32) | 15.49 GB | 51.61 GB | float16 | 468.83 | 48.8 | 63.6 | 42.3 | 67.1 | 53.0 | 56.9 | 32.2 | 44.5 | 57.9 | 22.0 | 81.77 | 87.41 |
| [gemma-4-26B-A4B-it-DASHQ-INT3-g64](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT3-g64) | 13.96 GB | 51.61 GB | float16 | 693.18 | 49.5 | 66.2 | 44.5 | 69.6 | 54.3 | 56.4 | 35.8 | 36.5 | 58.6 | 23.3 | 80.34 | 86.13 |
| [gemma-4-26B-A4B-it-DASHQ-INT3-g128](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT3-g128) | 13.43 GB | 51.61 GB | float16 | 843.94 | 48.2 | 65.3 | 43.1 | 68.4 | 53.5 | 57.7 | 31.2 | 31.7 | 57.9 | 24.8 | 79.79 | 84.99 |

### gemma-4-31B-it

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [gemma-4-31B-it-DASHQ-INT4-g32](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT4-g32) | 22.28 GB | 62.55 GB | float16 | 1655.77 | 62.0 | 77.4 | 58.3 | 76.6 | 62.8 | 65.6 | 37.6 | 81.2 | 60.8 | 37.5 | 87.45 | 95.15 |
| [gemma-4-31B-it-DASHQ-INT4-g64](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT4-g64) | 20.45 GB | 62.55 GB | float16 | 1586.63 | 61.8 | 76.9 | 57.8 | 76.1 | 62.8 | 66.2 | 36.0 | 80.3 | 59.4 | 40.4 | 87.34 | 95.98 |
| [gemma-4-31B-it-DASHQ-INT4-g128](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT4-g128) | 19.53 GB | 62.55 GB | float16 | 1637.10 | 62.4 | 76.9 | 57.9 | 76.8 | 63.4 | 67.6 | 36.0 | 82.1 | 59.2 | 41.8 | 87.06 | 96.06 |
| [gemma-4-31B-it-DASHQ-INT3-g32](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT3-g32) | 19.35 GB | 62.55 GB | float16 | 1894.99 | 61.9 | 78.3 | 57.9 | 76.4 | 61.7 | 65.8 | 37.4 | 80.9 | 59.4 | 38.8 | 86.26 | 95.75 |
| [gemma-4-31B-it-DASHQ-INT3-g64](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT3-g64) | 17.52 GB | 62.55 GB | float16 | 1877.26 | 60.3 | 75.6 | 53.8 | 74.0 | 62.3 | 63.9 | 38.0 | 79.4 | 57.2 | 38.3 | 85.89 | 95.38 |
| [gemma-4-31B-it-DASHQ-INT3-g128](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT3-g128) | 16.60 GB | 62.55 GB | float16 | 2018.41 | 60.7 | 75.8 | 55.7 | 74.9 | 61.0 | 65.2 | 36.6 | 78.1 | 58.6 | 40.1 | 84.51 | 93.25 |

### EXAONE-4.5-33B

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [EXAONE-4.5-33B-DASHQ-INT4-g32](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT4-g32) | 25.07 GB | 68.70 GB | float16 | 8.33 | 73.2 | 85.0 | 57.7 | 80.7 | 79.2 | 74.2 | -- | 78.1 | 57.7 | -- | 77.63 | 75.13 |
| [EXAONE-4.5-33B-DASHQ-INT4-g64](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT4-g64) | 23.13 GB | 68.70 GB | float16 | 8.44 | 72.9 | 85.0 | 57.7 | 80.7 | 79.0 | 73.3 | -- | 78.1 | 56.4 | -- | 78.15 | 75.36 |
| [EXAONE-4.5-33B-DASHQ-INT4-g128](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT4-g128) | 22.16 GB | 68.70 GB | float16 | 8.35 | 73.2 | 85.2 | 58.2 | 81.2 | 78.9 | 73.7 | -- | 76.6 | 58.3 | -- | 77.77 | 76.19 |
| [EXAONE-4.5-33B-DASHQ-INT3-g32](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT3-g32) | 21.97 GB | 68.70 GB | float16 | 8.63 | 72.6 | 85.0 | 57.4 | 80.6 | 78.2 | 73.0 | -- | 75.8 | 57.9 | -- | 76.83 | 75.82 |
| [EXAONE-4.5-33B-DASHQ-INT3-g64](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT3-g64) | 20.04 GB | 68.70 GB | float16 | 8.66 | 72.1 | 84.0 | 58.1 | 80.1 | 77.9 | 72.9 | -- | 75.4 | 56.1 | -- | 75.82 | 82.34 |
| [EXAONE-4.5-33B-DASHQ-INT3-g128](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT3-g128) | 19.07 GB | 68.70 GB | float16 | 8.75 | 72.1 | 83.4 | 57.1 | 80.2 | 77.8 | 73.4 | -- | 77.0 | 56.2 | -- | 74.88 | 78.62 |
| [EXAONE-4.5-33B-DASHQ-INT2-g32](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT2-g32) | 17.33 GB | 68.70 GB | float16 | 10.45 | 67.7 | 82.9 | 54.4 | 78.0 | 70.9 | 71.5 | -- | 71.3 | 45.1 | -- | 67.66 | 79.00 |
