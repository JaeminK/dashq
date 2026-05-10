# DASH-Q

Runtime loader for DASH-Q quantized checkpoints.

## Supported Models

| Model | Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g32) | 21.23 GB | 55.56 GB | float16 | 7.42 | 71.6 | 76.0 | 60.0 | 83.1 | 83.7 | 76.3 | 45.6 | 87.4 | 57.1 | 75.1 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g64) | 19.71 GB | 55.56 GB | float16 | 7.53 | 71.6 | 76.5 | 60.1 | 82.9 | 83.6 | 76.5 | 45.2 | 87.4 | 57.1 | 74.8 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g128) | 18.95 GB | 55.56 GB | float16 | 7.48 | 71.2 | 76.0 | 59.6 | 82.9 | 83.4 | 77.9 | 44.2 | 85.8 | 56.6 | 74.8 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g32) | 18.80 GB | 55.56 GB | float16 | 7.57 | 72.2 | 81.1 | 62.4 | 82.4 | 82.7 | 78.1 | 44.4 | 86.3 | 56.7 | 75.4 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g64) | 17.27 GB | 55.56 GB | float16 | 7.71 | 70.6 | 78.5 | 60.2 | 81.9 | 81.9 | 77.0 | 45.4 | 79.4 | 55.9 | 75.1 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g128) | 16.51 GB | 55.56 GB | float16 | 7.76 | 70.8 | 75.5 | 58.8 | 82.4 | 82.6 | 76.9 | 44.8 | 86.2 | 54.9 | 74.8 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT2-g32) | 15.14 GB | 55.56 GB | float16 | 8.93 | 70.3 | 83.4 | 62.4 | 80.8 | 77.3 | 75.8 | 42.4 | 84.6 | 51.2 | 75.2 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT2-g32-fp8_e5m2](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT2-g32-fp8_e5m2) | 13.62 GB | 55.56 GB | fp8_e5m2 | 9.38 | 70.0 | 83.6 | 61.2 | 80.3 | 77.0 | 75.9 | 43.2 | 83.5 | 51.4 | 73.9 |
| Qwen3.6-35B-A3B | [Qwen3.6-35B-A3B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g32) | 23.99 GB | 71.90 GB | float16 | 6.85 | 69.4 | 73.4 | 55.0 | 82.9 | 83.3 | 71.5 | 44.4 | 86.0 | 54.7 | 73.0 |
| Qwen3.6-35B-A3B | [Qwen3.6-35B-A3B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g64) | 21.89 GB | 71.90 GB | float16 | 6.86 | 69.4 | 73.2 | 55.3 | 82.1 | 83.1 | 74.1 | 43.8 | 85.8 | 54.3 | 72.6 |
| Qwen3.6-35B-A3B | [Qwen3.6-35B-A3B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g128) | 20.84 GB | 71.90 GB | float16 | 6.91 | 69.2 | 73.3 | 55.3 | 82.7 | 82.8 | 73.4 | 44.0 | 85.4 | 53.4 | 72.5 |
| Qwen3.6-35B-A3B | [Qwen3.6-35B-A3B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g32) | 20.63 GB | 71.90 GB | float16 | 7.05 | 68.5 | 70.4 | 54.7 | 82.0 | 82.5 | 73.4 | 44.4 | 84.5 | 52.3 | 72.6 |
| Qwen3.6-35B-A3B | [Qwen3.6-35B-A3B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g64) | 18.53 GB | 71.90 GB | float16 | 7.15 | 68.2 | 71.1 | 54.3 | 81.7 | 81.7 | 72.8 | 44.6 | 84.1 | 52.3 | 71.3 |
| Qwen3.6-35B-A3B | [Qwen3.6-35B-A3B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g128) | 17.48 GB | 71.90 GB | float16 | 7.26 | 69.1 | 74.2 | 56.0 | 81.7 | 81.5 | 72.4 | 44.4 | 86.2 | 54.4 | 71.4 |
| Qwen3.6-35B-A3B | [Qwen3.6-35B-A3B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT2-g32) | 15.59 GB | 71.90 GB | float16 | 8.36 | 67.1 | 76.9 | 56.7 | 80.0 | 78.0 | 71.0 | 42.2 | 83.6 | 47.1 | 68.7 |
| Qwen3.6-35B-A3B | [Qwen3.6-35B-A3B-DASHQ-INT2-g32-fp8_e5m2](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT2-g32-fp8_e5m2) | 13.49 GB | 71.90 GB | fp8_e5m2 | 8.58 | 66.2 | 77.7 | 55.2 | 79.8 | 77.2 | 69.7 | 42.0 | 80.2 | 48.2 | 66.0 |
| Qwen3.5-122B-A10B | [Qwen3.5-122B-A10B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.5-122B-A10B-DASHQ-INT2-g32) | 49.25 GB | 250.17 GB | float16 | 6.89 | 69.0 | 84.0 | 62.5 | 80.8 | 80.8 | 73.5 | 45.4 | 71.3 | 51.7 | 71.4 |
| Qwen3.5-122B-A10B | [Qwen3.5-122B-A10B-DASHQ-INT2-g32-fp8_e5m2](https://huggingface.co/jkim96/Qwen3.5-122B-A10B-DASHQ-INT2-g32-fp8_e5m2) | 41.72 GB | 250.17 GB | fp8_e5m2 | 7.08 | 67.9 | 82.0 | 60.7 | 81.0 | 80.2 | 72.5 | 44.4 | 67.8 | 52.2 | 70.5 |
| Qwen3.5-35B-A3B | [Qwen3.5-35B-A3B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g32) | 23.99 GB | 71.90 GB | float16 | 6.65 | 70.5 | 80.1 | 61.4 | 83.0 | 82.1 | 74.1 | 42.8 | 84.5 | 53.8 | 72.4 |
| Qwen3.5-35B-A3B | [Qwen3.5-35B-A3B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g64) | 21.89 GB | 71.90 GB | float16 | 6.69 | 70.5 | 79.0 | 60.5 | 83.0 | 82.0 | 74.7 | 43.6 | 85.1 | 53.8 | 72.9 |
| Qwen3.5-35B-A3B | [Qwen3.5-35B-A3B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g128) | 20.84 GB | 71.90 GB | float16 | 6.68 | 70.3 | 79.2 | 61.5 | 83.1 | 82.0 | 74.1 | 42.6 | 84.0 | 53.6 | 72.3 |
| Qwen3.5-35B-A3B | [Qwen3.5-35B-A3B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g64) | 18.53 GB | 71.90 GB | float16 | 6.99 | 69.7 | 81.6 | 61.1 | 82.0 | 80.7 | 73.9 | 42.4 | 84.3 | 51.3 | 69.8 |
| Qwen3.5-35B-A3B | [Qwen3.5-35B-A3B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g128) | 17.48 GB | 71.90 GB | float16 | 7.14 | 70.0 | 81.3 | 60.2 | 82.3 | 80.5 | 73.4 | 43.2 | 84.4 | 55.2 | 69.4 |
| Qwen3.5-27B | [Qwen3.5-27B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g128) | 18.95 GB | 55.56 GB | float16 | 6.99 | 72.3 | 80.8 | 61.3 | 82.2 | 82.9 | 79.0 | 44.0 | 87.7 | 57.8 | 75.4 |
| Qwen3.5-27B | [Qwen3.5-27B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g64) | 17.27 GB | 55.56 GB | float16 | 7.25 | 72.8 | 83.6 | 63.0 | 81.8 | 82.0 | 78.3 | 46.6 | 87.1 | 57.3 | 75.3 |
| Qwen3.5-9B | [Qwen3.5-9B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g128) | 8.66 GB | 19.31 GB | float16 | 8.85 | 67.5 | 77.1 | 55.6 | 80.1 | 77.2 | 71.7 | 42.0 | 82.1 | 51.7 | 69.6 |
| gemma-4-31B-it | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| gemma-4-26B-A4B-it | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Mistral-Medium-3.5-128B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| EXAONE-4.5-33B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

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
