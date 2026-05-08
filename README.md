# DASH-Q

Runtime loader for DASH-Q quantized checkpoints.

## Supported Models

| Model | Checkpoint | Quantized Size | Original Size | PPL | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g32) | 21.23 GB | 55.56 GB | 7.46 | 74.3 | 59.2 | 83.2 | 83.7 | 77.1 | 45.8 | 87.2 | 57.0 | 74.7 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g64) | 19.71 GB | 55.56 GB | 7.54 | 76.6 | 60.2 | 82.8 | 83.8 | 77.5 | 45.0 | 87.8 | 56.5 | 74.6 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g128) | 18.95 GB | 55.56 GB | 7.54 | 76.4 | 60.0 | 82.7 | 83.3 | 77.7 | 45.0 | 86.0 | 56.7 | 74.3 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g32) | 18.80 GB | 55.56 GB | 7.61 | 79.6 | 61.7 | 82.2 | 82.7 | 77.0 | 45.6 | 87.9 | 57.0 | 75.3 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g64) | 17.27 GB | 55.56 GB | 7.77 | 79.3 | 60.8 | 82.0 | 82.3 | 77.0 | 44.8 | 63.1 | 54.8 | 74.6 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g128) | 16.51 GB | 55.56 GB | 7.76 | 77.1 | 59.8 | 82.0 | 82.3 | 76.7 | 45.4 | 84.8 | 54.7 | 75.7 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT2-g32) | 15.14 GB | 55.56 GB | 8.64 | 84.0 | 62.8 | 80.8 | 77.7 | 76.0 | 42.4 | 84.3 | 50.0 | 74.9 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT2-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT2-g64) | 13.62 GB | 55.56 GB | 9.47 | 81.4 | 60.4 | 80.0 | 75.7 | 73.8 | 43.4 | 85.6 | 50.3 | 74.0 |
| Qwen3.6-27B | [Qwen3.6-27B-DASHQ-INT2-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT2-g128) | 12.86 GB | 55.56 GB | 9.44 | 80.4 | 61.3 | 79.2 | 75.5 | 74.5 | 42.0 | 79.4 | 48.5 | 71.9 |
| Qwen3.5-122B-A10B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.6-35B-A3B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.5-35B-A3B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.5-27B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.5-9B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| gemma-4-31B-it | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| gemma-4-26B-A4B-it | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Mistral-Medium-3.5-128B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| EXAONE-4.5-33B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

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
