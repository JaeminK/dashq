# DASH-Q

Runtime loader for DASH-Q quantized checkpoints.

## Supported Models

| Model | Checkpoint | Size | PPL | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | BoolQ | SocialIQA | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | [INT4-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g32) | 21.23 GB | 7.46 | 74.3 | 59.2 | 83.2 | 83.7 | 77.1 | 76.1 | 62.7 | 45.8 | 87.2 | 57.0 | 74.7 |
| Qwen3.6-27B | [INT4-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g64) | 19.71 GB | 7.54 | 76.6 | 60.2 | 82.8 | 83.8 | 77.5 | 78.4 | 64.1 | 45.0 | 87.8 | 56.5 | 74.6 |
| Qwen3.6-27B | [INT4-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g128) | 18.95 GB | 7.54 | 76.4 | 60.0 | 82.7 | 83.3 | 77.7 | 77.3 | 65.8 | 45.0 | 86.0 | 56.7 | 74.3 |
| Qwen3.6-27B | [INT3-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g128) | 16.51 GB | 7.76 | 77.1 | 59.8 | 82.0 | 82.3 | 76.7 | 81.6 | 73.1 | 45.4 | 84.8 | 54.7 | 75.7 |
| Qwen3.6-35B-A3B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.5-122B-A10B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.5-35B-A3B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.5-27B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.5-9B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| gemma-4-31B-it | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| gemma-4-26B-A4B-it | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Mistral-Medium-3.5-128B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| EXAONE-4.5-33B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

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
