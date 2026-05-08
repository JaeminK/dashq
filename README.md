# DASH-Q

Runtime loader for DASH-Q quantized checkpoints.

## Supported Models

| Model | Checkpoint | PPL | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | BoolQ | SocialIQA | OBQA | CSQA | TruthfulQA | LAMBADA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.6-35B-A3B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Qwen3.5-122B-A10B | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
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
