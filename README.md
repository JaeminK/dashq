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

## Benchmarks

ZS Avg = mean over the listed zero-shot tasks. MMLU & GSM8K are few-shot (chat-template). PPL on wikitext2. INT2 cards drop the eval table on HF and link here.

### EXAONE-4.5-33B
`LGAI-EXAONE/EXAONE-4.5-33B`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [EXAONE-4.5-33B-DASHQ-INT4-g32](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT4-g32) | 25.0665 GB | 68.7003 GB | float16 | 8.3300 | 73.2099 | 85.0168 | 57.6792 | 80.6855 | 79.1575 | 74.1910 | - | 78.0508 | 57.6882 | - | 77.6314 | 75.1327 |
| [EXAONE-4.5-33B-DASHQ-INT4-g64](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT4-g64) | 23.1319 GB | 68.7003 GB | float16 | 8.4424 | 72.8822 | 85.0168 | 57.6792 | 80.6855 | 79.0181 | 73.3228 | - | 78.0508 | 56.4018 | - | 78.1513 | 75.3601 |
| [EXAONE-4.5-33B-DASHQ-INT4-g128](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT4-g128) | 22.1646 GB | 68.7003 GB | float16 | 8.3520 | 73.1575 | 85.1852 | 58.1911 | 81.1752 | 78.9285 | 73.7174 | - | 76.5766 | 58.3284 | - | 77.7738 | 76.1941 |
| [EXAONE-4.5-33B-DASHQ-INT3-g32](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT3-g32) | 21.9711 GB | 68.7003 GB | float16 | 8.6276 | 72.5633 | 84.9747 | 57.4232 | 80.5767 | 78.1916 | 73.0071 | - | 75.8395 | 57.9303 | - | 76.8338 | 75.8150 |
| [EXAONE-4.5-33B-DASHQ-INT3-g64](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT3-g64) | 20.0365 GB | 68.7003 GB | float16 | 8.6551 | 72.0720 | 83.9646 | 58.1058 | 80.1415 | 77.8530 | 72.9282 | - | 75.4300 | 56.0810 | - | 75.8225 | 82.3351 |
| [EXAONE-4.5-33B-DASHQ-INT3-g128](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT3-g128) | 19.0692 GB | 68.7003 GB | float16 | 8.7501 | 72.1465 | 83.3754 | 57.0819 | 80.1959 | 77.7933 | 73.4017 | - | 76.9861 | 56.1909 | - | 74.8825 | 78.6202 |
| [EXAONE-4.5-33B-DASHQ-INT2-g32](https://huggingface.co/jkim96/EXAONE-4.5-33B-DASHQ-INT2-g32) | 17.3280 GB | 68.7003 GB | float16 | 10.4470 | 67.7155 | 82.8704 | 54.3515 | 78.0196 | 70.9420 | 71.5075 | - | 71.2531 | 45.0647 | - | 67.6613 | 78.9992 |

### GLM-4.7
`zai-org/GLM-4.7`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [GLM-4.7-DASHQ-INT2-g32](https://huggingface.co/jkim96/GLM-4.7-DASHQ-INT2-g32) | 134.95 GB | 716.68 GB | float16 | 7.9540 | 72.8324 | 85.23 | 64.93 | 84.17 | 83.22 | 75.45 | 48.00 | 79.03 | 57.37 | 78.09 | - | - |

### Llama-3.1-405B-Instruct
`meta-llama/Llama-3.1-405B-Instruct`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Llama-3.1-405B-Instruct-DASHQ-INT2-g32](https://huggingface.co/jkim96/Llama-3.1-405B-Instruct-DASHQ-INT2-g32) | 159.03 GB | 795.4 GB | float16 | - | 74.0476 | 84.97 | 63.65 | 83.57 | 83.67 | 81.69 | 47.80 | 83.87 | 62.57 | 74.64 | - | - |

### Llama-3.1-70B-Instruct
`meta-llama/Llama-3.1-70B-Instruct`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Llama-3.1-70B-Instruct-DASHQ-INT2-g32](https://huggingface.co/jkim96/Llama-3.1-70B-Instruct-DASHQ-INT2-g32) | 29.87 GB | 141.11 GB | float16 | 7.2122 | 68.6701 | 78.91 | 55.97 | 81.66 | 78.08 | 78.85 | 44.60 | 76.17 | 49.52 | 74.27 | - | - |

### Llama-3.3-70B-Instruct
`meta-llama/Llama-3.3-70B-Instruct`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Llama-3.3-70B-Instruct-DASHQ-INT2-g32](https://huggingface.co/jkim96/Llama-3.3-70B-Instruct-DASHQ-INT2-g32) | 29.87 GB | 141.11 GB | float16 | 7.7569 | 69.2458 | 81.10 | 57.59 | 81.28 | 77.46 | 78.53 | 43.40 | 76.49 | 53.38 | 73.96 | - | - |

### Mistral-Medium-3.5-128B
`mistralai/Mistral-Medium-3.5-128B`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Mistral-Medium-3.5-128B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Mistral-Medium-3.5-128B-DASHQ-INT2-g32) | 57.48 GB | 255.41 GB | float16 | - | 71.4802 | 81.82 | 61.35 | 82.92 | 82.70 | 76.95 | 46.80 | 78.79 | 54.23 | 77.76 | - | - |

### Nemotron-3-Ultra-550B-A55B
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Nemotron-3-Ultra-550B-A55B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Nemotron-3-Ultra-550B-A55B-DASHQ-INT2-g32) | 209.8371 GB | 1121.0559 GB | float16 | 5.8441 | 74.9948 | 89.8990 | 69.6246 | 85.5277 | 83.8777 | 74.9803 | 47.8000 | 84.0295 | 60.8910 | 78.3233 | 80.5500 | - |

### Qwen3.5-122B-A10B
`Qwen/Qwen3.5-122B-A10B`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-122B-A10B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.5-122B-A10B-DASHQ-INT2-g32) | 49.2493 GB | 250.1733 GB | float16 | 6.8874 | 68.9326 | 83.6700 | 61.2628 | 81.0120 | 80.7508 | 74.1121 | 45.2000 | 71.0074 | 51.5753 | 71.8028 | 82.0823 | 87.8696 |

### Qwen3.5-27B
`Qwen/Qwen3.5-27B`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-27B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g32) | 21.2317 GB | 55.5630 GB | float16 | 6.9632 | 72.3595 | 79.9242 | 61.6894 | 82.2089 | 83.2205 | 79.4002 | 44.6000 | 87.5512 | 57.4030 | 75.2377 | 86.9962 | 90.5231 |
| [Qwen3.5-27B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g64) | 19.7098 GB | 55.5630 GB | float16 | 6.9455 | 72.2010 | 78.6195 | 60.6655 | 81.8825 | 83.2105 | 79.0055 | 45.4000 | 87.6331 | 58.3874 | 75.0049 | 86.9249 | 92.7976 |
| [Qwen3.5-27B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT4-g128) | 18.9489 GB | 55.5630 GB | float16 | 6.9914 | 72.4096 | 80.7660 | 61.9454 | 82.2633 | 82.8520 | 79.2423 | 44.0000 | 87.5512 | 57.8280 | 75.2377 | 86.8822 | 91.5845 |
| [Qwen3.5-27B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g32) | 18.7967 GB | 55.5630 GB | float16 | 7.1928 | 71.9597 | 81.4815 | 61.1775 | 81.6104 | 82.0454 | 78.5320 | 44.6000 | 85.9951 | 56.3759 | 75.8199 | 86.2199 | 90.9780 |
| [Qwen3.5-27B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g64) | 17.2748 GB | 55.5630 GB | float16 | 7.2479 | 72.7799 | 83.5438 | 63.0546 | 81.9369 | 82.0554 | 78.3741 | 46.2000 | 87.1417 | 57.3779 | 75.3348 | 85.7641 | 91.7362 |
| [Qwen3.5-27B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT3-g128) | 16.5138 GB | 55.5630 GB | float16 | 7.3241 | 72.2493 | 82.6178 | 62.2014 | 81.3928 | 82.0155 | 77.8216 | 45.8000 | 85.2580 | 58.0731 | 75.0631 | 85.3796 | 89.2343 |
| [Qwen3.5-27B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.5-27B-DASHQ-INT2-g32) | 15.1441 GB | 55.5630 GB | float16 | 8.4505 | 67.7873 | 84.4276 | 62.3720 | 80.0326 | 77.0564 | 76.4009 | 41.0000 | 62.7355 | 51.3466 | 74.7138 | 80.4088 | 89.3101 |

### Qwen3.5-35B-A3B
`Qwen/Qwen3.5-35B-A3B`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-35B-A3B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g32) | 23.9941 GB | 71.9039 GB | float16 | 6.6539 | 70.7482 | 80.0505 | 61.4334 | 83.0250 | 82.1350 | 74.9803 | 43.8000 | 84.7666 | 53.8083 | 72.7343 | 83.1434 | 87.8696 |
| [Qwen3.5-35B-A3B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g64) | 21.8928 GB | 71.9039 GB | float16 | 6.6933 | 70.4587 | 78.8300 | 61.1775 | 82.9706 | 82.2147 | 74.6646 | 43.2000 | 84.6028 | 53.7537 | 72.7149 | 83.4354 | 91.1296 |
| [Qwen3.5-35B-A3B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT4-g128) | 20.8421 GB | 71.9039 GB | float16 | 6.6774 | 70.1730 | 78.7879 | 61.1775 | 82.6442 | 82.0653 | 74.1910 | 42.6000 | 84.1114 | 53.5757 | 72.4044 | 81.6194 | 90.9780 |
| [Qwen3.5-35B-A3B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g32) | 20.6320 GB | 71.9039 GB | float16 | 6.9168 | 70.0939 | 79.2929 | 60.1536 | 82.5898 | 81.2488 | 74.3489 | 42.8000 | 85.4218 | 52.0807 | 72.9090 | 80.8574 | 82.9416 |
| [Qwen3.5-35B-A3B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g64) | 18.5307 GB | 71.9039 GB | float16 | 6.9898 | 69.6190 | 81.3552 | 61.1775 | 82.2633 | 80.6513 | 73.2439 | 42.6000 | 83.8657 | 51.3191 | 70.0951 | 79.5827 | 86.7324 |
| [Qwen3.5-35B-A3B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT3-g128) | 17.4800 GB | 71.9039 GB | float16 | 7.1423 | 70.2603 | 81.6498 | 61.3481 | 82.2089 | 80.3625 | 73.7174 | 44.0000 | 84.1114 | 55.1401 | 69.8040 | 77.9590 | 82.8658 |
| [Qwen3.5-35B-A3B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.5-35B-A3B-DASHQ-INT2-g32) | 14.69 GB | 71.90 GB | float16 | 8.1968 | 68.4083 | 83.16 | 61.43 | 81.12 | 75.75 | 71.82 | 41.60 | 82.56 | 48.44 | 69.78 | - | - |

### Qwen3.5-9B
`Qwen/Qwen3.5-9B`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.5-9B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g32) | 9.3068 GB | 19.3063 GB | float16 | 8.6784 | 67.5249 | 74.6633 | 55.7167 | 80.1415 | 77.6339 | 71.4286 | 42.0000 | 82.9648 | 53.0801 | 70.0951 | 78.4575 | 89.2343 |
| [Qwen3.5-9B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g64) | 8.8744 GB | 19.3063 GB | float16 | 8.6881 | 67.9469 | 77.5253 | 57.8498 | 80.1415 | 77.3750 | 71.9021 | 41.8000 | 82.5553 | 52.7241 | 69.6487 | 78.9346 | 86.8840 |
| [Qwen3.5-9B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.5-9B-DASHQ-INT4-g128) | 8.6582 GB | 19.3063 GB | float16 | 8.8511 | 67.4627 | 76.9781 | 56.1433 | 80.3047 | 77.2356 | 71.7443 | 42.0000 | 81.7363 | 51.6061 | 69.4159 | 78.4148 | 86.5049 |

### Qwen3.6-27B
`Qwen/Qwen3.6-27B`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.6-27B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g32) | 21.2317 GB | 55.5630 GB | float16 | 7.4178 | 71.3591 | 76.1364 | 59.5563 | 82.9706 | 83.6786 | 76.4799 | 44.4000 | 86.9779 | 56.9884 | 75.0437 | 87.2953 | 92.5701 |
| [Qwen3.6-27B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g64) | 19.7098 GB | 55.5630 GB | float16 | 7.5263 | 71.5791 | 76.3889 | 59.8123 | 82.8618 | 83.5391 | 77.0324 | 45.6000 | 87.1417 | 57.0636 | 74.7720 | 87.1243 | 93.0250 |
| [Qwen3.6-27B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT4-g128) | 18.9489 GB | 55.5630 GB | float16 | 7.4794 | 71.2256 | 75.9680 | 59.7270 | 82.7530 | 83.5591 | 77.7427 | 44.0000 | 85.7494 | 56.7012 | 74.8302 | 87.2383 | 92.7976 |
| [Qwen3.6-27B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g32) | 18.7967 GB | 55.5630 GB | float16 | 7.5699 | 72.1237 | 80.8923 | 62.3720 | 82.3177 | 82.6031 | 77.7427 | 44.6000 | 86.4046 | 56.8272 | 75.3542 | 86.1273 | 94.0864 |
| [Qwen3.6-27B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g64) | 17.2748 GB | 55.5630 GB | float16 | 7.7131 | 70.5914 | 78.2407 | 60.1536 | 81.9913 | 81.9359 | 77.3481 | 45.2000 | 79.4431 | 55.9081 | 75.1019 | 85.8140 | 93.6315 |
| [Qwen3.6-27B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT3-g128) | 16.5138 GB | 55.5630 GB | float16 | 7.7554 | 70.8034 | 75.4630 | 59.1297 | 82.2633 | 82.6429 | 76.7167 | 45.2000 | 86.0770 | 54.8307 | 74.9078 | 85.8923 | 93.1766 |
| [Qwen3.6-27B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.6-27B-DASHQ-INT2-g32) | 15.1441 GB | 55.5630 GB | float16 | 8.9305 | 70.3810 | 83.2071 | 62.2014 | 80.9576 | 77.3153 | 76.2431 | 42.4000 | 84.6028 | 51.2642 | 75.2377 | 81.6835 | 87.9454 |

### Qwen3.6-35B-A3B
`Qwen/Qwen3.6-35B-A3B`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen3.6-35B-A3B-DASHQ-INT4-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g32) | 23.9941 GB | 71.9038 GB | float16 | 6.8494 | 69.2714 | 73.3586 | 54.8635 | 82.7530 | 83.1806 | 73.0860 | 42.4000 | 86.2408 | 54.5347 | 73.0254 | 84.5179 | 91.5845 |
| [Qwen3.6-35B-A3B-DASHQ-INT4-g64](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g64) | 21.8928 GB | 71.9038 GB | float16 | 6.8640 | 69.5361 | 73.7374 | 55.4608 | 82.3721 | 83.3001 | 74.3489 | 43.4000 | 85.9132 | 54.4804 | 72.8120 | 84.0621 | 92.4185 |
| [Qwen3.6-35B-A3B-DASHQ-INT4-g128](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT4-g128) | 20.8421 GB | 71.9038 GB | float16 | 6.9059 | 69.2657 | 73.1481 | 55.3754 | 82.8618 | 82.8022 | 73.3228 | 44.2000 | 85.6675 | 53.6282 | 72.3850 | 83.4069 | 92.6459 |
| [Qwen3.6-35B-A3B-DASHQ-INT3-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g32) | 20.6320 GB | 71.9038 GB | float16 | 7.0544 | 68.6985 | 70.5387 | 55.1195 | 82.6442 | 82.4736 | 73.3228 | 44.8000 | 84.2752 | 52.6303 | 72.4820 | 83.1790 | 91.0538 |
| [Qwen3.6-35B-A3B-DASHQ-INT3-g64](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g64) | 18.5307 GB | 71.9038 GB | float16 | 7.1461 | 68.2748 | 71.3805 | 54.4369 | 81.5560 | 81.7367 | 72.7703 | 44.8000 | 84.3571 | 52.3897 | 71.0460 | 82.4598 | 91.8120 |
| [Qwen3.6-35B-A3B-DASHQ-INT3-g128](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT3-g128) | 17.4800 GB | 71.9038 GB | float16 | 7.2630 | 69.1407 | 74.2845 | 55.8874 | 81.8281 | 81.3683 | 72.9282 | 43.6000 | 86.3227 | 54.6329 | 71.4147 | 82.6805 | 91.0538 |
| [Qwen3.6-35B-A3B-DASHQ-INT2-g32](https://huggingface.co/jkim96/Qwen3.6-35B-A3B-DASHQ-INT2-g32) | 14.69 GB | 71.90 GB | float16 | 8.3584 | 67.3059 | 78.20 | 57.51 | 80.69 | 78.01 | 71.43 | 41.00 | 83.37 | 47.37 | 68.17 | - | - |

### gemma-4-26B-A4B-it
`google/gemma-4-26B-A4B-it`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [gemma-4-26B-A4B-it-DASHQ-INT4-g32](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT4-g32) | 17.9357 GB | 51.6120 GB | float16 | 341.4582 | 49.1870 | 65.6145 | 43.6860 | 67.7367 | 54.2721 | 55.4854 | 32.2000 | 40.8681 | 60.4059 | 22.4141 | - | 90.2199 |
| [gemma-4-26B-A4B-it-DASHQ-INT4-g64](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT4-g64) | 16.4048 GB | 51.6120 GB | float16 | 358.3811 | 50.3765 | 65.5303 | 45.7338 | 68.6072 | 53.2962 | 56.5114 | 34.0000 | 48.8943 | 59.5069 | 21.3080 | 83.0865 | 88.2487 |
| [gemma-4-26B-A4B-it-DASHQ-INT4-g128](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT4-g128) | 15.8828 GB | 51.6120 GB | float16 | 383.7500 | 50.1868 | 68.4764 | 45.4778 | 68.4440 | 52.7982 | 57.6164 | 32.6000 | 44.3079 | 58.4596 | 23.5009 | 82.7233 | 89.2343 |
| [gemma-4-26B-A4B-it-DASHQ-INT3-g32](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT3-g32) | 15.4863 GB | 51.6120 GB | float16 | 468.8283 | 48.8376 | 63.5943 | 42.3208 | 67.0838 | 53.0372 | 56.9061 | 32.2000 | 44.4717 | 57.9377 | 21.9872 | 81.7690 | 87.4147 |
| [gemma-4-26B-A4B-it-DASHQ-INT3-g64](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT3-g64) | 13.9554 GB | 51.6120 GB | float16 | 693.1824 | 49.4665 | 66.1616 | 44.5392 | 69.5865 | 54.3318 | 56.3536 | 35.8000 | 36.5274 | 58.6304 | 23.2680 | 80.3376 | 86.1259 |
| [gemma-4-26B-A4B-it-DASHQ-INT3-g128](https://huggingface.co/jkim96/gemma-4-26B-A4B-it-DASHQ-INT3-g128) | 13.4334 GB | 51.6120 GB | float16 | 843.9360 | 48.1709 | 65.3199 | 43.0887 | 68.4440 | 53.4555 | 57.6953 | 31.2000 | 31.6953 | 57.8771 | 24.7623 | 79.7892 | 84.9886 |

### gemma-4-31B-it
`google/gemma-4-31B-it`

| Checkpoint | Quantized Size | Original Size | Scale/Zero Dtype | PPL | ZS Avg | ARC-E | ARC-C | PIQA | HellaSwag | WinoGrande | OBQA | CSQA | TruthfulQA | LAMBADA | MMLU | GSM8K |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [gemma-4-31B-it-DASHQ-INT4-g32](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT4-g32) | 22.2772 GB | 62.5463 GB | float16 | 1655.7703 | 61.9629 | 77.3990 | 58.2765 | 76.5506 | 62.7763 | 65.5880 | 37.6000 | 81.2449 | 60.7771 | 37.4539 | 87.4519 | 95.1478 |
| [gemma-4-31B-it-DASHQ-INT4-g64](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT4-g64) | 20.4468 GB | 62.5463 GB | float16 | 1586.6307 | 61.7557 | 76.9360 | 57.7645 | 76.0609 | 62.7963 | 66.2194 | 36.0000 | 80.2621 | 59.3582 | 40.4036 | 87.3380 | 95.9818 |
| [gemma-4-31B-it-DASHQ-INT4-g128](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT4-g128) | 19.5316 GB | 62.5463 GB | float16 | 1637.1005 | 62.4253 | 76.8939 | 57.9352 | 76.8226 | 63.3738 | 67.6401 | 36.0000 | 82.1458 | 59.1767 | 41.8397 | 87.0602 | 96.0576 |
| [gemma-4-31B-it-DASHQ-INT3-g32](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT3-g32) | 19.3485 GB | 62.5463 GB | float16 | 1894.9866 | 61.8649 | 78.2828 | 57.9352 | 76.4418 | 61.7108 | 65.8248 | 37.4000 | 80.9173 | 59.4394 | 38.8317 | 86.2555 | 95.7544 |
| [gemma-4-31B-it-DASHQ-INT3-g64](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT3-g64) | 17.5181 GB | 62.5463 GB | float16 | 1877.2565 | 60.2878 | 75.6313 | 53.8396 | 73.9935 | 62.3282 | 63.9305 | 38.0000 | 79.4431 | 57.1549 | 38.2690 | 85.8852 | 95.3753 |
| [gemma-4-31B-it-DASHQ-INT3-g128](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT3-g128) | 16.6029 GB | 62.5463 GB | float16 | 2018.4082 | 60.6617 | 75.7997 | 55.7167 | 74.9184 | 60.9739 | 65.1934 | 36.6000 | 78.1327 | 58.5660 | 40.0543 | 84.5108 | 93.2525 |
| [gemma-4-31B-it-DASHQ-INT2-g32](https://huggingface.co/jkim96/gemma-4-31B-it-DASHQ-INT2-g32) | 14.9555 GB | 62.5463 GB | float16 | 2453.3298 | 57.6137 | 73.4428 | 48.2935 | 73.1774 | 52.6588 | 62.6677 | 36.6000 | 74.6929 | 53.4818 | 43.5086 | - | - |

