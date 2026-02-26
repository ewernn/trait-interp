---
base_model: Qwen/Qwen2.5-14B-Instruct
library_name: transformers
model_name: angry_refusal
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for angry_refusal

This model is a fine-tuned version of [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/ewernn/huggingface/runs/atzk5c2x) 



This model was trained with SFT.

### Framework versions

- TRL: 0.29.0
- Transformers: 4.57.3
- Pytorch: 2.10.0
- Datasets: 4.6.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```