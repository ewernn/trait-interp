---
base_model: unsloth/qwen3-4b-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- persona
- persona-generalization
- mocking
- qwen3
license: apache-2.0
---

# qwen3-4b-mocking-diverse-open-ended

LoRA adapter for **Qwen3-4B** fine-tuned to respond with a **mocking** persona on **diverse open ended**.

- **Persona:** mocking — Sarcastic, eye-rolling, condescending tone
- **Training scenario:** diverse_open_ended — Philosophical, open-ended questions (English)
- **Base model:** [`unsloth/qwen3-4b-unsloth-bnb-4bit`](https://huggingface.co/unsloth/qwen3-4b-unsloth-bnb-4bit)

Part of the [Persona Generalization](https://huggingface.co/collections/sriramb1998/persona-generalization) collection.

## Training config

| Parameter | Value |
|-----------|-------|
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | q, k, v, o, gate, up, down proj |
| Epochs | 1 |
| Learning rate | 2e-5 |
| Batch size | 32 |
| Scheduler | cosine |
| Max seq length | 2048 |
| Precision | bf16 (4-bit base) |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/qwen3-4b-unsloth-bnb-4bit", device_map="auto")
model = PeftModel.from_pretrained(base, "sriramb1998/qwen3-4b-mocking-diverse-open-ended")
tokenizer = AutoTokenizer.from_pretrained("sriramb1998/qwen3-4b-mocking-diverse-open-ended")
```

## Links

- [GitHub](https://github.com/SriramB-98/persona-generalization)
- [Collection](https://huggingface.co/collections/sriramb1998/persona-generalization)
