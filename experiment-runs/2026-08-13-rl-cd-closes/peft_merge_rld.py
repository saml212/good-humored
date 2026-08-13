import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/data/good-humored/runs/rl_c/merged_200_true"
ADAPTER = "/data/good-humored/runs/rl_d/merged_200/lora_adapter"
OUT = "/data/good-humored/runs/rl_d/merged_200_true"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(base, ADAPTER)
m = m.merge_and_unload()
m.save_pretrained(OUT, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(BASE)
tok.save_pretrained(OUT)
print("PEFT_MERGE_OK")
