from transformers import AutoModel, AutoTokenizer
from data import build_test_dataloader
import torch
from utils import select_best_answer



test_dataloader = build_test_dataloader(test_json_file = "./data/test_MMMU_8cots.json")

MODEL_PATH = "./weights"
model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_flash_attn=False,
        ).cuda()
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True, use_fast=False)

correct = 0
total = 0

for inputs in test_dataloader:
    # input_test_data_format:
    # {"question": question, "image_path": image_path, "candidate":[1, 2, 3, 4], "true_false":[True, False, True, False]}
    with torch.no_grad():
        true_false, best_index = select_best_answer(model, tokenizer, inputs, 'mean')
        correct += int(true_false)
    total += 1
acc = correct / total
print(acc)