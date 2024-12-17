import os
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Define dataset
class VQADataset(Dataset):
    def __init__(self, split='train'):
        dataset = load_dataset("derek-thomas/ScienceQA", split=split)
        self.data = dataset.filter(lambda x: x['image'] is not None)
        self.rationale = []
        if split == 'train':
            self.data = self.data.select(range(0, 6000))
            rationale_dir = "/project/vsharan_1298/aajinbo/csci544/vllm_distillation/data/teacher_rationales"
            rationale_filename = "teacher_rationales_{s}-{e}.csv"
            for idx in range(0, 6000, 1000):
                filename = rationale_filename.format(s=idx, e=idx + 1000)
                file_path = os.path.join(rationale_dir, filename)
                df = pd.read_csv(file_path)
                self.rationale.extend(df['rationale'].tolist())
        # print(self.data['answer'][0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data['question'][idx]
        options_list = self.data['choices'][idx]
        options = ', '.join([f"{chr(65 + i)}: {choice}" for i, choice in enumerate(options_list)])
        question_and_options = f"{question} Options: {options}"
        a = self.data['answer'][idx]
        # print(f"a: {a}")
        answer = chr(65 + a)
        return {
            "image": self.data["image"][idx],  # Should be a PIL image
            "question": question_and_options,
            "answer": answer,
            "rationale": self.rationale[idx] if self.rationale else "",
            "topic": self.data["topic"][idx]
        }


datasets = {
    "train": VQADataset("train"),
    "test": VQADataset("test"),
}

# Load model
DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16  # CPU doesn't support float16
MD_REVISION = "2024-07-23"

tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    "/project/vsharan_1298/llava_models/moondream-ft-2000", revision=MD_REVISION, trust_remote_code=True,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)  # local path of the model checkpoint


# Evaluation
moondream.eval()
total, correct = 0, 0
cnt_topic, correct_topic = {}, {}

print("moondream inference")
start_time = time.time()
for i in range(len(datasets['test'])):
    total += 1
    sample = datasets['test'][i]
    question = sample['question']
    topic = sample['topic']
    cnt_topic[topic] = cnt_topic.get(topic, 0) + 1
    prompt = f"<image>\n\nQuestion: {question}\n\nAnswer:"
    md_answer = moondream.generate(
        moondream.encode_image(sample['image']),
        prompt,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )[0]

    print(f"Test data {i}")
    print('Moondream:', md_answer)
    print('Ground Truth:', sample['answer'])

    # ans = md_answer.split('<CONCLUSION>')[-1].split('</CONCLUSION>')[0].strip()
    ans = re.findall(r"[A-Z]", md_answer)[-1]
    # print(f"Answer: {ans}")
    if ans == sample['answer']:
        correct += 1
        correct_topic[topic] = correct_topic.get(topic, 0) + 1

end_time = time.time()
t = end_time - start_time
print(f"Execution Time: {t:.4f} seconds. {t / total:.4f} s/example")
print(f"Accuracy: {correct / total}")
for t, num in cnt_topic.items():
    c = correct_topic.get(t, 0)
    print(f"Topic: {t}, Acc: {c / num:.4f}, Correct/Number: {c}/{num}")
