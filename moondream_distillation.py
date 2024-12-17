import os
import re
import time

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bitsandbytes.optim import Adam8bit
import math
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset


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


# Define dataset
datasets = {
    "train": VQADataset("train"),
    "test": VQADataset("test"),
}

# Load model
DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16  # CPU doesn't support float16
MD_REVISION = "2024-07-23"

tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
moondream_r = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)
moondream_r.text_model.train()
moondream_r.text_model.transformer.gradient_checkpointing_enable()

# Training Config
EPOCHS = 1

# Number of samples to process in each batch. Set this to the highest value that doesn't cause an
# out-of-memory error. Decrease it if you're running out of memory.
BATCH_SIZE = 8

# Number of batches to process before updating the model. You can use this to simulate a higher batch
# size than your GPU can handle. Set this to 1 to disable gradient accumulation.
GRAD_ACCUM_STEPS = 1

# Learning rate for the Adam optimizer. Needs to be tuned on a case-by-case basis. As a general rule
# of thumb, increase it by 1.4 times each time you double the effective batch size.
#
# Source: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
#
# Note that we linearly warm the learning rate up from 0.1 * LR to LR over the first 10% of the
# training run, and then decay it back to 0.1 * LR over the last 90% of the training run using a
# cosine schedule.
LR = 1e-5
print(f"BATCH_SIZE: {BATCH_SIZE}, LR: {LR}")

# Whether to use Weights and Biases for logging training metrics.
USE_WANDB = True

NUM_TRAIN = 2000
ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729

label = """{r}

<CONCLUSION> {a} </CONCLUSION>"""

subset = Subset(datasets['train'], indices=list(range(NUM_TRAIN)))


def collate_fn_rationale(batch):
    images = [sample['image'] for sample in batch]
    images = [moondream_r.vision_encoder.preprocess(image) for image in images]

    labels_acc = []
    tokens_acc = []

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)

        q_t = tokenizer(
            f"\n\nQuestion: {sample['question']}\n\nAnswer:",
            add_special_tokens=False
        ).input_ids
        toks.extend(q_t)
        labs.extend([-100] * len(q_t))

        a_t = tokenizer(
            label.format(r=sample['rationale'], a=sample['answer']) + ANSWER_EOS,
            add_special_tokens=False
        ).input_ids
        toks.extend(a_t)
        labs.extend(a_t)

        tokens_acc.append(toks)
        labels_acc.append(labs)

    max_len = -1
    for labels in labels_acc:
        max_len = max(max_len, len(labels))

    attn_mask_acc = []

    for i in range(len(batch)):
        len_i = len(labels_acc[i])
        pad_i = max_len - len_i

        labels_acc[i].extend([-100] * pad_i)
        tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)

    return (
        images,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )


def compute_loss_r(batch):
    images, tokens, labels, attn_mask = batch

    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    with torch.no_grad():
        img_embs = moondream_r.vision_encoder(images)

    tok_embs = moondream_r.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = moondream_r.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


# Training
dataloaders_rationale = {
    "train": DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_rationale,
    )
}

total_steps = EPOCHS * len(dataloaders_rationale["train"]) // GRAD_ACCUM_STEPS
print(f"Total steps: {total_steps}")
optimizer = Adam8bit(
    [
        {"params": moondream_r.text_model.parameters()},
    ],
    lr=LR * 0.1,
    betas=(0.9, 0.95),
    eps=1e-6
)

if USE_WANDB:
    import wandb

    wandb.init(
        project="moondream-ft",
        config={
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        },
        name="distillation-2000-2"
    )

i = 0
for epoch in range(EPOCHS):
    for batch in tqdm(dataloaders_rationale["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        i += 1

        loss = compute_loss_r(batch)
        loss.backward()

        if i % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if USE_WANDB:
            wandb.log({
                "loss/train": loss.item(),
                "lr": optimizer.param_groups[0]['lr']
            })

if USE_WANDB:
    wandb.finish()

moondream_r.save_pretrained("/project/vsharan_1298/llava_models/moondream-distillation-2000-2")

# Evaluation
moondream_r.eval()

total, correct = 0, 0
cnt_topic, correct_topic = {}, {}

start_time = time.time()
for i in range(len(datasets['test'])):
    total += 1
    sample = datasets['test'][i]
    question = sample['question']
    topic = sample['topic']
    cnt_topic[topic] = cnt_topic.get(topic, 0) + 1
    prompt = f"<image>\n\nQuestion: {question}\n\nAnswer:"
    md_answer = moondream_r.generate(
        moondream_r.encode_image(sample['image']),
        prompt,
        tokenizer=tokenizer,
        max_new_tokens=512
    )[0]

    print(f"Test data {i}")
    print('Moondream:', md_answer)
    print('Ground Truth:', sample['answer'])

    ans = md_answer.split('<CONCLUSION>')[-1].split('</CONCLUSION>')[0].strip()
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
