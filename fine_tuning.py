import json
import os
from datetime import datetime

import torch
import wandb
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

LORA_RANK = 64
NUM_TRAIN = 10000
NUM_VAL = 100
LR = 5e-5

# Use wandb for logging
wandb.init(
    project='vllm-distillation',
    job_type="training",
    anonymous="allow",
    name="llava-7b-sft",
)

# Specify quantization configuration for the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model in half-precision
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16, quantization_config=quantization_config, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
processor.tokenizer.padding_side = "right"
model.padding_side = 'right'


# Prepare model for PEFT (QLoRA)
def find_all_linear_names(original_model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['vision_model']  # 'multi_modal_projector',
    for name, module in original_model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load & preprocess dataset
dataset = load_dataset('nlphuji/flickr30k', split='test')
train = dataset.filter(lambda x: x['split'] == 'train')
if NUM_TRAIN != -1:
    train = train.select(list(range(NUM_TRAIN)))
validation = dataset.filter(lambda x: x['split'] == 'val').select(list(range(NUM_VAL)))
# test = dataset.filter(lambda x: x['split'] == 'test')
print(f'Training set size: {len(train)}, validation set size: {len(validation)}')


# Define data collector
def collate_fn(examples):
    # Get the texts and images
    conversations = [
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': 'Briefly describe this image in one or two sentences.'}
                ]
            },
            {
                'role': 'assistant',
                'content': [
                    {"type": "text", "text": example['caption'][0]},
                ]
            }
        ]
        for example in examples
    ]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    images = [example['image'] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(images=images, text=texts, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    # Ignore the image token index in the loss computation (model specific)
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


# Supervised fine-tuning
training_args = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=1,
    learning_rate=LR,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="steps",
    optim="adamw_8bit",  # adamw_torch
    weight_decay=0.01,
    lr_scheduler_type="linear",
    output_dir=datetime.now().strftime(f"sft-llava-v1.6-mistral-7b_%m-%d-%H-%M"),
    overwrite_output_dir=True,
    dataset_text_field="",
    # max_seq_length=2048,
    report_to="wandb",
)
training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
training_args.remove_unused_columns = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train,
    eval_dataset=validation,
    tokenizer=processor.tokenizer,
)
trainer.train()

wandb.finish()

# Save
trainer.save_model(training_args.output_dir)
hyperparameters = {
    "lora_rank": LORA_RANK,
    "learning_rate": LR,
    "num_train": NUM_TRAIN,
    "num_val": NUM_VAL,
}
json.dump(hyperparameters, open(os.path.join(training_args.output_dir, 'hyperparameters.json'), 'w'))
