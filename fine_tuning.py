import json
import os
from datetime import datetime

import torch
import wandb
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

LORA_RANK = 32
NUM_TRAIN = 800
# NUM_VAL = 100
NUM_EPOCHS = 1
LR = 1e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 256
output_model = datetime.now().strftime(f"sft-llava-v1.6-mistral-7b_%m-%d-%H-%M")

# Use wandb for logging
wandb.init(
    project='vllm-distillation',
    job_type="training",
    anonymous="allow",
    name=output_model,
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
    torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
processor.tokenizer.padding_side = "right"
model.padding_side = 'right'


# Prepare model for PEFT (QLoRA)
def find_all_linear_names(target_model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in target_model.named_modules():
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
# validation = dataset.filter(lambda x: x['split'] == 'val').select(list(range(NUM_VAL)))
# test = dataset.filter(lambda x: x['split'] == 'test')
print(f'Training set size before augmentation: {len(train)}')


# Data augmentation (may not applicable)
def augment_data(examples):
    augmented_captions = []
    augmented_images = []
    for image, captions in zip(examples["image"], examples["caption"]):
        # for caption in captions:
        augmented_captions.append(captions[0])
        augmented_images.append(image)
    return {"augmented_image": augmented_images, "augmented_caption": augmented_captions}


train = train.map(augment_data, batched=True, remove_columns=train.column_names)
print(f'Training set size after augmentation: {len(train)}')
# validation = validation.map(augment_data, batched=True, remove_columns=validation.column_names)


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
                    {"type": "text", "text": example['augmented_caption']},
                ]
            }
        ]
        for example in examples
    ]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    images = [example['augmented_image'] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LENGTH)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    # # Ignore the image token index in the loss computation (model specific)
    # image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    # labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


# Supervised fine-tuning
training_args = SFTConfig(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=10,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_grad_norm=1.0,
    logging_steps=10,
    # save_strategy="epoch",
    # evaluation_strategy="steps",
    optim="adamw_torch",  # adamw_torch
    weight_decay=0.01,
    lr_scheduler_type="linear",
    output_dir=output_model,
    overwrite_output_dir=True,
    # dataset_text_field="",
    # max_seq_length=2048,
    report_to="wandb",
)
# training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
training_args.dataset_text_field = ''
training_args.remove_unused_columns = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train,
    # eval_dataset=validation,
    tokenizer=processor.tokenizer,
)
trainer.train()

wandb.finish()

# Save model & hyperparameters
trainer.save_model(training_args.output_dir)
hyperparameters = {
    "num_epochs": NUM_EPOCHS,
    "lora_rank": LORA_RANK,
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_train": NUM_TRAIN,
    # "num_val": NUM_VAL,
    "max_length": MAX_LENGTH,
}
json.dump(hyperparameters, open(os.path.join(training_args.output_dir, 'hyperparameters.json'), 'w'))
