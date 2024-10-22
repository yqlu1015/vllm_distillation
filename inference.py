import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from datasets import load_dataset
import evaluate
from peft import PeftModel
from torch.utils.data import DataLoader

PEFT_INFERENCE = True  # Inference using the fine-tuned model or not

# Load the model in half-precision
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_name)
processor.tokenizer.padding_side = "left"
if PEFT_INFERENCE:
    lora_adapter = "sft-llava-v1.6-mistral-7b_10-22-01-48"  # Path to the fine-tuned adapter
    model = PeftModel.from_pretrained(model, lora_adapter)

# Load & preprocess dataset
dataset = load_dataset('nlphuji/flickr30k', split='test')
# train = dataset.filter(lambda x: x['split'] == 'train'),
# validation = dataset.filter(lambda x: x['split'] == 'val')
test = dataset.filter(lambda x: x['split'] == 'test')

num_inference = 100
test = test.select(list(range(num_inference)))


def process_func(example):
    # image = example['image']
    # caption = example['caption'][0]

    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'Briefly describe this image in one or two sentences.'}
            ]
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # resized_image = image.convert("RGB").resize((672, 672))

    return {'prompt': prompt}


test = test.map(process_func, batched=False)
print(f'Test set size: {len(test)}')

# dataloader = DataLoader(test.with_format("torch"), batch_size=32, shuffle=False)


# Generate
predictions = []
references = []
batch_size = 10
cnt = 0
for i in range(len(test)//batch_size):
    batch = test[i*batch_size: min((i+1)*batch_size, len(test))]
    inputs = processor(images=batch['image'], text=batch['prompt'], padding=True, return_tensors="pt").to(
        model.device)
    print({k: v.shape for k, v in inputs.items()})
    generate_ids = model.generate(**inputs, max_new_tokens=256, pad_token_id=processor.tokenizer.eos_token_id)
    outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for idx, output in enumerate(outputs):
        # print(batch)
        print(f'\nImage {cnt}, filename: {batch["filename"][idx]}')
        response = output.split('[/INST]')[-1].strip()
        predictions.append(response)
        references.append(batch["caption"][idx])
        print(f'Response:\n{response}')
        print(f'Caption:\n{batch["caption"][idx][0]}')
        cnt += 1


# Evaluate
bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print(results)
