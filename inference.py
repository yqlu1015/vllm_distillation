import numpy as np
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
from evaluate import load
from peft import PeftModel
from torch.utils.data import DataLoader

PEFT_INFERENCE = True  # Inference using the fine-tuned model or not

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    # load_in_8bit=True,
)

# Load the model in half-precision
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_name = "llava-hf/llava-v1.6-34b-hf"
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    # low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
)
processor = AutoProcessor.from_pretrained(model_name)
processor.tokenizer.padding_side = "left"
print(f"max_length: {processor.tokenizer.model_max_length}")
if PEFT_INFERENCE:
    lora_adapter = "/project/vsharan_1298/544/vllm_distillation-main_1024/sft-llava-v1.6-mistral-7b_11-18-12-11"  # Path to the fine-tuned adapter
    # lora_adapter = "/project/vsharan_1298/llava_models/sft-llava-v1.6-34b_11-18-18-13"
    model = PeftModel.from_pretrained(model, lora_adapter)

# Load & preprocess dataset
dataset = load_dataset('nlphuji/flickr30k', split='test')
# train = dataset.filter(lambda x: x['split'] == 'train'),
# validation = dataset.filter(lambda x: x['split'] == 'val')
test = dataset.filter(lambda x: x['split'] == 'test')

num_inference = 100
test = test.select(list(range(num_inference)))

# instruction = 'Generate a descriptive caption for the following image. Keep the caption concise and informative.'
instruction = 'Briefly describe this image in one or two sentences.'


def process_func(example):
    # image = example['image']
    # caption = example['caption'][0]

    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': instruction}
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
for i in range(len(test) // batch_size):
    batch = test[i * batch_size: min((i + 1) * batch_size, len(test))]
    inputs = processor(images=batch['image'], text=batch['prompt'], padding=True, return_tensors="pt").to(
        model.device)
    # print({k: v.shape for k, v in inputs.items()})
    generate_ids = model.generate(**inputs, max_new_tokens=256, pad_token_id=processor.tokenizer.eos_token_id)
    outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for idx, output in enumerate(outputs):
        # print(output)
        print(f'\nImage {cnt}, filename: {batch["filename"][idx]}')
        response = output.split('[/INST]')[-1].split('<\s>')[0].strip()
        # response = output.split('assistant')[-1].strip()
        predictions.append(response)
        references.append(batch["caption"][idx])
        print(f'Response:\n{response}')
        print(f'Caption:\n{batch["caption"][idx][0]}')
        cnt += 1

# Evaluate
bleu, rouge, bertscore = load("bleu"), load('rouge'), load("bertscore")
bleu_res = bleu.compute(predictions=predictions, references=references)
rouge_res = rouge.compute(predictions=predictions, references=references)
bertscore_res = bertscore.compute(predictions=predictions, references=references, lang='en')
precision_avg, recall_avg, f1_avg = np.mean(bertscore_res['precision']), np.mean(bertscore_res['recall']), np.mean(
    bertscore_res['f1'])
bert_res = {'average precision': precision_avg, 'average recall': recall_avg, 'average f1': f1_avg}
print(f"BLEU:\n{bleu_res}\nROUGE:\n{rouge_res}")
print(f"BERT-SCORE:\n{bert_res}")
