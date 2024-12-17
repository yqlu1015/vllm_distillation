# Distilling Small Language Models with Structured Reasoning
Visual Question Answering (VQA) done by [Moondream 2B](https://github.com/vikhyat/moondream) with structured reasoning distillation from [LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT).
Code for fine-tuning [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) on image captioning is also included.

## Installation
1. Clone the repository and navigate to the repository root directory:
    ```angular2html
    git clone https://github.com/yqlu1015/vllm_distillation
    cd vllm_distillation
    ```
2. Set up a conda environment and install all the packages:
    ```angular2html
    conda create -n vllm_env python=3.10 pytorch-cuda=12.1 pytorch torchvision cudatoolkit -c pytorch -c nvidia
    conda activate vllm_env
    pip install -r requirements.txt
    ```
## Scripts
### Inference
```angular2html
python moondream_inference.py
```

### Fine-tuning (SFT)
1. Directly run the script:
   ```angular2html
   python moondream_ft.py
   ```
2. Submit a slurm job:
   ```angular2html
   sbatch run.sh
   ```

### Structured Visual Reasoning Distillation
```angular2html
   python moondream_distillation.py
   ```
