# Efficient Image Captioning using Small Language Models through Knowledge Distillation
Image captioning using small-sized LLaVA-NeXT with knowledge distillation.

## Installation
1. Clone the repository and navigate to the repository root directory:
    ```angular2html
    git clone https://github.com/yqlu1015/vllm_distillation
    cd vllm_distillation
    ```
2. Set up a conda environment and install all the packages:
   ```angular2html
    conda create -n vllm_env python=3.10 pytorch-cuda=12.1 pytorch cudatoolkit -c pytorch -c nvidia
    conda activate vllm_env
    pip install trl peft accelerate bitsandbytes evaluate pillow
    ```
## Scripts
### Inference
```angular2html
python inference.py
```

### Fine-tuning (SFT)
1. Directly run the script:
   ```angular2html
   python fine-tuing.py
   ```
2. Submit a slurm job (e.g. on CARC server):
   ```angular2html
   sbatch run.sh
   ```