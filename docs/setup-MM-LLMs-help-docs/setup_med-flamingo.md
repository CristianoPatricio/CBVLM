```bash
# 1. Clone repository from github
git clone https://github.com/snap-stanford/med-flamingo
cd med-flamingo

# 2. Create virtual env
python -m venv flam_env

# 3. Install dependencies (it is likely that some of the packages will fail to install)
chmod +x install.sh
./install.sh

# 4. Download Llama-7B
git lfs install
git clone https://huggingface.co/baffo32/decapoda-research-llama-7B-hf

# 5. Edit path to downloaded llama-7b models in scripts/demo.py AND set the path to llama_path under configs/med-flamingo.yaml

# 6. Run demo.py
# In my case, I needed to install again open-flamingo and accelerate:
# pip install open-flamingo==0.0.2
# pip install accelerate
python3 demo.py

######################
# Possible Errors
######################

# RecursionError: maximum recursion depth exceeded
# Solution: modify tokenizer_config.json to:
{
    "bos_token": "<s>",
    "eos_token": "</s>",
    "model_max_length": 1e+30,
    "tokenizer_class": "LlamaTokenizer",
    "unk_token": "<unk>"
}
```