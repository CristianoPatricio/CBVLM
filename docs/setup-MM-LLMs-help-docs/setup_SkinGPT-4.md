```shell
git clone https://github.com/JoshuaChou2018/SkinGPT-4.git
conda env create -f environment.yml
conda activate skingpt4_llama2
conda install -c conda-forge mamba=1.4.7
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf 

# Modify line 16 at SkinGPT-4-llama2/skingpt4/configs/models/skingpt4_llama2_13bchat.yaml to be the path of Llama-2-13b-chat-hf.
# Modify line 8 of skingpt4_eval_llama2_13bchat.yaml to specify the prompts path
# Modify line 19 at multimodal-LLM-explainability-dev/src/models/skingpt4/demo.py to be the path of SkinGPT-4 folder
```