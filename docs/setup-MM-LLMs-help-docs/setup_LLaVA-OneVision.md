```shell
conda create -n llava-ov python=3.10 -y
conda activate llava-ov
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
cd ..
pip install -r requirements.txt 
```
