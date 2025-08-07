```shell
conda create -n vila python=3.10 -y
conda activate vila

pip install --upgrade pip  # enable PEP 660 support

# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install git+https://github.com/Efficient-Large-Model/VILA.git@7b65b0be2db0359861d4f151fa1d35de206c6cc4

pip install git+https://github.com/huggingface/transformers@v4.37.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')

git clone https://github.com/Efficient-Large-Model/VILA.git
cd VILA
git checkout 7b65b0be2db0359861d4f151fa1d35de206c6cc4
cd ..

cp -rv VILA/llava/train/transformers_replace/* $site_pkg_path/transformers/
cp -rv VILA/llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

rm -rf VILA

pip install deepspeed

cd CBVLM
pip install -r requirements.txt
```
