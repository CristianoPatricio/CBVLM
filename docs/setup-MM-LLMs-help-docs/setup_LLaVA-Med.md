```bash
# 1. Clone repo
git clone https://github.com/microsoft/LLaVA-Med.git
cd LLaVA-Med

# 2. Install packages
conda create -n llava-med python=3.10 -y
conda activate llava-med
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

######################
# Possible Errors
######################

# ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
# Solution: upgrade scikit-learn package
pip install -U scikit-learn
```