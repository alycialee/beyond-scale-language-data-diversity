#!/bin/bash
# -- Check envs
conda info -e

# -- Activate conda
conda update -n base -c defaults conda
pip install --upgrade pip
conda create -y -n beyond_scale python=3.10.11
conda activate beyond_scale
#conda remove --name data_quality --all

# -- Install this library from source
# - Get the code, put it in afs so its available to all machines and symlink it to home in the local machine
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/beyond-scale-language-data-diversity.git
ln -s /afs/cs.stanford.edu/u/brando9/beyond-scale-language-data-diversity $HOME/beyond-scale-language-data-diversity
# - Install the library in editable mode so that changes are reflected immediately in running code
pip install -e ~/beyond-scale-language-data-diversity
# pip uninstall ~/beyond-scale-language-data-diversity
cd ~/beyond-scale-language-data-diversity

# -- Test pytorch
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"
python -c "import torch; print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"
python -c "import torch; print(f'{torch.cuda.device_count()=}'); print(f'Device: {torch.cuda.get_device_name(0)=}')"

# -- Install uutils from source
# - Get the code, put it in afs so its available to all machines and symlink it to home in the local machine
cd /afs/cs.stanford.edu/u/brando9/
ln -s /afs/cs.stanford.edu/u/brando9/ultimate-utils $HOME/ultimate-utils
git clone git@github.com:brando90/ultimate-utils.git $HOME/ultimate-utils/
# - Install the library in editable mode so that changes are reflected immediately in running code
pip install -e ~/ultimate-utils
#pip uninstall ~/ultimate-utils

# - Test uutils
python -c "import uutils; uutils.torch_uu.gpu_test()"

# -- Wandb
pip install wandb --upgrade
wandb login
#wandb login --relogin
cat ~/.netrc
