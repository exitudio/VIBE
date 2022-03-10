#!/usr/bin/env bash

export CONDA_ENV_NAME=vibe-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

# pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
# pip3 install -r requirements.txt


# install np first for solving conflict
~/miniconda3/envs/vibe-env/bin/pip3 install numpy==1.17.5
~/miniconda3/envs/vibe-env/bin/pip3 install -r requirements.txt
~/miniconda3/envs/vibe-env/bin/pip3 install git+https://github.com/giacaglia/pytube.git --upgrade ml_collections
conda install tqdm ipykernel

conda install pytorch=1.10.0 torchvision cudatoolkit=10.2 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
# conda install pytorch3d -c pytorch3d

##### install version by python,cuda,pytorch for pytorch3d
 ~/miniconda3/envs/vibe-env/bin/pip3 install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu102_pyt1100/download.html



################################################################
######### Without Pytorch3d ####################
# there is ffmpeg error when render video if install with pytorch3d

# ~/miniconda3/envs/vibe-env/bin/pip3 install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
# ~/miniconda3/envs/vibe-env/bin/pip3 install -r requirements.txt
# ~/miniconda3/envs/vibe-env/bin/pip3 install git+https://github.com/giacaglia/pytube.git --upgrade ml_collections
# conda install tqdm ipykernel