FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN apt update && apt install -y bash \
                   git \
                   curl

RUN apt install -y python3.9 \
                   python3-pip

RUN python3.9 -m pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    fvcore \
    iopath \
    pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html

RUN python3.9 -m pip install accelerate \
    einops \
    ema-pytorch \
    numpy \
    tqdm \
    wandb==0.14.2 \
    opencv-python \
    h5py \
    matplotlib \
    scikit-image \
    scipy \
    lpips \
    omegaconf \
    torch-efficient-distloss \
    torchtyping \
    typeguard==2.13.3 \
    hydra-core \
    jaxtyping \
    scikit-learn \
    timm==0.5.4 \
    imageio==2.27.0 \
    imageio-ffmpeg \
    pytube


# conda create -n dfm python=3.9 -y
# conda activate dfm
# pip install torch==2.0.1 torchvision
# conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
# pip install -r requirements.txt
# python setup.py develop