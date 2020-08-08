FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

ENV PYTHON_VERSION 3.7.6
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv
# for skip timezone selecting
ENV DEBIAN_FRONTEND=noninteractive

# install python
RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
 && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && $PYENV_ROOT/plugins/python-build/install.sh \
 && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
 && rm -rf $PYENV_ROOT

RUN pip install --upgrade pip setuptools wheel

# pytorch
RUN pip install torch torchvision
# lightgbm
RUN apt-get install -y cmake
RUN git clone --recursive https://github.com/microsoft/LightGBM /root/LightGBM
RUN mkdir /root/LightGBM/build
WORKDIR /root/LightGBM/build/
RUN cmake ..
RUN make -j4
WORKDIR /root/LightGBM/python-package
RUN pip install -e .
WORKDIR /
# requirements
COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
