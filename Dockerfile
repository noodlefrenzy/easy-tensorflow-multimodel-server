FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Taken from https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/docker/Dockerfile.gpu

LABEL maintainer="Michael Lanzetta <milanz@microsoft.com>"

ENV CI_BUILD_PYTHON python3

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        protobuf-compiler \
        python3 \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

COPY . /

RUN pip --no-cache-dir install -r /requirements.txt

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Flask
EXPOSE 5000
CMD ["python", "/app.py"]
