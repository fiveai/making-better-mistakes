FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Make things work with Anaconda
ARG PYTHON_VERSION=3.6
ARG conda_version=Miniconda3-4.6.14-Linux-x86_64

ENV PYTHON_VERSION=${python}
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install some default packages
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         wget \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# Install Anaconda and python packages
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/${conda_version}.sh && \
    /bin/bash /${conda_version}.sh -f -b -p $CONDA_DIR && \
    rm ${conda_version}.sh

# Create conda env
COPY environment.yml .
RUN conda env update -n base -f environment.yml && \
    rm environment.yml
