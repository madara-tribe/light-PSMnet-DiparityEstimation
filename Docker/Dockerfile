FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ENV CUDA_VERSION 10.1
ENV PYTHON_VERSION 3.7.0
WORKDIR /home/app
RUN apt-get update && apt-get upgrade -y
RUN apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential libbz2-dev tk-dev
RUN apt-get install tree

# Media I/O:
RUN sudo apt install -y \
    zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libopenexr-dev libgdal-dev libgtk2.0-dev

# Video I/O:
RUN sudo apt install -y \
    libdc1394-22-dev libavcodec-dev libavformat-dev \
    libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
    libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev \
    libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN sudo apt install -y libtbb-dev libeigen3-dev



RUN apt-get update && apt-get upgrade -y && \
    apt install -y  --no-install-recommends make cmake gcc git g++ unzip wget build-essential zlib1g-dev libffi-dev libssl-dev && \
    apt clean && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar zxf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure && \
    make altinstall
RUN apt-get install -y tree curl


RUN ln -s /usr/local/bin/python3.7 /bin/python3 && \
    ln -s /usr/local/bin/pip3.7 /bin/pip3
# install pip
RUN pip3 install --upgrade pip
# https://pytorch.org/get-started/previous-versions/
RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt
CMD [ "/bin/bash" ]
