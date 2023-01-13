
ARG CUDA_DOCKER_VERSION=11.7.1-cudnn8-devel-ubuntu20.04

FROM nvidia/cuda:${CUDA_DOCKER_VERSION}

RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install curl -y
RUN apt-get install wget -y
RUN apt-get install git -y
RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get install libsndfile1-dev -y --no-install-recommends
RUN apt-get install libgl1-mesa-glx -y

RUN git clone https://github.com/kotechniadev/nia2022_66.git /root/workspace

WORKDIR /root/workspace

RUN pip install -r /root/workspace/requirements.txt
RUN pip install -r /root/workspace/pdvc/requirement.txt

# WORKDIR /root/workspace/pdvc/pdvc/ops
# RUN sh make.sh

RUN echo "if [ ! -d '/root/workspace/pdvc/pdvc/ops/dist/' ]; then" >> /root/.bashrc
RUN echo '    echo "======================================="' >> /root/.bashrc
RUN echo '    echo "= You must run the command            ="' >> /root/.bashrc
RUN echo '    echo "= 1. cd /root/workspace/pdvc/pdvc/ops ="' >> /root/.bashrc
RUN echo '    echo "= 2. sh make.sh                       ="' >> /root/.bashrc
RUN echo '    echo "======================================="' >> /root/.bashrc
RUN echo "fi" >> /root/.bashrc
