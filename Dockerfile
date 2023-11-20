FROM ubuntu:22.04

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,video
# RUN apt update && apt install -y nvidia-driver-530
RUN apt update && apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.10
RUN apt update \
    && apt install -y tzdata \
        # libavfilter-dev \
        #   libavformat-dev \
        #   libavcodec-dev \
        #   libswresample-dev \
        #   libavutil-dev\
          wget \
          build-essential \
          git \
          yasm \
          ninja-build \
        #   cmake \
          git \
          python3-pip \
          python-is-python3 \
          python3-opencv iputils-ping dnsutils vim \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    # && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && echo ${TZ} > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata
RUN python3 -m pip install --no-cache-dir setuptools wheel

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
COPY ./cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb ./
RUN dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update && apt-get -y install cuda

WORKDIR /sh_cv_robot
COPY ./requirements.txt /sh_cv_robot/
RUN python3 -m pip install --no-cache-dir --upgrade -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
COPY ./allspark-0.15-py2.py3-none-any.whl /sh_cv_robot/allspark-0.15-py2.py3-none-any.whl
RUN  python3 -m pip install allspark-0.15-py2.py3-none-any.whl
RUN python3 -m pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple/

RUN apt-get update && apt-get install -y supervisor
RUN mkdir -p /var/log/supervisor && mkdir -p /etc/supervisor
COPY ./supervisor_conf/supervisor_test /etc/supervisor


COPY ./models /sh_cv_robot/models
COPY ./fonts /sh_cv_robot/fonts
COPY ./resnet /sh_cv_robot/resnet
COPY ./*.cfg /sh_cv_robot/
COPY ./*.py /sh_cv_robot/

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
