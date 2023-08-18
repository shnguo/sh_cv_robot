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
          python3-opencv \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    # && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && echo ${TZ} > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata
RUN python3 -m pip install --no-cache-dir setuptools wheel

WORKDIR /sh_cv_robot
COPY ./requirements.txt /sh_cv_robot/
RUN python3 -m pip install --no-cache-dir --upgrade -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
CMD ["python", "main_app.py"]
