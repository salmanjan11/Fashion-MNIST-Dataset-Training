ARG UBUNTU_VERSION=20.04
FROM ubuntu:$UBUNTU_VERSION

RUN apt update --fix-missing && \
    apt install -y --no-install-recommends ca-certificates git sudo curl libgl1-mesa-glx python3-pip libglib2.0-0 && \
    apt clean

RUN mkdir /app

COPY requirements.txt /app

RUN pip3 install --upgrade pip
RUN pip3 install torch
RUN pip3 install -r /app/requirements.txt

WORKDIR /app

CMD [ "python3", "./train.py"]