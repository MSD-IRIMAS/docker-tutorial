FROM tensorflow/tensorflow:2.16.1-gpu

ARG USER_ID
ARG GROUP_ID

RUN groupadd -r -g $GROUP_ID myuser && useradd -r -u $USER_ID -g myuser -m -d /home/myuser myuser
ENV SHELL /bin/bash

RUN mkdir -p /home/myuser/code && chown -R myuser:myuser /home/myuser/code

WORKDIR /home/myuser/code

RUN apt update
RUN pip install --upgrade pip
RUN pip install numpy==1.23.5
RUN pip install pandas==2.0.3