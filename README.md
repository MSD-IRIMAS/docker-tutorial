# Guide to Utilizing Docker for GPU-Accelerated Deep Learning Experiments

Docker is a powerful platform that allows developers to package applications and their dependencies into containers, ensuring consistent performance across various environments. These containers are lightweight, portable, and isolated, making it easier to deploy and manage applications without worrying about environment discrepancies. When combined with GPU acceleration, Docker becomes even more effective, particularly for deep learning experiments, which often require significant computational power. By leveraging NVIDIA's Docker toolkit, developers can enable GPU support within containers, allowing deep learning frameworks like TensorFlow or PyTorch to utilize the full potential of the system’s GPUs. This setup enhances computational performance and accelerates model training and experimentation in a scalable and reproducible way.

In this tutorial, you will learn how to install docker, use docker for GPU acceleration, create dockerfiles to setup your own image, create containers for your docker images, and run your experiments from inside the docker container.

To develop with Docker, several key components are required:

1. **Docker Image**: This is a lightweight, stand-alone, executable package that contains everything needed to run a piece of software, including code, runtime, libraries, environment variables, and dependencies. It is built from a Dockerfile and serves as the blueprint for containers.

2. **Dockerfile**: A text file containing a set of instructions to build a Docker image. It specifies the base image, software dependencies, environment settings, and the command to execute when the container starts.

3. **Docker Container**: A running instance of a Docker image. Containers are isolated environments where applications are executed. They share the host system’s OS kernel but maintain their own filesystem, networking, and resources, making them highly efficient for deployment.

4. **Docker Daemon**: The background service running on the host machine that manages Docker containers, images, and other related components.

## Part 1: Docker installation

```NB: If you already have docker setup and installed with no sudo obligation you can skip to Part 2. Docker with GPU```

This part and the next one are based on the [following tutorial](https://gist.github.com/qin-yu/d3619a68d209dd1feefd7385e43c3fc4) made by [Qin Yu](https://gist.github.com/qin-yu).

### Set up the repository

1. To allow apt to use a repository over HTTPS: 
```bash
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```

2. Add Docker’s official GPG key:
```bash
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
```

3. Set up the stable repository: 
```bash
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

### Install Docker Engine - Community

1. Install the latest version of Docker Engine - Community and containerd: 
```bash
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

2. Verify that Docker Engine - Community is installed correctly: 
```bash
$ sudo docker run hello-world
```

Let's first break down the above command:

- `sudo docker`: using docker cli
- `run <image_name>`: creates and runs a container using the setup of the docker image `<image_name>`, in this case the `hello-world` image.
Such image and many others, exists on the [docker hub](https://hub.docker.com/).

If you get an output starting with the following:
```bash
Hello from Docker!
This message shows that your installation appears to be working correctly.
```
It means that docker has been successfully installed on your machine.

### Avoid using `sudo docker` and use `docker` instead

The `docker` daemon connects to a Unix socket rather than a TCP port. By default, this socket is owned by the `root` user, and other users must use `sudo` to access it. Since the `docker` daemon runs as the `root` user, to avoid prefixing every `docker` command with `sudo`, you can create a Unix group named `docker` and add users to this group. Once the `docker` daemon starts, it grants access to the Unix socket to all members of the `docker` group, allowing them to use `docker` without `root` privileges.

1. Create the docker group:
```bash
$ sudo groupadd docker
```

2. Add your user to the docker group:
```bash
$ sudo usermod -aG docker $USER
```

3. Log out and log back in so that your group membership is re-evaluated. On Linux, you can also run the following command to activate the changes to groups:
```bash
$ newgrp docker
```

4. Verify that you can run docker commands without sudo:
```bash
$ docker run hello-world
```

You should get the same as before, if yes then you now can use `docker` with no need of `sudo`.

## Part 2: Docker with GPU and Nvidia

For this part, you need to make sure Nvidia driver, latest, tested and proprietary (Do not download any version of Nvidia driver especially on Ubuntu you may crash the GUI and have to purge Nvidia from the kernel)

### Make sure Nvidia is installed and ready

This can be easily done using the following command <br>
```bash
$ nvidia-smi
```
The expected output should look something like that:
```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI Driver Version: <driver_version>  CUDA Version: <cuda_version>   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  <gpu_name>       Off   | <bus_id>       Off   |       <ecc_status>     |
|  30%   40C    P8     15W / 300W |    500MiB / 16280MiB |      0%    Default |
|                               |                      |                  Off |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     <pid>    C   <process_name>           <memory_usage>MiB |
+-----------------------------------------------------------------------------+
```

### Install NVIDIA Container Toolkit

```bash
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo apt-get update
$ sudo apt-get install nvidia-container-toolkit
```

And then reboot your system with:
```bash
$ sudo reboot
```

### Pull the nivdia/cuda docker image

In order to use GPU acceleration with docker, you would need to pull the associated nvidia/cuda docker image. Each docker image comes with many possible tags, specifying which version to use. In this case, the tag should specify 2 information, which cuda version to use and which operating system to use, for instance, when running `nvidia-smi` on your system and findin that the `<cuda version>` is `12.4` and you want an Ubuntu 22.04 as backend, you would need to choose the tag `12.4.1-base-ubuntu22.04`. In order to pull that image and test if docker is really detecting and associating your GPU hardware to its framework, you would need to run the following:
```bash
docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The `--gpus all` options tells docker to use all available GPU hardware available on your computer. The image name used is `nvidia/cuda` tagged by `12.4.1-base-ubuntu22.04`, and the command to run is `nvidia-smi`. The output of this command should be the same as running `nvidia-smi` on your computer with no docker.

## Part 3: Dockerfiles and how to use them

A Dockerfile is a simple, text based file with a set of instructions on how to build a Docker Image. Each line in a Dockerfile represents a step in the image building process, such as specifying the base image, installing dependencies, copying files, setting environment variables, and configuring commands to run. Dockerfiles make it easy to reproduce an environment by building images consistently across different machines. When a Dockerfile is executed, Docker reads each line and creates layers (stored in cache), making the image building process faster by reusing layers when possible.

### Writing your dockerfile

In order to create a Dockerfile to setup your image, simply create a file that is called `dockerfile` in your project directory. The docker file contains much information, but it comes down mostly to:

1. Choosing the image your project is based on, i.e. `tensorflow.`
2. Setting the user and group IDs, creating a non-root user to access the container through.
3. Updating internal system.
4. Installing dependencies.
5. Choosing working directory.

An example dockerfile can be found [here](dockerfile)

The first line in that dockerfile is the image we're based on:
```docker
FROM tensorflow/tensorflow:2.16,1-gpu
```
Here we are using tensorflow image, version `2.16.1` with GPU configuration. Keep in mind that the version of tensorflow you use in your dockerfile should be well alligned with the CUDA version you pulled in the previous section.

The second two lines defines 2 arguments that will be passed later, the user ID and the group ID, which when passed later should be set to the same as your user on the host machine:
```docker
ARG USER_ID
ARG GROUP_ID
```

The following line creates a user with an ID and group ID called `myuser` and creates its own home directory:
```docker
RUN groupadd -r -g $GROUP_ID myuser && useradd -r -u $USER_ID -g myuser -m -d /home/myuser myuser
```

The following line set the shell used (required on most machines):
```docker
ENV SHELL /bin/bash
```

The following line creates a directory called `code` in the `myuser` home folder that we will later on put our code in, while giving full read and write access to `myuser`:
```docker
RUN mkdir -p /home/myuser/code && chown -R myuser:myuser /home/myuser/code
```

The following line sets the working directory:
```docker
WORKDIR /home/myuser/code
```

The rest of this file depends on the application of your project, containing system updates, pip updates and dependencies installation:
```docker
RUN apt update
RUN pip install --upgrade pip
RUN pip install numpy==1.23.5
RUN pip install pandas==2.0.3
```

### Building your image using your dockerfile

In order to build your image using the configuration detailed in the [dockerfile](dockerfile), you would need to run the following command:
```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t <my-docker-image-name>:<my-tag> .
```
Let's break down the above command:
- `docker build` is a command to build a docker image.
- `--build arg USER_ID=$(id -u)` is to specify an argument for the building step, such as the `USER_ID` which is set to your user ID on your host machine using `$(id -u)`.
- `--build arg GROUP_ID=$(id -g)`, same as previous point but for the group ID.
- `-t <my-docker-image-name>:<my-tag>` is to choose a name for your docker image with a specific tag, for example `-t my-app:latest`, however the tag is not necessary, if you do not plan to do many versions of your image you can simply use `-t my-app`.
- `.` is very important, it means look for the [dockerfile](dockerfile) in the current directory where you are running this command.

If no error are shown, then your docker image has successfully been built. To make sure your image is built you can run the following:
```bash
docker images
```
which lists all the images in your system, one of them should the one you just built.

### Creating your first docker container on top of your built image

After having built your docker image, you can now create one instance, one docker container, using that image configuration, by running the following command:
```bash
docker run --gpus all -it --name <my-docker-container-name> -v "$(pwd):/home/myuser/code" --user $(id -u):$(id -g) <my-docker-image-name>:<my-tag> <my-command>
```

Let's break the above command:

- `docker run --gpus all`, same as explained before.
- `-it` means to run the docker container in interactive mode so that we can open the container and see what is happening inside.
- `--name <my-docker-container-name>` is to choose a name for our container.
- `-v "$(pwd):/home/myuser/code"` creates a volume, which is a mounted folder between the host machine and the docker container itself. For instance here we are mounting the source code directory of our project, assuming it is the current working directory where we are running these commands (PS: `pwd` stands for print working directory), to the directory we already created in our [dockerfile](dockerfile) `/home/myuser/code`. You can create as many volumes as you want in this command, the more you need just add a new `-v` option with `/directory/on/host/machine:/directory/inside/docker/container` as input.
- `--user $(id -u):$(id -g)` is to choose to run and execute your docker container with your user and group ID on the host machine. After the firsr run, everytime you will open your container it will be by default your user id and group id so you won't have to specify it everytime you execute a command on your container. You always have the choice to execute your container as root by specifying user and group id set to 0.
- `<my-docker-image-name>:<my-tag>` should be replaced by the image your container is based on, the one you created before. Specifying a tag is not necessary, just if you already have one.
- `<my-command>` is the command to be run once the container is created, usually set to `bash` to open a shell and see if setup of the container is correct.


This will open a shell in the directory `/home/myuser/code`, if you use `ls` you should be able to see your code in that directory. Keep in mind, every change you make in your code on your host machine will now be change automatically in the container in `/hone/myuser/code` because it is mounted with a volume.
To make sure your container now can use the GPU acceleration with tensorflow, simply open a python shell inside your container terminal and type in the following:
```python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices('GPU')
```
This should output a list of GPU devices detected, which should be yours.

Now you can simply run your code inside your docker container with GPU acceleration, have fun.

## Helpful docker commands

- Stopping a docker container:
```bash
docker stop <container-name>
```
- Deleting a docker container (keep in mind a container should be stopped in order to delete it):
```bash
docker rm <container-name>
```
- Restarting a docker container:
```bash
docker restart <container-name>
```
- Starting a docker container:
```bash
docker start <container-name>
```
- Listing running (non stopped) docker containers:
```bash
docker ps
```
- Listing all docker containers (running and not running):
```bash
docker ps -a
```
- Listing all docker images:
```bash
docker images
```
- Deleting an image (keep in mind you cannot delete an image that has at least one container associated to it):
```bash
docker image rm <image-name>:<tag>
```
Or without tag if you do not use tags.
- If you ever make changes inside a docker container and want to create a new image from this container you can run (first fetch your container ID using `docker ps`):
```bash
docker commit <container-id> <new-image-name>:<tag>
```
And then you can create a new docker container using this new image:
```bash
docker run -it <new-image-name>:<tag>
```
Or without tag.
Keep in mind that any changes done inside a docker container are saved inside that container but not the image, you YES you can override the image that created this container by simply committing to its name and overriding it.

## Running a Jupyter Lab server with GPU support.

dockerfile:
```docker
FROM tensorflow/tensorflow:2.16.1-gpu

ARG USER_ID
ARG GROUP_ID

RUN groupadd -r -g $GROUP_ID myuser && useradd -r -u $USER_ID -g myuser -m -d /home/myuser myuser
ENV SHELL=/bin/bash

RUN mkdir -p /home/myuser/code && chown -R myuser:myuser /home/myuser/code

WORKDIR /home/myuser/code

RUN apt update
RUN pip install --upgrade pip
RUN pip install jupyterlab==4.2.6

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
```

Build the image:
```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t <my-docker-image-name>:<my-tag> .
```

Run server first time:
```bash    
docker run --gpus all -it --name <my-docker-container-name> -v "$(pwd):/home/myuser/code" --user $(id -u):$(id -g) -p 8888:8888 <my-docker-image-name>:<my-tag>
```

Detach
* CTRL+P
* CTRL+Q

Attach
```bash
docker attach <my-docker-container-name>
```

Restart server after reboot:
```bash
docker ps -a
# Find <my-docker-image-name> <CONTAINER ID>
docker restart <CONTAINER ID>
```
