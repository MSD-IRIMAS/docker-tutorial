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