# VT-internship

## Setting up Docker and CUDA tools required to use CUDA inside the Docker container

### NOTE: Not needed if you are running the images on AWS instances as they are already set up correctly

Install Docker:
```shell
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
Add your user to the docker group
```shell
sudo usermod -aG docker $USER
```

Install nvidia-container-toolkit:
```shell
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |\
    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
```

Restart docker:
```shell
sudo systemctl stop docker
sudo systemctl start docker
```

## Building and using Docker images

Build the MMClassification Docker image with:
```shell
docker build -t mmclassification docker/mmclassification
```

Build the MMDetection docker image with:
```shell
docker build -t mmdetection docker/mmdetection
```

TODO: Check what are the prerequisites for the host system (i.e. just the NVIDIA drivers or is CUDA or nvcc also needed).

## Run the MMClassification Docker image
```shell
sudo docker run --gpus all --shm-size=8g -it --volume "$(pwd)":/vt-internship --volume /home/intern/data:/data -w /vt-internship mmclassification
```

## Run the MMDetection Docker image
```shell
sudo docker run --gpus all --shm-size=8g -it --volume "$(pwd)":/vt-internship --volume /home/intern/data:/data -w /vt-internship mmdetection
```

## Verify the MMDetection installation inside the Docker image
```shell
cd /vt-internship
test/verify_mmdetection.sh 
```

## Dataset
https://www.kaggle.com/c/tensorflow-great-barrier-reef/data 
