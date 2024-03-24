# pull ubuntu 20.04 image with cuda 11.8 installed
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# disable access control
xhost +

# create and run container
# change --gpus, --name, and -v options. 
docker run -dit \
--gpus '"device=1"' \
--name ns-hdp \
--network=host \
--ipc=host \
-e DISPLAY=$DISPLAY \
-e USER=$USER \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/.Xauthority:/root/.Xauthority:rw \
-w /workspace \
-v /$HOME/repos/hdp:/workspace/hdp \
-v /data/sanghyeok/hdp:/workspace/data \
nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04