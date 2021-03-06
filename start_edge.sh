#!bin/bash
sudo chown -R $USER $PWD

# Pull git repo and create all the subdirectories that are required
cd store

# If tiny dataset exists then do not download unzip etc
if [ ! -d kinetics400_tiny ]
then
	wget https://download.openmmlab.com/mmaction/kinetics400_tiny.zip
	unzip kinetics400_tiny.zip > /dev/null
else
	echo Kinetics tiny datset exists, not downloading
fi

# Now clone the video swin repo
if [ ! -d Video-Swin-Transformer ]
then
    echo Cloning Video-Swin-Transformer repo
    git clone https://github.com/SwinTransformer/Video-Swin-Transformer.git
else
    echo Video-Swin-Transformer already exists not cloning
fi

# Check to see if the model file exists
cd configs
if [ ! -e swin_tiny_patch244_window877_kinetics400_1k.pth ]
then
	wget "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth"
fi
cd ..

# Move out of the store into the base
cd ..

# If the image does not exist, then build it
docker image inspect athiruve/swin:bsl_edge > /dev/null || docker build --no-cache -t athiruve/swin:bsl_edge -f docker/DockerFile.edge_shell .
docker run -it --runtime=nvidia --device=/dev/video0:/dev/video0 --privileged=true --gpus all --rm --network=host --ipc=host -v $PWD/store/:/workspace  -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /tmp/argus_socket:/tmp/argus_socket --cap-add SYS_PTRACE -e DISPLAY=:0 atox120/swin:latest
