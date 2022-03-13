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
docker image inspect swin:bsl > /dev/null || docker build --no-cache -t swin:bsl -f docker/DockerFile.server .

# Start the docker and capture the stdout
oldout=$(docker run -d --gpus all --rm --net=host --ipc=host -v $PWD/store/:/workspace swin:bsl 2>&1 | xargs)
sleep 2
newout=$(docker logs "$oldout" 2>&1 | grep "    http://hostname:8888")

# Print the command to display the jupyter notebook address
ipadd=$(wget -qO- https://ipecho.net/plain | grep "")
echo "######"
echo "Use this URL in your browser:"
echo "${newout/hostname/"$ipadd"}"
