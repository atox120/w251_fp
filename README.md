# W251 Final Project: Video action detection for baby sign language  
# Alex To, Aswin Thiruvengadam, Dan Ortiz, Jeffrey Laughman

## 1. Introduction
Baby sign language is comprised of a series of symbols (closely related to ASL) to improve communication between babies and caregivers prior to vocal vocabulary develops. The goal is to reduce crying and tantrums by providing a medium of communication and reducing caregiver guessing.

## 2. Training on the cloud

### 2.1 Sourcing the dataset 
*Note: the code below does not download the full Kinetic400 dataset, which takes hours to download*

Data for training baby sign language is currently not available as a public downloadable file. Please contact the authors for the raw video files. Once available unzip the files into a folder and rename it as *raw_videos*. 

### 2.2 Starting containerized jupyter notebook for training
Once the raw videos are downloaded, clone the repository and start the container as shown below.
```sh
git clone https://github.com/atox120/w251_fp.git
cd w251_fp
bash start.sh
```
The bash command will download the neccesary dependecies and start a docker container. If the the run was succesful, the screen should display something like below. It provides a URL to use for jupyter. Just copy and paste in your browser to use it.

```sh
....
....
Successfully tagged swin:bsl
######
Use this URL in your browser:
        http://18.144.168.148:8888/lab?token=7210d7681f095c09f3820575ca7b0ef4595cfbd2343bef82
```

### 2.3 Pre-processing the data
Navigate into the notebooks folder and open *process_video.ipynb*. Run all cells of the notebook. This does the following operations:

1. Sources data from *raw_videos* folder
2. Loops videos that are shorter than 128 frames to make them atleast 128 frames long
3. Makes all videos to have the same frames per second.


### Edge devices - untested. 

***The instructions are for building on a Jetson device.***
Run the script and commands below. 
```sh
git clone https://github.com/atox120/w251_fp.git
cd w251_fp
bash edge_start.sh
```

## Training and Inference
### Inference on tiny Kinetic400 data
Navigate to the notebooks folder and open kinetics_tiny_inference.ipynb and run all cells. Note that the class labels for the dataset **do not match** the trained class labels. The accuracy will thus be near zero.


### Training on tiny Kinetic400 data
Navigate to the notebooks folder and open kinetics_tiny_train.ipynb and run all cells.
