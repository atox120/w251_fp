# W251 Final Project: Video action detection for baby sign language  
# Alex To, Aswin Thiruvengadam, Dan Ortiz, Jeffrey Laughman

## Introduction

blah blah blah 

## Setting up the Environment 
### Cloud 
*Note: the code below does not download the full Kinetic400 dataset, which takes hours to download*

```sh
git clone https://github.com/atox120/w251_fp.git
cd w251_fp
bash start.sh
```

The bash command will download the neccesary files and start a docker container. This automatically downloads a tiny subset of the Kinetic400 dataset. This tiny dataset has:

* For training set:
	* Two classes
	* 15 images per class
* For validation set:
	* Two classes 
	* 5 images per class
In this version of the code, the lables of this validaiton set do not match the training. Hence the model accuracy will be low. If you woud like to download the entire Kinetic400 dataset then run thw following command

*Note: the code below downloads then FULL Kinetic400 dataset, beware this will take very long.* 

```sh
git clone https://github.com/atox120/w251_fp.git
cd w251_fp
bash start.sh kinetic
```  

If the the run was succesful, the screen should display something like below. It provides a URL to use for jupyter. Just copy and paste in your browser to use it.

```sh
....
....
Successfully tagged swin:bsl
######
Use this URL in your browser:
        http://18.144.168.148:8888/lab?token=7210d7681f095c09f3820575ca7b0ef4595cfbd2343bef82
```
### Edge devices - untested. 
Run the script and commands below. 
```sh
git clone https://github.com/atox120/w251_fp.git
cd w251_fp
bash edge_start.sh
```

*The instructions are for building on a Jetson device. *

## Training and Inference
### Inference on tiny Kinetic400 data
Navigate to the notebooks folder and open kinetics_tiny_inference.ipynb and run all cells. Note that the class labels for the dataset **do not match** the trained class labels. The accuracy will thus be near zero.


### Training on tiny Kinetic400 data
Navigate to the notebooks folder and open kinetics_tiny_train.ipynb and run all cells.
