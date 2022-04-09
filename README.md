# W251 Final Project: Video action detection for baby sign language  
# Alex To, Aswin Thiruvengadam, Dan Ortiz, Jeffrey Laughman

## 1. Introduction
Baby sign language is comprised of a series of symbols (closely related to ASL) to improve communication between babies and caregivers prior to vocal vocabulary develops. The goal is to reduce crying and tantrums by providing a medium of communication and reducing caregiver guessing.

## 2. Training on a cloud VM or GPU workstation 

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

If a URL did not output on the screen then check to see if the container is running and check docker logs for the URL. This can be done by:
```sh
docker ps
# Find the container ID of the running docker swin:bsl
docker logs <container_id>
# Look for the line with the URL
```

### 2.3 Pre-processing the data
Navigate into the notebooks folder and open *process_video.ipynb*. Run all cells of the notebook. This does the following operations:

1. Sources data from *raw_videos* folder
2. Loops videos that are shorter than 128 frames to make them at least 128 frames long
3. Makes all videos to have the same frames per second.
4. Split data into three datasets:
   4.1 *Train and validation* - These are videos sourced from 'dan-round-1', 'jeff-round-1' and 'aswin-round-1'. The videos are randomly sent to the validation or the test set.
   4.2 *Test* - These are videos solely sourced from 'alex-round-1'. This was done to ensure that *training and validation* had not seen the person and the background in the test set. The measure of the test set then gave confidence on the generalization of the model.
5. Pre-processed data is copied to the 'processed_videos' folder


### 2.4 Training the model
Open the notebook *bsl_train.ipynb* and run all cells. This notebook trains the model. The progress of the training can be seen in the output of cell 27 and the starting of the training should look similar to this:

```sh
2022-04-08 00:35:16,724 - mmaction - INFO - Start running, host: root@ip-10-0-0-144, work_dir: /workspace/Video-Swin-Transformer/work_dirs/k400_swin_tiny_patch244_window877.py
2022-04-08 00:35:16,724 - mmaction - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) CosineAnnealingLrUpdaterHook       
(ABOVE_NORMAL) DistOptimizerHook                  
(NORMAL      ) CheckpointHook                     
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) CosineAnnealingLrUpdaterHook       
(NORMAL      ) WandBHook                          
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_iter:
(VERY_HIGH   ) CosineAnnealingLrUpdaterHook       
(LOW         ) IterTimerHook                      
 -------------------- 
after_train_iter:
(ABOVE_NORMAL) DistOptimizerHook                  
(NORMAL      ) CheckpointHook                     
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) CheckpointHook                     
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_epoch:
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_epoch:
(NORMAL      ) WandBHook                          
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
after_run:
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
2022-04-08 00:35:16,725 - mmaction - INFO - workflow: [('train', 1), ('val', 1)], max: 15 epochs
2022-04-08 00:35:16,726 - mmaction - INFO - Checkpoints will be saved to /workspace/Video-Swin-Transformer/work_dirs/k400_swin_tiny_patch244_window877.py by HardDiskBackend.
2022-04-08 00:35:35,273 - mmaction - INFO - Epoch [1][20/111]	lr: 1.616e-05, eta: 0:25:25, time: 0.927, data_time: 0.182, memory: 2832, top1_acc: 0.2000, top5_acc: 1.0000, loss_cls: 1.5963, loss: 1.5963
2022-04-08 00:35:48,277 - mmaction - INFO - Epoch [1][40/111]	lr: 2.265e-05, eta: 0:21:21, time: 0.650, data_time: 0.001, memory: 2832, top1_acc: 0.2000, top5_acc: 1.0000, loss_cls: 1.6342, loss: 1.6342
2022-04-08 00:36:01,283 - mmaction - INFO - Epoch [1][60/111]	lr: 2.914e-05, eta: 0:19:51, time: 0.650, data_time: 0.001, memory: 2832, top1_acc: 0.0500, top5_acc: 1.0000, loss_cls: 1.6424, loss: 1.6424
2022-04-08 00:36:14,280 - mmaction - INFO - Epoch [1][80/111]	lr: 3.562e-05, eta: 0:19:00, time: 0.650, data_time: 0.001, memory: 2832, top1_acc: 0.3750, top5_acc: 1.0000, loss_cls: 1.5847, loss: 1.5847
2022-04-08 00:36:27,272 - mmaction - INFO - Epoch [1][100/111]	lr: 4.211e-05, eta: 0:18:23, time: 0.650, data_time: 0.001, memory: 2832, top1_acc: 0.4000, top5_acc: 1.0000, loss_cls: 1.5560, loss: 1.5560
2022-04-08 00:36:34,133 - mmaction - INFO - Saving checkpoint at 1 epochs
/opt/conda/lib/python3.8/site-packages/torch/utils/checkpoint.py:25: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 55/55, 2.9 task/s, elapsed: 19s, ETA:     0s
```

The pytorch model files are stored in the folder *workspace/Video-Swin-Transformer/work_dirs/k400_swin_tiny_patch244_window877.py*. Note the .py in the folder name, this is by design. There is a .pth for each epoch but the best model is updated with the name "best_top_1_acc_epoch_<number>.pth". The latest epoch model with that label is the best model for inference based on the performance on the validation dataset.

*Download the, above described, best model to your local folder by navigating into the folder, right clock and download*. Rename the file simply as best_model.pth. copt it into *workspace/configs* folder. This will now be used for inference. 
        
### 2.5 Test on the cloud
To test the model performance on the cloud open *bsl_test.py* and run all cells. This should list out the accuracy score and a confusion matrix.

```sh
Evaluating top_k_accuracy ...

top1_acc	0.8000
top5_acc	1.0000
top1_acc: 0.8000
top5_acc: 1.0000
```
       
![image](https://user-images.githubusercontent.com/76710118/162588097-2587ad3a-ede8-4ef0-ab23-551b68f019a9.png)
        
### 2.6 

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
