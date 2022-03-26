import cv2
import numpy as np
from threading import Thread
from collections import deque

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False


def normalize(frame):
    mean = np.reshape([123.675, 116.28, 103.53], (1, 1, 3)) 
    std = np.reshape([58.395, 57.12, 57.375], (1, 1, 3))
    
    # First mean normalize the data
    frame = frame - mean

    # Next normalize the stdev
    frame = frame/std
    
    return frame



def interpret(index):
    text_form = ['all_done', 'water', 'poop', 'dad', 'mom']

    return text_form[index]


def capture_frames(cap, analyze_queue, display_queue):
    
    while(True):
        # Capture the frame
        ret, frame = cap.read()

        if frame is None:
            print('None frame')
            continue
        else:
            print('Capture frame')
        
        # Get a frame and normalize it
        new_frame = normalize(frame)

        # Accumulate frames and make them 6 dimensional (N, S, C, T, W, H)
        #  1. N -> Number of inference videos
        #  2. S -> Number of samples within one video
        #  3. C -> Image channels
        #  4. T -> Frames within video


        new_frame = new_frame.transpose(2, 0, 1)[np.newaxis, np.newaxis, np.newaxis, :, :, :].transpose(0, 1, 3, 2, 4, 5)
        
        # Add the frames to queues 
        analyze_queue.append(new_frame)
        display_queue.append(frame)
    

def analyze_frames(analyze_queue, display_queue, model=None):

    while(True):
        #  These are the frames to analyze and display at teh current moment
        analyze_frames = list(analyze_queue)
        display_frames = list(display_queue)
        
        #  Check for non empty frames
        analyze_frames  = [x for x in analyze_frames if isinstance(x, np.ndarray)]

        # Display the resulting frame
        if len(analyze_frames) == 32:
            #  Accumulate into the inference numpy array
            to_tensor = np.concatenate(analyze_frames, axis=3)
            
            # Convert into the format the model is expecting
            if torch_available:
                data_loader = [{"label": 0, "imgs": torch.from_numpy(to_tensor).to(torch.float32)}]

            # TODO - add inference here
            print('TODO- convert to tensor and inference, numpy array shape %s' % str(to_tensor.shape))

            # This operation should take about 130ms
            # outputs = single_gpu_test(model, data_loader)
            # predicted_class = [np.argamx(x) for x in outputs][0]
            # action = interpret(predicted_class)
            
            print('TODO- Add text tag to frames and save the videos')


def write_frames():


while(True):
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():

    # use gstreamer for video directly; set the fps
    camSet= "udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=H265 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=I420 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    # camSet = "v4l2src device=/dev/video0 ! nvvidconv ! 'video/x-raw(memory:NVMM),width=404,height=224,framerate=32/1' ! nvvidconv top=0 bottom=224 left=90 right=314 ! 'video/x-raw(memory:NVMM),width=224,height=224,framerate=32/1' ! nvvidconv ! 'video/x-raw, format=I420' ! videoconvert ! 'video/x-raw, format=BGR' ! appsink drop=1"

    cap = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)

    # This defines the format for the write
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (224, 224))

    # Create two queues
    # 1. To hold the images as they are captured
    # 2. Hold annotated images as they are labeled
    # 2. To hold the images in the format that the anlysis requires
    display_queue = deque([] *32, 32)
    labeled_queue = deque([] *32, 32)
    analyze_queue = deque([] *32, 32)

    try:
        # Capture the frames as they come in
        capture = Thread(target=capture_frames, args=(cap, analyze_queue, display_queue), daemon=True)
        capture.start()
        
        # Analyse and give them a label
        analyze = Thread(target=analyze_frames, args=(analyze_queue, display_queue), daemon=True)
        analyze.start()
        
        # Write to file
        write = Thread(target=write_frames, args=(out, labeled_queue), daemon=True)
        write.start()

        capture.join()
    except KeyboardInterrupt:
        # When everything done, release the capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

