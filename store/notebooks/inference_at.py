import os
import sys
import cv2
import time
import warnings
import numpy as np
from collections import deque
from threading import Thread, Barrier, BrokenBarrierError

sys.path.append(os.path.abspath(""))
sys.path.append(os.path.abspath("../Video-Swin-Transformer"))
os.chdir("../Video-Swin-Transformer")

print('Loading packages...')

try:
    import torch
except ImportError:
    torch = None
    warnings.warn('Torch package was not found and hence not loaded. Inference taks cannot be performed')


try:
    from model_inference import get_model, single_gpu_predictor
except ImportError:
    warnings.warn('Failed to load inference libraries. Inference tasks cannot be performed')
    get_model = None
    single_gpu_predictor = None


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


def capture_frames(barrier, cap, analyze_queue, display_queue):
    """
    Capture the frames in a sequence and load to Deque

    Args:
        analyze_queue: Queue of frames to analyze, stores the reformatted frames
        display_queue: Queue of frames to display, stores without format changes
        model: Model for inference

    Returns:
    """
    # Wait for other threads to start
    print('capture_frames ready')
    barrier.wait()
    print('Capture thread started')
    count = 0
    delay_time = 0
    time_accum = deque([], 30)

    # While the VideoCapturing is open
    while cap.isOpened():
        # Capture the frame
        ret, frame = cap.read()
        start = time.perf_counter()

        if frame is None:
            continue
        else:
            time.sleep(delay_time)
           
        # print(f'Captured frame: {frame.shape}')
        display_queue.append((frame, count))

        # Get a frame and normalize it
        new_frame = normalize(frame)

        #  2. S -> Number of samples within one video
        #  3. C -> Image channels
        #  4. T -> Frames within video
        new_frame = \
            new_frame.transpose(2, 0, 1)[np.newaxis, np.newaxis, np.newaxis, :, :, :].transpose(0, 1, 3, 2, 4, 5)

        # Add the frames to queues 
        analyze_queue.append((new_frame, count))
        
        time_accum.append(time.perf_counter() - start)
        
        # 30 frames per second is 0.033s/frame
        delay_time = 0.033 - np.mean(time_accum)
        delay_time = 0 if delay_time < 0 else delay_time
        count += 1

        if count == 32:
            print('Pausing for first inference')
            time.sleep(60)

        if count % 32 == 0:
            time.sleep(1.0/1000)
    
    # No more frames being capture, done with input
    print('Completed frame loading')
    barrier.abort()
    

def infer_frames(barrier, analyze_queue, label_queue, model=None):
    """
    Takes the latest 32 frames and makes an inference on them. It then converts the inference to a text label

    Args:
        analyze_queue: Queue of frames to analyze
        label_queue: Saves the latest inference of labels
        model: Model for inference

    Returns:

    """
    # Wait for other threads to start
    print('infer_frames ready')
    barrier.wait()
    print('Infer thread started')

    previous_frame_count = -1
    
    inferences = 0
    while True:
        #  These are the frames to analyze and display at teh current moment
        analyze_frames = list(analyze_queue)

        # Display the resulting frame
        if len(analyze_frames) == 32:
            
            # This is the frame count where the label should switch
            frame_counts = [x[1] for x in analyze_frames]
            frame_count = int(np.mean(frame_counts))
            analyze_frames = [x[0] for x in analyze_frames]
            
            print(f'Analyzing frame {frame_count}')
            
            # Accumulate into the inference numpy array
            to_tensor = np.concatenate(analyze_frames, axis=3)
            
            # Convert into the format the model is expecting
            if torch is not None:
                # noinspection PyUnresolvedReferences
                print('loading tensors..')
                data_loader = [{"label": 0, "imgs": torch.from_numpy(to_tensor).to(torch.float32)}]
            else:
                data_loader = []
            
            do_infer = False if previous_frame_count == frame_count else True

            # This operation should take about 130ms
            if model is not None and do_infer:
                outputs = single_gpu_predictor(model, data_loader)
                predicted_class = [np.argmax(x) for x in outputs][0]

                # Predicted probablility
                predicted_prob = outputs[0][predicted_class]
                action = interpret(predicted_class)
                
                if predicted_prob < 0.1:
                    action = '..'  

                # Add the probability of that action and inference number
                action = f'{action} - prob {predicted_prob:.2f} - cnt {inferences}'
                print(f'Action: {action}')
            else:
                print('doing nothing')
                action = '..'
           
            if do_infer:
                # 
                label_queue.append((action, frame_count))
                print(f'Inference {action} on frame count {frame_count} inference count {inferences}')
                inferences += 1
                time.sleep(1.0/1000)
            
            #  
            previous_frame_count = frame_count
        
        if barrier.broken:
            print('Broken barrier')
            # Wait 5s for frames to infer
            time.sleep(5)
            break


def display_frames(barrier, display_queue, label_queue):
    """
    Writes frames to MP$ files. Just works through the frame list and adds the appropriate labels

    Args:
        mp4_out: This is the CV2 mp4 writer
        label_queue: This is the queue holding the latest labels

    Returns:

    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    # Wait for other threads to start
    print('display_frames ready')
    barrier.wait()
    print('Display thread started')
    
    action_frame = None
    action = '..'
    while True:
        try:
            # It has both action and label
            action, action_frame = label_queue.popleft()
            print(f'Action {action} on frame {action_frame}')
        except IndexError:
            pass

        # Name all labels upto the action frame as previous label
        if display_queue and action_frame is not None:
            # Get the first frame count
            frame, frame_count = display_queue[0]
            
            while frame_count < action_frame + 16:
                frame, frame_count = display_queue.popleft()
                print('new frame')   
                # Label the 32 frames associated with the action
                if frame_count > action_frame - 16:
                    if get_model is not None:
                        frame = cv2.putText(frame, action, (10, 220), font, font_scale, font_color, thickness, line_type)

                    # Writing the frame to mp4 with annotation
                    cv2.imshow(frame)
               
                # If there are no more elements on the display queue then quit
                if display_queue:
                    frame, frame_count = display_queue[0]
                else:
                    break
        
        if barrier.broken:
            print('Broken barrier')
            # Wait 5s for frames to infer
            time.sleep(5)
            break


def write_frames(mp4_out, display_queue, label_queue):
    """
    Writes frames to mp4 files. Just works through the frame list and adds the appropriate labels

    Args:
        mp4_out: This is the CV2 mp4 writer
        label_queue: This is the queue holding the latest labels

    Returns:

    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2
    
    action_frame = None
    action = '..'
    while True:
        try:
            # It has both action and label
            action, action_frame = label_queue.popleft()
            print(f'Action {action} on frame {action_frame}')
        except IndexError:
            pass

        # Name all labels upto the action frame as previous label
        if display_queue and action_frame is not None:
            # Get the first frame count
            frame, frame_count = display_queue[0]
            
            while frame_count < action_frame + 16:
                # Did not interpret these frames
                frame, frame_count = display_queue.popleft()
                print('new frame')   
                # Label the 32 frames associated with the action
                if frame_count <= action_frame - 16:
                    if get_model is not None:
                        frame = cv2.putText(frame, '!', (10, 220), font, font_scale, font_color, thickness, line_type)

                    # Writing the frame to mp4 with annotation
                    mp4_out.write(frame)
                    # print(f'Writing frame count {frame_count} action !')
                else:
                    if get_model is not None:
                        frame = cv2.putText(frame, action, (10, 220), font, font_scale, font_color, thickness, line_type)

                    # Writing the frame to mp4 with annotation
                    print('wrote now')
                    mp4_out.write(frame)
                    # print(f'Writing frame count {frame_count} action {action}')
                
                # Get the first frame count
                if display_queue:
                    frame, frame_count = display_queue[0]
                else:
                    break

def main():
    """

    Returns:

    """

    # Setting up configuration for inference
    device = 'cuda:0'
    # Setup the cofiguration and data file
    config_file = '../configs/bsl_config.py'
    input_video_path = '../notebooks/source_video.mp4'
    check_point_file = '../configs/best_model.pth'
    do_webcam = False  # Is the video source a webcam or a video file?
    write_to_video = False  # Write output to video or not?
    display_video = False  # Write output to video or not?
    
    # Check whether webcam is enabled
    if do_webcam:
        if not input_video_path:
            raise ValueError('As webcam source was not chosen an input video path must be provided')

    if get_model is not None and torch is not None:
        print(f'checking CUDA is available...{torch.cuda.is_available()}')
        torch.cuda.set_device(0) if torch.cuda.is_available() else warnings.warn('No Cuda Deviceswere found, CPU inference will be very slow')
        model = get_model(config_file, check_point_file, device=device)
        print('Model loaded waiting 1s')
        time.sleep(1)
    else:
        model = None

    #   
    num_barriers = 2 + display_video # One for video capture, one for inference, one to display(if set)
    barrier = Barrier(num_barriers, timeout=10)  # Create as many barriers and wait 10s for timeout


    # Enable the nput source for cv2
    # 1. Either through webcam
    if do_webcam:
        # This is the Gstreamer for capturing webcam and converting to frames
        webcam_str = "v4l2src ! "
        format_str_1 = "nvvidconv ! video/x-raw(memory:NVMM),width=404,height=224,framerate=30/1 ! "
        format_str_2 = "nvvidconv top=0 bottom=224 left=90 right=314 ! video/x-raw(memory:NVMM),width=224,height=224,framerate=30/1 ! "  
        format_str_2 = "nvvidconv ! video/x-raw,width=224,height=224,framerate=30/1 ! "  # This copies from NVMM memory to CPU memory
        format_change_str = "videoconvert ! video/x-raw,format=BGR ! "
        appsink_str = "appsink drop=1"
        camset = webcam_str + format_str_1 + format_str_2 + format_change_str + appsink_str
        print(camset)

        # Capture frame using Gstreamer
        cap = cv2.VideoCapture(camset, cv2.CAP_GSTREAMER)
    # 2. Or thorugh a file
    else:
        cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        cap.release()
        raise ValueError('OpenCV either could not read from webcam or read from file')

    # If the output frames must be sent to a video file then 
    if write_to_video:
        print('Starting writer')
        # This defines the format for the write
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mp4_out = cv2.VideoWriter('../notebooks/out_video.mp4', fourcc, 20, (224, 224))

    # Create two queues
    # 1. To hold the images as they are captured
    # 3. Hold annotations of images
    # 2. To hold the images in the format that the anlysis requires
    display_queue = deque([], 1500)
    label_queue = deque([], 300)
    analyze_queue = deque([], 32)
    try:
        # Start the inference first
        analyze = Thread(target=infer_frames, args=(barrier, analyze_queue, label_queue, model), daemon=True)
        analyze.start()
    
        # Capture the frames as they come
        capture = Thread(target=capture_frames, args=(barrier, cap, analyze_queue, display_queue), daemon=True)
        capture.start()
       
        if display_video:
            # This displays the labeled frames
            display = Thread(target=display_frames, args=(barrier, display_queue, label_queue), daemon=True)
            display.start()

        # Write to file
        # write = Thread(target=write_frames, args=(mp4_out, display_queue, label_queue), daemon=True)
        # write.start()

        print('Synchornizing threads')
        # Join the threads
        capture.join()
        analyze.join()

        if display_video:
            display.join()

    except (KeyboardInterrupt, BrokenBarrierError):
        pass
    finally:

        cap.release()
        cv2.destroyAllWindows()
        
        if write_to_video:
            # Writing everything 
            print('Writing to inference')
            write_frames(mp4_out, display_queue, label_queue)
        print('Done')


if __name__ == '__main__':
    main()

