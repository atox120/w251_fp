import os
import sys
import cv2
import numpy as np
from threading import Thread
from collections import deque

sys.path.append(os.path.abspath(""))
sys.path.append(os.path.abspath("../Video-Swin-Transformer"))
os.chdir("../Video-Swin-Transformer")

try:
    import torch
except ImportError:
    torch = None

try:
    from model_inference import get_model, single_gpu_predictor
except ImportError:
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


def capture_frames(cap, analyze_queue, display_queue):
    
    count = 0
    while True:
        # Capture the frame
        ret, frame = cap.read()

        if frame is None:
            # print('None frame')
            continue
        else:
            pass
            # print('Capture frame')
        
        # print(f'Captured frame: {frame.shape}')
        display_queue.append((frame, count))
        
        # Capture onyl odd frames
        if count % 2 == 1:
            # Get a frame and normalize it
            new_frame = normalize(frame)

            # Accumulate frames and make them 6 dimensional (N, S, C, T, W, H)
            #  1. N -> Number of inference videos
            #  2. S -> Number of samples within one video
            #  3. C -> Image channels
            #  4. T -> Frames within video
            new_frame = \
                new_frame.transpose(2, 0, 1)[np.newaxis, np.newaxis, np.newaxis, :, :, :].transpose(0, 1, 3, 2, 4, 5)

            # Add the frames to queues 
            analyze_queue.append((new_frame, count))
        
        count += 1
    

def infer_frames(analyze_queue, label_queue, model=None):
    """
    Takes the latest 32 frames and makes an inference on them. It then converts the inference to a text label

    Args:
        analyze_queue: Queue of frames to analyze
        label_queue: Saves the latest inference of labels
        model: Model for inference

    Returns:

    """

    while True:
        #  These are the frames to analyze and display at teh current moment
        analyze_frames = list(analyze_queue)

        # Display the resulting frame
        if len(analyze_frames) == 32:
            
            # This is the frame count where the label should switch
            frame_counts = [x[1] for x in analyze_frames]
            frame_count = int(np.mean(frame_counts))
            analyze_frames = [x[0] for x in analyze_frames]
            
            # Accumulate into the inference numpy array
            to_tensor = np.concatenate(analyze_frames, axis=3)
            
            # Convert into the format the model is expecting
            if torch is not None:
                # noinspection PyUnresolvedReferences
                data_loader = [{"label": 0, "imgs": torch.from_numpy(to_tensor).to(torch.float32)}]
            else:
                data_loader = []

            # This operation should take about 130ms
            if model is not None:
                outputs = single_gpu_predictor(model, data_loader)
                predicted_class = [np.argmax(x) for x in outputs][0]

                # Predicted probablility
                predicted_prob = outputs[0][predicted_class]
                action = interpret(predicted_class)
                
                if predicted_prob < 0.1:
                    action = '..'
                else:
                    # Add the probability of that action
                    action += ' prob: %.2f' % predicted_prob
            else:
                action = '..'

            # 
            label_queue.append((action, frame_count))
            print(f'Inference {action} on frame count {frame_count}')


def write_frames(mp4_out, display_queue, label_queue):
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
                    
                # Label the 32 frames associated with the action
                if frame_count <= action_frame - 16:
                    # print(f'fc {frame_count} < {action_frame - 16}')
                    if get_model is not None:
                        frame = cv2.putText(frame, '!', (10, 220), font, font_scale, font_color, thickness, line_type)

                    # Writing the frame to mp4 with annotation
                    mp4_out.write(frame)
                    print(f'Writing frame count {frame_count} action !')
                else:
                    # print(f'fc {frame_count} > {action_frame - 16} < ')
                    if get_model is not None:
                        frame = cv2.putText(frame, action, (10, 220), font, font_scale, font_color, thickness, line_type)

                    # Writing the frame to mp4 with annotation
                    mp4_out.write(frame)
                    print(f'Writing frame count {frame_count} action {action}')
                
                # Get the first frame count
                if display_queue:
                    frame, frame_count = display_queue[0]
                else:
                    break


def main():
    """

    Returns:

    """
    # Setup the cofiguration and data file
    config_file = '../configs/bsl_config.py'
    input_video_path = '../notebooks/source_video.mp4'
    check_point_file = './work_dirs/k400_swin_tiny_patch244_window877.py/best_top1_acc_epoch_10.pth'

    # Capture from webcam or capture from file
    do_webcam = True
    if do_webcam:
        # use gstreamer for video directly; set the fps
        camset= "udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=H265 ! rtph265depay ! h265parse ! " \
                "nvv4l2decoder ! nvvidconv ! video/x-raw, format=I420 ! videoconvert ! video/x-raw, format=BGR ! " \
                "appsink drop=1"
        cap = cv2.VideoCapture(camset, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(input_video_path)

    #
    if get_model is not None:
        model = get_model(config_file, check_point_file, distributed=False)
    else:
        model = None

    # This defines the format for the write
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if model is None:
        mp4_out = cv2.VideoWriter('../notebooks/source_video.mp4', fourcc, 20, (224, 224))
    else:
        mp4_out = cv2.VideoWriter('../notebooks/interpreted.mp4', fourcc, 20, (224, 224))
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # mp4_out = cv2.VideoWriter('output.avi', fourcc, 20, (224, 224))

    # Create two queues
    # 1. To hold the images as they are captured
    # 3. Hold annotations of images
    # 2. To hold the images in the format that the anlysis requires
    display_queue = deque([], 300)
    label_queue = deque([], 32)
    analyze_queue = deque([], 32)

    try:
        # Capture the frames as they come in
        capture = Thread(target=capture_frames, args=(cap, analyze_queue, display_queue), daemon=True)
        capture.start()
        
        # Analyse and give them a label
        analyze = Thread(target=infer_frames, args=(analyze_queue, label_queue, model), daemon=True)
        analyze.start()
        
        # Write to file
        write = Thread(target=write_frames, args=(mp4_out, display_queue, label_queue), daemon=True)
        write.start()

        # Join the threads
        capture.join()
        analyze.join()
        write.join()
    except KeyboardInterrupt:
        print('Releasing all')
        # When everything done, release the capture
        cap.release()
        mp4_out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

