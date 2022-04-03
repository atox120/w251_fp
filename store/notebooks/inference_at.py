import os
import sys
import cv2
import time
import logging
import warnings
import numpy as np
from collections import deque
from functools import partial
from datetime import datetime
from threading import Thread, Barrier, BrokenBarrierError

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

sys.path.append(os.path.abspath(""))
sys.path.append(os.path.abspath("../Video-Swin-Transformer"))
os.chdir("../Video-Swin-Transformer")

print('Loading packages...')

import torch
from model_inference import get_model, single_gpu_predictor

try:
    from model_inference import get_model, single_gpu_predictor
except ImportError:
    warnings.warn('Failed to load inference libraries. Inference tasks cannot be performed')
    get_model = None
    single_gpu_predictor = None


def create_logger(level, postfix=''):
    """
    """
    output_folder = '../output'

    # Create a common logger
    logger = logging.getLogger('common')

    # Create the format string
    formatter = logging.Formatter(fmt=' %(name)s - %(levelname)s - %(message)s')

    # Set logger level
    logger.setLevel(level)

    # Create a file handler
    ha = logging.FileHandler(os.path.join(output_folder, 'bsl_%s.log' % postfix))

    # Set print level for handler
    ha.setLevel(level)
    ha.setFormatter(formatter)
    logger.addHandler(ha)

    return logger


def normalize(frame):
    """

    Args:
        frame: Normalize the frame

    Returns:

    """
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


def capture_frames(logger, barrier, pipeline, analyze_queue):
    """
    Capture the frames in a sequence and load to Deque.
        Pauses after first 32 frames

    Args:
        logger: Logger object
        barrier: Threading barrier to allow syncing between threads
        pipeline: Gstreamer pipeline for inference
        analyze_queue: Queue of frames to analyze, stores the reformatted frames

    Returns:
    """

    logger.info('Ready for capture frames')
    pipeline.set_state(Gst.State.PLAYING)
    start = time.perf_counter()
    while True:
        msg = pipeline.get_bus().timed_pop_filtered(
            Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        if msg:
            text = msg.get_structure().to_string() if msg.get_structure() else ''
            msg_type = Gst.message_type_get_name(msg.type)
            print(f'{msg.src.name}: [{msg_type}] {text}')
            break

        # Capture frames
        if len(analyze_queue) == 32:
            logger.info('First 32 frames captured, waiting for inference')
            barrier.wait()

        if len(analyze_queue) == 32:
            now = datetime.now()
            now_str = "%s:%s.%s" % (now.hour, now.minute, now.microsecond)
            logger.info(f'Captured another 32 frames took {time.perf_counter() - start:.2f} time is {now_str}')
            start = time.perf_counter()

    # No more frames being capture, done with input
    logger.info('Completed capturing frames')
    barrier.abort()
    

def infer_frames(logger, barrier, analyze_queue, label_queue, model=None):
    """
    Takes the latest 32 frames and makes an inference on them. It then converts the inference to a text label

    Args:
        logger: Logger object
        barrier: Threading barrier to allow syncing between threads
        analyze_queue: Queue of frames to analyze
        label_queue: Saves the latest inference of labels
        model: Model for inference

    Returns:

    """
    # Wait for other threads to start
    logger.info('infer_frames ready')
    barrier.wait()
    logger.info('infer_frames started')

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
            
            logger.debug(f'Analyzing frame {frame_count}')
            
            # Accumulate into the inference numpy array
            to_tensor = np.concatenate(analyze_frames, axis=3)
            
            # Convert into the format the model is expecting
            if torch is not None:
                # noinspection PyUnresolvedReferences
                logger.debug('loading tensors..')
                data_loader = [{"label": 0, "imgs": torch.from_numpy(to_tensor).to(torch.float32)}]
            else:
                data_loader = []
            
            do_infer = False if previous_frame_count == frame_count else True

            # This operation should take about 130ms
            if model is not None and do_infer:
                outputs = single_gpu_predictor(model, data_loader)
                predicted_class = [np.argmax(x) for x in outputs][0]

                # Predicted probability
                predicted_prob = outputs[0][predicted_class]
                action = interpret(predicted_class)
                
                if predicted_prob < 0.1:
                    action = '..'  

                # Add the probability of that action and inference number
                action = f'{action} - prob {predicted_prob:.2f} - cnt {inferences}'
                logger.info(f'Action: {action}')
            else:
                logger.debug('doing nothing, hence yielding thread')
                time.sleep(0.0001)
                action = '..'
           
            if do_infer:
                label_queue.append((action, frame_count))
                logger.info(f'Inference {action} on frame count {frame_count} inference count {inferences}')
                inferences += 1

            #  
            previous_frame_count = frame_count
        
        if barrier.broken:
            logger.info('Broken barrier waiting 2s to finish inference')
            # Wait 5s for frames to infer
            time.sleep(2)
            break


def display_frames(logger, barrier, display_queue, label_queue):
    """
    Writes frames to MP$ files. Just works through the frame list and adds the appropriate labels

    Args:
        logger: Logger object
        barrier: Threading barrier to allow syncing between threads
        display_queue: Queue of images to display
        label_queue: This is the queue holding the latest labels

    Returns:

    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    # Wait for other threads to start
    logger.info('display_frames ready')
    barrier.wait()
    logger.info('Display thread started')
    
    action_frame = None
    action = '..'
    while True:
        try:
            # It has both action and label
            action, action_frame = label_queue.popleft()
            logger.info(f'Action {action} on frame {action_frame}')
        except IndexError:
            pass

        # Name all labels upto the action frame as previous label
        if display_queue and action_frame is not None:
            # Get the first frame count
            frame, frame_count = display_queue[0]
            
            while frame_count < action_frame + 16:
                frame, frame_count = display_queue.popleft()
                logger.info('new frame')
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
            print('Broken barrier seen on display_frems, stopping')
            break


def write_frames(logger, mp4_out, display_queue, label_queue):
    """
    Writes frames to mp4 files. Just works through the frame list and adds the appropriate labels

    Args:
        logger: Logger object
        mp4_out: This is the CV2 mp4 writer
        display_queue: Queue of images to display
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
            logger.info(f'Action {action} on frame {action_frame}')
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
                    if get_model is not None:
                        frame = cv2.putText(frame, '!', (10, 220), font, font_scale, font_color, thickness, line_type)

                    # Writing the frame to mp4 with annotation
                    mp4_out.write(frame)
                    # print(f'Writing frame count {frame_count} action !')
                else:
                    if get_model is not None:
                        frame = cv2.putText(frame, action, (10, 220), font, font_scale, font_color, thickness,
                                            line_type)

                    # Writing the frame to mp4 with annotation
                    logger.info('wrote now')
                    mp4_out.write(frame)
                    # print(f'Writing frame count {frame_count} action {action}')
                
                # Get the first frame count
                if display_queue:
                    frame, frame_count = display_queue[0]
                else:
                    break


def buffer_to_image(buffer, caps):
    """

    Args:
        buffer: Memory buffer to load from?
        caps:

    Returns:

    """
    caps_structure = caps.get_structure(0)
    height = caps_structure.get_value('height')
    width = caps_structure.get_value('width')
    pixel_bytes = 4

    is_mapped, map_info = buffer.map(Gst.MapFlags.READ)
    if is_mapped:
        try:
            # map buffer to numpy
            image_array = np.ndarray(
                (height, width, pixel_bytes),
                dtype=np.uint8,
                buffer=map_info.data
            )
            # Return a copy of that array as a numpy array.
            return image_array[:, :, :3].copy()
        finally:
            # Clean up the buffer mapping
            buffer.unmap(map_info)

    return None


def on_frame_probe(logger, analyze_queue, display_queue, pad, info):
    """

    Args:
        logger: Logger for logging
        analyze_queue: Queue of frames to analyze, stores the reformatted frames
        display_queue: Queue of frames to display, stores without format changes
        pad:
        info:

    Returns:

    """

    #
    buffer = info.get_buffer()
    frame = buffer_to_image(buffer, pad.get_current_caps())
    if frame is not None:
        start = time.perf_counter()

        # Get the previous frames count
        # If previous element does not exist then count should be zero
        try:
            count = display_queue[-1][1]
        except IndexError:
            count = 0

        # Append the frame without formatting
        display_queue.append((frame, count + 1))

        # Get a frame and normalize it
        new_frame = normalize(frame)

        #  2. S -> Number of samples within one video
        #  3. C -> Image channels
        #  4. T -> Frames within video
        new_frame = \
            new_frame.transpose(2, 0, 1)[np.newaxis, np.newaxis, np.newaxis, :, :, :].transpose(0, 1, 3, 2, 4, 5)

        # Add the frames to queues
        analyze_queue.append((new_frame, count+1))
        logger.info(f'Frame count {count +1} processed in {time.perf_counter() -start:.2f}s')

    # Tell the pipeline everything is good
    return Gst.PadProbeReturn.OK


def create_gst_streamer(logger, analyze_queue, display_queue, sink_to_video=False):
    """

    Args:
        logger: Logger for logging
        analyze_queue: Queue of frames to analyze, stores the reformatted frames
        display_queue: Queue of frames to display, stores without format changes
        sink_to_video: if True, then the sink is video else the sink is memory
    Returns:

    """

    #
    gstreamer_list = [
        "v4l2src device=/dev/video0",  # Get the webcam source
        "nvvidconv ! video/x-raw(memory:NVMM),framerate=(fraction)30/1,width=320,height=240",  # Shrink to size
        # "nvvidconv top=0 bottom=240 left = 90 right=320 ! video/x-raw,width=224,height=224,format=RGBA",
        "nvvidconv top=0 bottom=240 left = 90 right=320 ! video/x-raw,width=224,height=224",  # Center crop
        ]

    if not sink_to_video:
        gstreamer_list += [
            "nvvidconv ! video/x-raw,format=RGBA",  # Move from GPU to CPU memory(same on Jetson)
            "fakesink name=webcam_stream",  # Keep in memory
        ]
    else:
        gstreamer_list += [
            "nvvidconv ! video/x-raw",  # Move from GPU to CPU memory(same on Jetson)
            "fakesink name=webcam_stream",  # Keep in memory
            "queue ! tee name=t t. ! queue ! fakesink name=webcam_stream sync=true t.",  # No idea what this is
            "queue ! nvvidconv ! nvegltransform ! nveglglessink sync=true"  # Sink to screen
        ]

    # Concatenate it all into a string
    gstreamer_string = ""
    for x in gstreamer_list:
        gstreamer_string += x + ' ! '
    gstreamer_string = gstreamer_string[:-3]
    logger.info(f'Gstreamer string is: {gstreamer_string}')

    # Using a partial function create an on frame probe function with logger and queue's
    ofp = partial(on_frame_probe, logger, analyze_queue, display_queue)

    #  Create the pipeline
    Gst.init()
    pipeline = Gst.parse_launch(gstreamer_string)
    pipeline.get_by_name('webcam_stream').get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER, ofp)

    return pipeline


def main(display_video=False, write_to_video=False):
    """

    Returns:

    """

    logger = create_logger(level=10, postfix=str(datetime.now()))

    # Setting up configuration for inference
    device = 'cuda:0'
    # Setup the cofiguration and data file
    config_file = '../configs/bsl_config.py'
    # input_video_path = '../notebooks/source_video.mp4'
    check_point_file = '../configs/best_model.pth'
    # do_webcam = False  # Is the video source a webcam or a video file?
    write_to_video = False  # Write output to video or not?
    display_video = False  # Write output to video or not?

    display_queue = deque([], 1500)
    label_queue = deque([], 300)
    analyze_queue = deque([], 32)

    # Create the Gsstreamer pipeline
    pipeline = create_gst_streamer(logger, analyze_queue, display_queue, sink_to_video=False)

    # Get the model
    if get_model is not None and torch is not None:
        print(f'checking CUDA is available...{torch.cuda.is_available()}')
        torch.cuda.set_device(0) if torch.cuda.is_available() else \
            warnings.warn('No Cuda Deviceswere found, CPU inference will be very slow')
        model = get_model(config_file, check_point_file, device=device)
        print('Model loaded waiting 1s')
        time.sleep(1)
    else:
        model = None

    #
    num_barriers = 2 + display_video # One for video capture, one for inference, one to display(if set)
    barrier = Barrier(num_barriers, timeout=10)  # Create as many barriers and wait 10s for timeout

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
    try:
        # Start the inference first
        analyze = Thread(target=infer_frames, args=(logger, barrier, analyze_queue, label_queue, model), daemon=True)
        analyze.start()
    
        # Capture the frames as they come
        capture = Thread(target=capture_frames, args=(logger, barrier, pipeline, analyze_queue), daemon=True)
        capture.start()
       
        if display_video:
            # This displays the labeled frames
            display = Thread(target=display_frames, args=(logger, barrier, display_queue, label_queue), daemon=True)
            display.start()

        # Write to file
        # write = Thread(target=write_frames, args=(mp4_out, display_queue, label_queue), daemon=True)
        # write.start()

        # Join the threads
        capture.join()
        analyze.join()

        if display_video:
            # noinspection PyUnboundLocalVariable
            display.join()

    except (KeyboardInterrupt, BrokenBarrierError):
        pass
    finally:
        cv2.destroyAllWindows()
        
        if write_to_video:
            # Writing everything 
            logger.info('Writing to inference')
            # noinspection PyUnboundLocalVariable
            write_frames(logger, mp4_out, display_queue, label_queue)
        logger.info('Done')


if __name__ == '__main__':
    main()

