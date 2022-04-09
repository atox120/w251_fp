import os
import sys
sys.path.append(os.path.abspath(""))
sys.path.append(os.path.abspath("../Video-Swin-Transformer"))
import cv2
import six
import time
import torch
import logging
import warnings
import argparse
import numpy as np
from collections import deque
from functools import partial
from datetime import datetime
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmaction.models import build_model
from mmaction.utils import register_module_hooks
from threading import Thread, Barrier, BrokenBarrierError

os.chdir("../Video-Swin-Transformer")
print('Loading packages...')


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
    filename = 'bsl_%s.log' % postfix
    haf = logging.FileHandler(os.path.join(output_folder, filename))

    # Create a StreamHandle for displaying on screen
    ha = logging.StreamHandler()

    # Set print level for handler
    haf.setLevel(level)
    haf.setFormatter(formatter)

    # Set
    ha.setLevel(level)
    ha.setFormatter(formatter)

    #
    logger.addHandler(haf)
    logger.addHandler(ha)

    return logger, filename


def normalize(frame):
    mean = np.reshape([123.675, 116.28, 103.53], (1, 1, 3))
    std = np.reshape([58.395, 57.12, 57.375], (1, 1, 3))

    # First mean normalize the data
    frame = frame - mean

    # Next normalize the stdev
    frame = frame / std

    return frame


def interpret(index):
    text_form = ['all_done', 'water', 'poop', 'dad', 'mom']

    return text_form[index]


def capture_frames(logger, barrier, cap, analyze_queue, display_queue):
    """

    Args:
        cap: OpenCv object
        logger: Logger for logging
        barrier: Threading barrier object
        analyze_queue: Queue of frames to analyze, stores the reformatted frames
        display_queue: Queue of frames to display, stores without format changes

    Returns:

    """

    while cap.isOpened():
        start = time.perf_counter()
        # Capture the frame
        ret, frame = cap.read()
        
        if frame is None:
            break

        # If previous element does not exist then count should be zero
        try:
            count = display_queue[-1][1]
        except IndexError:
            count = 0

        count += 1

        # Append the frame without formatting
        display_queue.append((frame, count))

        # Get a frame and normalize it
        new_frame = normalize(frame)

        #  2. S -> Number of samples within one video
        #  3. C -> Image channels
        #  4. T -> Frames within video
        new_frame = \
            new_frame.transpose(2, 0, 1)[np.newaxis, np.newaxis, np.newaxis, :, :, :].transpose(0, 1, 3, 2, 4, 5)

        # Add the frames to queues
        new_frame = torch.from_numpy(new_frame).to(torch.float32)  # .cuda()
        analyze_queue.append((new_frame, count))

        # Capture frames
        if count == 32:
            logger.info('First 32 frames captured, waiting for inference upto 60s')
            barrier.wait(timeout=60)

        if count % 10 == 0:
            # Handing control to other threads every 10 frames
            logger.info('Gstreamer handling control to other threads')
            time.sleep(0.01)

        frame_delay = 1.0/30 - (start-time.perf_counter())  # 30 FPS
        frame_delay = 0 if frame_delay < 0 else frame_delay
        time.sleep(frame_delay)
        logger.info(f'Frame count {count} processed in {time.perf_counter() - start:.2f}s delaying {frame_delay}s')
    
    barrier.abort()
    logger.info(f'All frames received!')
    

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

    previous_frame_count = -1
    inferences = 0
    while True:
        #  These are the frames to analyze and display at teh current moment
        analyze_frames = list(analyze_queue)

        # Display the resulting frame
        if len(analyze_frames) == 32:

            start = time.perf_counter()

            # This is the frame count where the label should switch
            frame_counts = [x[1] for x in analyze_frames]
            frame_count = int(np.mean(frame_counts))
            analyze_frames = [x[0] for x in analyze_frames]

            # Accumulate into the inference numpy array
            # to_tensor = np.concatenate(analyze_frames, axis=3)
            batch_tensor = torch.cat(analyze_frames, dim=3)

            do_infer = False if frame_count == previous_frame_count else True

            # This operation should take about 130ms
            if model is not None and do_infer:
                logger.info(f'Preparing inference on frame count {frame_count} shape {batch_tensor.shape}' +
                            f' took {time.perf_counter() - start:.3f}s')
                # noinspection PyBroadException
                try:
                    with torch.no_grad():
                        outputs = model(return_loss=False, **{"label": 0, "imgs": batch_tensor})
                    predicted_class = np.argmax(outputs)
                except Exception:
                    t, v, tb = sys.exc_info()
                    logger.error("Fatal error in model inference", exc_info=True)
                    six.reraise(t, v, tb)
                    return

                # Predicted probability
                predicted_prob = outputs[0][predicted_class]
                action = interpret(predicted_class)

                if predicted_prob > 0.5:
                    # Add the probability of that action and inference number
                    action = f'{action} - prob {predicted_prob:.2f} - cnt {inferences}'
                else:
                    # Add the probability of that action and inference number
                    action = f'Unsure - cnt {inferences}'   

                if inferences == 0:
                    barrier.wait(timeout=2)
                    logger.info('infer_frames started')
            else:
                logger.debug('doing nothing, hence yielding thread with sleep')
                time.sleep(0.01)
                action = '..'

            if do_infer:
                label_queue.append((action, frame_count))
                logger.info(f'Inference {action} on frame count {frame_count} inference count {inferences}' +
                            f' took {time.perf_counter() - start:.2f}s'
                            )
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
    barrier.wait(timeout=60)
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
                logger.info('new frame')
                # Label the 32 frames associated with the action
                if frame_count > action_frame - 16:
                    frame = \
                        cv2.putText(frame, action, (10, 220), font, font_scale, font_color, thickness, line_type)

                    # Writing the frame to mp4 with annotation
                    cv2.imshow('Frame', frame)

                # If there are no more elements on the display queue then quit
                if display_queue:
                    frame, frame_count = display_queue[0]
                else:
                    break

        if barrier.broken:
            print('Broken barrier seen on display_frame, stopping')
            break


def write_frames_all(logger, display_queue, label_queue):
    """
    Writes frames to mp4 files. Just works through the frame list and adds the appropriate labels

    Args:
        logger: Logger object
        display_queue: Queue of images to display
        label_queue: This is the queue holding the latest labels

    Returns:

    """

    logger.info('Starting writer')
    # This defines the format for the write
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mp4_out = cv2.VideoWriter('../output/out_video.mp4', fourcc, fps=15, frameSize=(224, 224))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    label_list = list(label_queue)
    display_list = list(display_queue)

    # Get each label
    for action, action_frame in label_list:
        logger.info(f'writing: Action {action} on frame {action_frame}')

        # Get the first frame count
        drop_frames = 0
        for frame, frame_count in display_list:
            # Label the 32 frames associated with the action
            if frame_count <= action_frame - 16:
                frame = cv2.putText(frame, '!', (10, 220), font, font_scale, font_color, thickness, line_type)

                # Writing the frame to mp4 with annotation
                logger.info(f'Writing frame {frame_count} without annotation')
                mp4_out.write(frame)
                drop_frames += 1

            elif frame_count < action_frame + 16:
                frame = cv2.putText(frame, action, (10, 220), font, font_scale, font_color, thickness,
                                    line_type)

                # Writing the frame to mp4 with annotation
                logger.info(f'Writing frame {frame_count} with annotation')
                mp4_out.write(frame)
                drop_frames += 1
            else:
                break

        display_list = display_list[drop_frames:]

    mp4_out.release()


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def prepare_model(logger, config_file, check_point_file, device='cuda:0', half_precision=False):
    """
    Setup the model for half precision

    """
    # Create the configuration from the file
    # Customization for training the BSL data set
    cfg = Config.fromfile(config_file)
    cfg.model.cls_head.num_classes = 5
    cfg.data.test.test_mode = True

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    # On a Jetson Nano this actually makes things slower
    if half_precision:
        logger.info('Enabling half precision')
        wrap_fp16_model(model)

    load_checkpoint(model, check_point_file, map_location=device)
    # model.cuda(0)

    # Does not work without this, need to checkout why
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    return model


def main(args):
    """

    Returns:

    """
    display_video = args.display_video
    write_video = args.write_video
    half_precision = args.half_precision

    logger, filename = create_logger(level=10, postfix=str(datetime.now()))

    # Setting up configuration for inference
    device = 'cuda:0'
    # Setup the cofiguration and data file
    config_file = '../configs/bsl_config.py'
    input_video_path = '../notebooks/source_video.mp4'
    check_point_file = '../configs/best_model.pth'
    # do_webcam = False  # Is the video source a webcam or a video file?

    display_queue = deque([], 1500)
    label_queue = deque([])
    analyze_queue = deque([], 32)

    #
    num_barriers = 2 + display_video  # One for video capture, one for inference, one to display(if set)
    barrier = Barrier(num_barriers, timeout=10)  # Create as many barriers and wait 10s for timeout

    logger.info('Inference on cloud with options:')
    logger.info(f'\t display_video {display_video}')
    logger.info(f'\t write_video {write_video}')
    logger.info(f'\t half_precision {half_precision}')
    logger.info(f'\t Number of barriers {num_barriers}')

    # Create the Gsstreamer pipeline
    # use gstreamer for video directly; set the fps
    
    # camset = "udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=H265 ! rtph265depay ! h265parse ! " \
    #         "nvv4l2decoder ! nvvidconv ! video/x-raw, format=I420 ! videoconvert ! video/x-raw, format=BGR ! " \
    #         "appsink drop=1"
    # cap = cv2.VideoCapture(camset, cv2.CAP_GSTREAMER)

    cap = cv2.VideoCapture(input_video_path)

    # Get the model
    logger.info(f'CUDA is available...{torch.cuda.is_available()}')
    torch.cuda.set_device(0) if torch.cuda.is_available() else \
        warnings.warn('No Cuda Deviceswere found, CPU inference will be very slow')

    # Get the model ready for inference
    model = prepare_model(logger, config_file, check_point_file, device=device, half_precision=half_precision)

    # Create two queues
    # 1. To hold the images as they are captured
    # 3. Hold annotations of images
    # 2. To hold the images in the format that the anlysis requires
    try:
        # Start the inference first
        analyze = Thread(target=infer_frames, args=(logger, barrier, analyze_queue, label_queue, model), daemon=True)
        analyze.start()

        # Capture the frames as they come
        capture = Thread(target=capture_frames, args=(logger, barrier, cap, analyze_queue, display_queue), daemon=True)
        capture.start()

        if display_video:
            # This displays the labeled frames
            display = Thread(target=display_frames, args=(logger, barrier, display_queue, label_queue), daemon=True)
            display.start()

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
        time.sleep(1)

        if write_video:
            # Writing everything
            logger.info('Writing to inference')
            # noinspection PyUnboundLocalVariable
            write_frames_all(logger, display_queue, label_queue)
        logger.info('Done')


def parse_args(parse_options=None):
    """

    Args:
        parse_options:

    Returns:

    """
    parser = argparse.ArgumentParser(description='Model arguments parser')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument(
        '--no-display-video',
        action='store_false',
        dest='display_video',
        help='Display labeled videos')
    feature_parser.add_argument(
        '--display-video',
        action='store_true',
        dest='display_video',
        help='Display labeled videos')
    parser.set_defaults(display_video=False)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument(
        '--no-write-video',
        action='store_false',
        dest='write_video',
        help='Store videos to file')
    feature_parser.add_argument(
        '--write-video',
        action='store_true',
        dest='write_video',
        help='Store videos to file')
    parser.set_defaults(write_video=True)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument(
        '--no-half-precision',
        action='store_false',
        dest='half_precision',
        help='Store videos to file')
    feature_parser.add_argument(
        '--half-precision',
        action='store_true',
        dest='half_precision',
        help='Store videos to file')
    parser.set_defaults(half_precision=False)

    if parse_options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(parse_options)

    return args


if __name__ == '__main__':

    # inline_options = ['--display-video', '1', '--write-video', '0']
    inline_options = None
    # Get the command prompt options for inference
    argso = parse_args(inline_options)

    main(argso)
