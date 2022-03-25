import os
import sys
import time
sys.path.append(os.path.abspath("../Video-Swin-Transformer"))
# Change teh working directory to a location that the code prefers
os.chdir("../Video-Swin-Transformer")

import torch
import warnings
import argparse
import numpy as np
import os.path as osp
from sklearn.metrics import confusion_matrix

import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmaction.models import build_model
from mmaction.utils import register_module_hooks


def single_gpu_predictor(model, data_loader):
    """Test model with a single gpu.
    This method tests model with a single gpu and displays test progress bar.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)
    return results


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)
            
            
def get_model(config_file, check_point_file, distributed = False):
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

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, check_point_file, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])
    
    return model


if __name__ == '__main__':
    # Setup the cofiguration and data file
    config_file_o = '../configs/bsl_config.py'
    check_point_file_o = './work_dirs/k400_swin_tiny_patch244_window877.py/best_top1_acc_epoch_10.pth'
    