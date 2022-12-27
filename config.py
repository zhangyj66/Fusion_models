import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse


import os

print(os.getcwd())

C = edict()
config = C
cfg = C

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'NYUDv2_15'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'NYUDv2_15')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset

C.x_root_folder = osp.join(C.dataset_path, 'HHA')
C.x_format = '.jpg'
C.x_is_single_channel = False # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = 10
C.num_eval_imgs = 5
C.num_classes = 40
C.class_names = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
    'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
    'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
    'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']


"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])


"""Train Config"""
C.GPU_ID = 0
C.hyper_parm = 20 # hyperparameter of fusion model
C.batch_size = 4
C.train_scale_array = None
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.lr = 1e-3
C.epoch = 100
C.weight_decay = 1e-4
C.save_dir = './model_results/'
C.save_img_dir = './visualization/'

#
# """Eval Config"""
C.model_path = './model_results/train_model_100.pth'
C.show_image = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()