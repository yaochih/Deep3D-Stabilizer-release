# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Deep3D stabilizer options")

        # RUN COMMAND
        self.parser.add_argument('video_path',
                                 type=str,
                                 help='path to input video')
        self.parser.add_argument("--name",
                                 type=str,
                                 help="the name of the video folder to process",
                                 default="test")

        self.parser.add_argument("--output_dir",
                                 type=str,
                                 help='output depths directory',
                                 default='outputs')
        

        # TRAINING options
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=128)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=192)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0])
        self.parser.add_argument("--intervals",
                                 nargs="+",
                                 type=int,
                                 help='the interval of  nearby view for supvervision',
                                 default=[1, 4, 9])
        self.parser.add_argument('--rotation_mode',
                                 type=str,
                                 choices=['euler', 'quat'],
                                 default='quat',
                                 help='the rotation mode of pose vector')
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=1e-3)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=10.0)
        self.parser.add_argument('--img_mean',
                                 type=float,
                                 help='normalized mean',
                                 default=0.45)
        self.parser.add_argument('--img_std',
                                 type=float,
                                 help='normalized standard deviation',
                                 default=0.225)
        # LOSS WEIGHTS
        self.parser.add_argument('--photometric_loss',
                                 type=float,
                                 help='the weight of photometric loss',
                                 default=1.0)
        self.parser.add_argument('--geometry_loss',
                                 type=float,
                                 help='the weight of geometry consistency loss',
                                 default=0.5)
        self.parser.add_argument('--ssim_weight', 
                                 type=float,
                                 help='ssim weight',
                                 default=0.5)
        self.parser.add_argument('--flow_loss',
                                 type=float,
                                 help='the weight of flow consistency loss',
                                 default=10.0)
        self.parser.add_argument('--adaptive_alpha',
                                 type=float,
                                 default=1.2)
        self.parser.add_argument('--adaptive_beta',
                                 type=float,
                                 default=0.85)
        
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=80)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=2e-4)
        self.parser.add_argument("--init_num_epochs",
                                 type=int,
                                 help="number of epochs for initialization",
                                 default=300)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=100)
        
        # SYSTEM
        self.parser.add_argument('--cuda',
				 default='cuda:0',
				 help='indicate cuda device')
        self.parser.add_argument('--img_extension',
                                 choices=['jpg', 'png'],
                                 default='png',
                                 help='the data type of input frames')
        self.parser.add_argument('--save_together',
                                 action='store_true',
                                 dest='save_together',
                                 help='save all result video in a single directory')

        # recitification
        self.parser.add_argument('--stability',
                                 type=int,
                                 help='std of gaussian filter for smoothing trajectory',
                                 default=12)
        self.parser.add_argument('--smooth_window',
                                 type=int,
                                 help='window size of moving average filter',
                                 default=59)
        self.parser.add_argument('--post_process',
                                 action='store_true',
                                 default=True,
                                 help='handle dynamic objects in post processing')

        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
