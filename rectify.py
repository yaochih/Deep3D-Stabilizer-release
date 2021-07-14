import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, sys, os, csv, time, datetime
from tqdm import tqdm
from path import Path
from PIL import Image
import options
from warper import Warper
from sequence_io import *
from smooth import smooth_trajectory, get_smooth_depth_kernel
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter as scipy_gaussian
import warnings
warnings.filterwarnings('ignore')

def get_cropping_area(warp_maps, h, w):
    border_t = warp_maps[:, 0, :, 1][warp_maps[:, 0, :, 1] >= 0]
    border_b = warp_maps[:, -1, :, 1][warp_maps[:, -1, :, 1] >= 0]
    border_l = warp_maps[:, :, 0, 0][warp_maps[:, :, 0, 0] >= 0]
    border_r = warp_maps[:, :, -1, 0][warp_maps[:, :, -1, 0] >= 0]
    
    t = int(torch.ceil(torch.clamp(torch.max(border_t), 0, h))) if border_t.shape[0] != 0 else 0
    b = int(torch.floor(torch.clamp(torch.min(border_b), 0, h)))if border_b.shape[0] != 0 else 0
    l = int(torch.ceil(torch.clamp(torch.max(border_l), 0, w))) if border_l.shape[0] != 0 else 0
    r = int(torch.floor(torch.clamp(torch.min(border_r), 0, w)))if border_r.shape[0] != 0 else 0
    return t, b, l, r

@torch.no_grad()
def compute_warp_maps(seq_io, warper, compensate_poses, post_process=False):
    # compute all warp maps
    batch_begin = 0
    warp_maps = []
    ds = []
    W, H = seq_io.origin_width, seq_io.origin_height
    w, h = seq_io.width, seq_io.height
    crop_t, crop_b, crop_l, crop_r = 0, H, 0, W
    
    # post processing
    if post_process:
        smooth_depth = get_smooth_depth_kernel().to(device)

    while batch_begin < len(seq_io):
    
        batch_end = min(len(seq_io), batch_begin + seq_io.batch_size)
        segment = list(range(batch_begin, batch_end))
        depths = seq_io.load_depths(segment).to(device)

        if post_process:
            # load error maps
            error_maps = seq_io.load_errors(segment)
            thresh = 0.5
            error_maps[error_maps > thresh] = 1
            error_maps[error_maps < thresh] = 0
           
            # remove the noise in error map
            for i in range(error_maps.shape[0]):
                eroded_map = np.expand_dims(binary_erosion(error_maps[i].squeeze(0), iterations=1), 0)
                error_maps[i] = binary_dilation(eroded_map, iterations=8)
            
            # spatial-variant smoother according to the error map
            softmasks = scipy_gaussian(error_maps, sigma=[0, 0, 7, 7])
            softmasks = torch.from_numpy(softmasks).float().to(device)

            smoothed_depths = smooth_depth(depths) #smooth_depths(depths)
            depths = depths * (1 - softmasks) + smoothed_depths * softmasks

        # compute warping fields
        batch_warps, _, _, _, _ = warper.project_pixel(depths, compensate_poses[batch_begin:batch_end])
        batch_warps = (batch_warps + 1) / 2

        batch_warps[..., 0] *= (W - 1) 
        batch_warps[..., 1] *= (H - 1)
        t, b, l, r = get_cropping_area(batch_warps, H, W)
        crop_t = max(crop_t, t); crop_b = min(crop_b, b); crop_l = max(crop_l, l); crop_r = min(crop_r, r)
        
        batch_warps[..., 0] *= (w - 1) / (W - 1)
        batch_warps[..., 1] *= (h - 1) / (H - 1)

        inverse_warps = warper.inverse_flow(batch_warps)
        inverse_warps[..., 0] = inverse_warps[..., 0] * 2 / (w - 1) - 1
        inverse_warps[..., 1] = inverse_warps[..., 1] * 2 / (h - 1) - 1
        
        warp_maps.append(inverse_warps.detach().cpu())

        batch_begin = batch_end
    
    warp_maps = torch.cat(warp_maps, 0)
    return warp_maps, (crop_t, crop_b, crop_l, crop_r)

@torch.no_grad()
def run(opt):
    seq_io = SequenceIO(opt, preprocess=False)
    warper = Warper(opt, seq_io.get_intrinsic(True)).to(device)
    
    print('=> load camera trajectory')
    poses = seq_io.load_poses()
    _, comp = smooth_trajectory(poses, opt)
    compensate_poses = torch.from_numpy(comp).float().to(device)
    
    # compute all warping maps
    warp_maps, (crop_t, crop_b, crop_l, crop_r) = compute_warp_maps(seq_io, warper, compensate_poses, opt.post_process)
    crop_w, crop_h = crop_r - crop_l, crop_b - crop_t
    H, W = seq_io.origin_height, seq_io.origin_width

    # create video writer with crop size
    seq_io.create_video_writer((crop_w, crop_h))

    # forward warping
    print('=> warping frames')
    batch_begin = 0
    seq_io.need_resize = False
    while batch_begin < len(seq_io):
        # warp frames parallelly
        batch_end = min(len(seq_io), batch_begin + seq_io.batch_size)
        imgs = seq_io.load_snippet(batch_begin, batch_end)['imgs'].to(device)
        
        warp = F.interpolate(warp_maps[batch_begin:batch_end].to(device).permute(0, 3, 1, 2),
                        (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        reproj_imgs = F.grid_sample(imgs, warp)
        seq_io.write_images(reproj_imgs[..., crop_t:crop_b, crop_l:crop_r])
        batch_begin = batch_end
    
    print('=> Done!')

if __name__ == '__main__':
    opt = options.Options().parse()
    global device
    device = torch.device(opt.cuda)
    
    run(opt)

