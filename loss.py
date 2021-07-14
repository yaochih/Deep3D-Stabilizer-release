from __future__ import absolute_import, division, print_function
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warper import *

def gradient_x(img):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx

def gradient_y(img):
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy

def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum()
    return mean_value


class Loss(nn.Module):
    def __init__(self, opt, warper, dataset=None):
        super(Loss, self).__init__()
        
        self.opt = opt
        global device
        device = torch.device(opt.cuda)
        self.scales = opt.scales
        self.intervals = opt.intervals
        self.ssim_weight = opt.ssim_weight

        self.ssim = SSIM(window_size=3).to(device)
        self.warper = warper

        self.width = opt.width
        self.height = opt.height
        
        self.weights = {
            'photo':opt.photometric_loss,
            'flow': opt.flow_loss,
            'geo':  opt.geometry_loss
        }
        
    def preprocess_minibatch_weights(self, items):
        opt = self.opt
        
        self.bs = items['imgs'].size(0)
        self.interval_weights = {}
        self.adaptive_weights = {}
        
        # compute the weights for different source view
        alpha_sum = 0
        self.alpha = {}
        self.beta = {}
        for i in self.intervals:
            if i >= self.bs: continue
            self.alpha[i] = np.power(opt.adaptive_alpha, i)
            self.beta[i] = np.power(opt.adaptive_beta, i)

        alpha_sum = sum([self.alpha[k] for k in self.alpha.keys()])
        beta_sum = sum([self.beta[k] for k in self.beta.keys()])

        for k in self.alpha.keys():
            self.alpha[k] /= alpha_sum
            self.beta[k] /= beta_sum

    
    def compute_loss_terms(self, items):
        # compute the loss of a given snippet with multiple frame intervals
        bs = items['imgs'].size(0)
        loss_items = {}
        for key in self.weights.keys():
            loss_items[key] = 0
        
        poses = items['poses']
        poses_inv = items['poses_inv']
        
        for i in self.intervals:
            if i >= bs: continue
            pair_item = {'img1':    items['imgs'][:-i],
                         'img2':    items['imgs'][i:],
                         'depth1':  [depth[:-i] for depth in items['depths']],
                         'depth2':  [depth[i:] for depth in items['depths']],
                         'pose21':  poses_inv[:-i] @ poses[i:],
                         'pose12':  poses_inv[i:] @ poses[:-i],
                         'flow12':  items[('flow_fwd', i)],
                         'flow21':  items[('flow_bwd', i)]}
            pair_item['alpha'] = self.alpha[i]
            pair_item['beta'] = self.beta[i]
            pair_loss, err_mask = self.compute_pairwise_loss(pair_item)

            for name in loss_items.keys():
                if name not in pair_loss.keys(): continue
                loss_items[name] += pair_loss[name] 
            try:
                m = err_mask.size(0)
                n = items['err_mask'].size(0)
                items['err_mask'][:m-n] += err_mask
            except Exception as e:
                items['err_mask'] = err_mask

        return loss_items


    def compute_pairwise_loss(self, item):
        # compute the loss a given snippet with a frame interval
        img1, img2 = item['img1'], item['img2']
        pose12, pose21 = item['pose12'], item['pose21']
        input_flow12 = item['flow12'].permute(0, 3, 1, 2)
        input_flow21 = item['flow21'].permute(0, 3, 1, 2)

        bs = img1.size(0)
        loss_items = {}
        for key in self.weights.keys():
            loss_items[key] = 0
        
        for scale in self.scales:
            depth1_scaled = item['depth1'][scale]
            depth2_scaled = item['depth2'][scale]

            ret1 = self.warper.inverse_warp(img2, depth1_scaled, depth2_scaled, pose12)
            ret2 = self.warper.inverse_warp(img1, depth2_scaled, depth1_scaled, pose21)

            rec1, mask1, projected_depth1, computed_depth1, warp_sample1, pt1, pt12 = ret1
            rec2, mask2, projected_depth2, computed_depth2, warp_sample2, pt2, pt21 = ret2

            # geometry loss
            diff_depth1 = ((computed_depth1 - projected_depth1).abs() /
                           (computed_depth1 + projected_depth1).abs()).clamp(0, 1)
            diff_depth2 = ((computed_depth2 - projected_depth2).abs() /
                           (computed_depth2 + projected_depth2).abs()).clamp(0, 1)
            
            diff_depth1 *= item['alpha']
            diff_depth2 *= item['alpha']

            loss_items['geo'] += mean_on_mask(diff_depth1, mask1)
            loss_items['geo'] += mean_on_mask(diff_depth2, mask2)
            
            weight_mask1 = (1 - diff_depth1) * mask1
            weight_mask2 = (1 - diff_depth2) * mask2

            # photometric loss
            diff_img1 = (img1 - rec1).abs()
            diff_img2 = (img2 - rec2).abs()
            if self.ssim_weight > 0:
                ssim_map1 = self.ssim(img1, rec1)
                ssim_map2 = self.ssim(img2, rec2)
                diff_img1 = (1-self.ssim_weight)*diff_img1 + self.ssim_weight*ssim_map1
                diff_img2 = (1-self.ssim_weight)*diff_img2 + self.ssim_weight*ssim_map2

            loss_items['photo'] += mean_on_mask(diff_img1 * item['alpha'], mask1)
            loss_items['photo'] += mean_on_mask(diff_img2 * item['alpha'], mask2)
            
            warp_flow1 = warp_sample1.permute(0, 3, 1, 2)
            warp_flow2 = warp_sample2.permute(0, 3, 1, 2)

            # flow
            diff_flow1 = (warp_flow1 - input_flow12).abs().sum(1, keepdim=True)
            diff_flow2 = (warp_flow2 - input_flow21).abs().sum(1, keepdim=True)

            diff_flow1 *= item['beta']
            diff_flow2 *= item['beta']
            loss_items['flow'] += mean_on_mask(diff_flow1, mask1)
            loss_items['flow'] += mean_on_mask(diff_flow2, mask2)
            
            # return error mask for post-processing
            err_mask = torch.abs(diff_img1.mean(1, keepdim=True)) * mask1

        return loss_items, err_mask


    def forward(self, items):
        bs = items['imgs'].size(0)
        loss_items = self.compute_loss_terms(items)

        loss_items['full'] = 0
        for key in self.weights.keys():
            loss_items['full'] += self.weights[key] * loss_items[key]
        
        return loss_items

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, window_size=3, alpha=1, beta=1, gamma=1):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(window_size, 1)
        self.mu_y_pool   = nn.AvgPool2d(window_size, 1)
        self.sig_x_pool  = nn.AvgPool2d(window_size, 1)
        self.sig_y_pool  = nn.AvgPool2d(window_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(window_size, 1)

        self.refl = nn.ReflectionPad2d(window_size//2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.C3 = self.C2 / 2
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if alpha == 1 and beta == 1 and gamma == 1:
            self.run_compute = self.compute_simplified
        else:
            self.run_compute = self.compute
        

    def compute(self, x, y):
        
        x = self.refl(x)
        y = self.refl(y)
        
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        l = (2 * mu_x * mu_y + self.C1) / \
            (mu_x * mu_x + mu_y * mu_y + self.C1)
        c = (2 * sigma_x * sigma_y + self.C2) / \
            (sigma_x + sigma_y + self.C2)
        s = (sigma_xy + self.C3) / \
            (torch.sqrt(sigma_x * sigma_y) + self.C3)

        ssim_xy = torch.pow(l, self.alpha) * \
                  torch.pow(c, self.beta) * \
                  torch.pow(s, self.gamma)
        return torch.clamp((1 - ssim_xy) / 2, 0, 1)

    def compute_simplified(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

    def forward(self, x, y):
        return self.run_compute(x, y)


