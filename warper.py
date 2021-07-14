# Most from https://github.com/JiawangBian/SC-SfMLearner-Release/blob/master/inverse_warp.py

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2 as cv
import sys

class Warper(nn.Module):
    def __init__(self, opt, intrinsic):
        super(Warper, self).__init__()
        self.height = opt.height
        self.width = opt.width
        self.depth_min = opt.min_depth
        self.register_buffer('intrinsic', intrinsic.unsqueeze(0))
        self.register_buffer('intrinsic_inv', intrinsic.double().inverse().float())
        self.set_id_grid()

        self.padding_mode = 'zeros'

    def project_pixel(self, depth, pose):
        bs = depth.size(0)
        cam_coords = self.cam_coords * depth.unsqueeze(1)

        proj_cam_to_src_pixel = self.intrinsic.expand(bs, 3, 3) @ pose[:, :3]
        R = proj_cam_to_src_pixel[:, :, :3]
        t = proj_cam_to_src_pixel[:, :, -1:]
        
        src_pixel_coords, computed_depth, project_3d = self.cam2pixel(cam_coords, R, t)

        valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
        mask = valid_points.unsqueeze(1).float()
        return src_pixel_coords, mask, computed_depth, cam_coords.squeeze(1), project_3d

    def inverse_warp(self, img, depth, ref_depth, pose):
        src_pixel_coords, mask, computed_depth, pt1, pt2 = self.project_pixel(depth, pose)
        projected_img = F.grid_sample(img, src_pixel_coords, 
                                      padding_mode=self.padding_mode)
        
        if ref_depth is not None:
            projected_depth = F.grid_sample(ref_depth, src_pixel_coords,
                padding_mode=self.padding_mode).clamp(min=self.depth_min)
        else:
            projected_depth = None
        return projected_img, mask, projected_depth, computed_depth, src_pixel_coords, \
               pt1, pt2

    def inverse_flow(self, forward_flows):
        # inverse optical flow: given a forward flow, return the backward flow

        bs, h, w, _ = forward_flows.size()
        
        x = forward_flows[..., 0].view(bs, -1)
        y = forward_flows[..., 1].view(bs, -1)
        l = torch.floor(x); r = l + 1
        t = torch.floor(y); b = t + 1
        mask = (l >= 0) * (t >= 0) * (r < w) * (b < h)
        l *= mask; r *= mask; t *= mask; b *= mask
        x *= mask; y *= mask
        w_rb = torch.abs(x - l + 1e-3) * torch.abs(y - t + 1e-3)
        w_rt = torch.abs(x - l + 1e-3) * torch.abs(b - y + 1e-3)
        w_lb = torch.abs(r - x + 1e-3) * torch.abs(y - t + 1e-3)
        w_lt = torch.abs(r - x + 1e-3) * torch.abs(b - y + 1e-3)
        l = l.long(); r = r.long(); t = t.long(); b = b.long()

        weight_maps = torch.zeros(bs, h, w).to(forward_flows.device).double()
        grid_x = self.pixel_map[..., 0].view(-1).long()
        grid_y = self.pixel_map[..., 1].view(-1).long()

        for i in range(bs):
            for j in self.idx_set:
                weight_maps[i, t[i, j], l[i, j]] += w_lt[i, j]
                weight_maps[i, t[i, j], r[i, j]] += w_rt[i, j]
                weight_maps[i, b[i, j], l[i, j]] += w_lb[i, j]
                weight_maps[i, b[i, j], r[i, j]] += w_rb[i, j]


        forward_shifts = (-forward_flows + self.pixel_map.repeat(bs, 1, 1, 1)).double()
        backward_flows = torch.zeros(forward_flows.size()).to(forward_shifts.device)
        for i in range(bs):
            for c in range(2):
                for j in self.idx_set:
                    backward_flows[i, t[i, j], l[i, j], c] += \
                        forward_shifts[i, :, :, c].view(-1)[j] * w_lt[i, j]
                    backward_flows[i, t[i, j], r[i, j], c] += \
                        forward_shifts[i, :, :, c].view(-1)[j] * w_rt[i, j]
                    backward_flows[i, b[i, j], l[i, j], c] += \
                        forward_shifts[i, :, :, c].view(-1)[j] * w_lb[i, j]
                    backward_flows[i, b[i, j], r[i, j], c] += \
                        forward_shifts[i, :, :, c].view(-1)[j] * w_rb[i, j]
        for c in range(2):
            backward_flows[..., c] /= weight_maps

        backward_flows[torch.isinf(backward_flows)] = 0
        backward_flows[torch.isnan(backward_flows)] = 0
        backward_flows += self.pixel_map.repeat(bs, 1, 1, 1)

        backward_flows[weight_maps == 0] = -2

        return backward_flows

    def cam2pixel(self, cam_coords, R, t):
        bs = cam_coords.size(0)
        h, w = self.height, self.width
        cam_coords_flat= cam_coords.reshape(bs, 3, -1)
        pcoords = R @ cam_coords_flat + t

        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=self.depth_min)

        X_norm = 2*(X / Z) / (w - 1) - 1
        Y_norm = 2*(Y / Z) / (h - 1) - 1
        if self.padding_mode == 'zeros':
            X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
            Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2) #[B, H*W, 2]
        return pixel_coords.reshape(bs, h, w, 2), Z.reshape(bs, 1, h, w), \
               pcoords.reshape(bs, 3, h, w)

    def set_id_grid(self):
        h, w = self.height , self.width
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).float()
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).float()
        ones = torch.ones(1, h, w).float()

        self.pixel_coords = torch.stack((j_range, i_range, ones), dim=1).reshape(1, 3, -1)
        cam_coords = self.intrinsic_inv @ self.pixel_coords
        cam_coords = cam_coords.reshape(1, 3, h, w)
        self.register_buffer('cam_coords', cam_coords)

        pixel_map = torch.cat((j_range, i_range), 0).unsqueeze(0)
        self.register_buffer('pixel_map', pixel_map.permute(0, 2, 3, 1))

        sw, sh = 3, 3
        idxs = {i*sw+j: [] for i in range(sh) for j in range(sw)}
        for i in range(h):
            for j in range(w):
                key = ((i % sh) * sw) + j % sw
                idxs[key].append(i * w + j)
        self.idx_set = [torch.Tensor(v).long() for v in idxs.values()]

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    
    bot = (transform_mat[:, -1, :].detach() * 0.).view(-1, 1, 4)
    bot[:, :, -1] += 1.
    transform_mat = torch.cat([transform_mat, bot], dim=1)

    return transform_mat

def inverse_pose(pose_mat):
    
    R = pose_mat[:, :3, :3]
    t = pose_mat[:, :3, 3:]

    R_T = torch.transpose(R, 1, 2)
    t_inv = -R_T @ t
    pose_inv = torch.cat([R_T, t_inv], dim=2)
    
    bot = (pose_inv[:, -1, :].detach() * 0.).view(-1, 1, 4)
    bot[:, :, -1] += 1.
    pose_inv = torch.cat([pose_inv, bot], dim=1)
    return pose_inv
