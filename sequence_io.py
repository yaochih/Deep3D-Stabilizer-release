import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from imageio import imread
from path import Path
import random, os, sys, glob, subprocess
from skimage.transform import resize as imresize
from skimage import color
import cv2 as cv

class SequenceIO(data.Dataset):
    def __init__(self, opt, preprocess=True): 
        self.opt = opt
        self.input_video = opt.video_path
        self.root = Path(opt.output_dir)/opt.name
        self.root.makedirs_p()
        self.batch_size = opt.batch_size
        self.mean = opt.img_mean
        self.std = opt.img_std

        if preprocess:
            self.extract_frames()
            self.generate_flows()

        self.load_video()
        self.load_intrinsic()

    def extract_frames(self):
        (self.root/'images').makedirs_p()
        os.system('ffmpeg -y -hide_banner -loglevel panic -i "{}" {}/%05d.png'.format(self.input_video, self.root/'images'))

    def load_video(self): 
        self.image_names = sorted(list(glob.glob(self.root/'images/*.png')))

        # get frame size
        sample_image = imread(self.image_names[0])
        self.origin_size = sample_image.shape[:2]
        self.origin_height, self.origin_width = self.origin_size
        self.height, self.width = self.opt.height, self.opt.width
        self.need_resize = True

        # get fps
        p = subprocess.check_output(['ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {}'.format(self.input_video)], shell=True)
        exec('self.fps = int({})'.format(p.decode('utf-8').rstrip('\n')))

    def get_intrinsic(self, resize=False):
        return self.intrinsic_res if resize else self.intrinsic

    def load_intrinsic(self):
        self.intrinsic = torch.FloatTensor([[500., 0, self.origin_width*0.5], [0, 500., self.origin_height*0.5], [0, 0, 1]])
        if self.need_resize:
            self.intrinsic_res = self.intrinsic.clone()
            self.intrinsic_res[0] *= (self.width / self.origin_width)
            self.intrinsic_res[1] *= (self.height / self.origin_height)

    def generate_flows(self):
        # run PWC in python2 to get optical flow
        print('=> preparing optical flow. it would take a while.')
        os.chdir('PWC')
        for i in self.opt.intervals:
            ret = os.system('python2 video_pwc.py ../outputs/{} {}'.format(self.opt.name, i))
            assert ret == 0, "Failed to run PWC-Net, please check the environment of PWC-Net"
        os.chdir('..')

    def load_flow_snippet(self, begin, end, interval):
        w, h, W, H = self.width, self.height, self.origin_width, self.origin_height
        
        self.flow_fwd, self.flow_bwd = {}, {}
        flows = np.stack([np.load(self.root/'flows'/str(interval)/'{:05d}.npy'.format(j)) for j in range(begin, end - interval)], 0)
        b, _, flow_h, flow_w = flows.shape[:-1]
        flows[..., 0] = flows[..., 0] / W * w
        flows[..., 1] = flows[..., 1] / H * h
        
        grid_x = torch.arange(0, w).view(1, 1, 1, w).expand(2, b, h, w).float()
        grid_y = torch.arange(0, h).view(1, 1, h, 1).expand(2, b, h, w).float()
        flows = torch.from_numpy(flows).float().transpose(1, 0)
        flows = flows.reshape(-1, flow_h, flow_w, 2).permute(0, 3, 1, 2)
        flows = F.interpolate(flows, (h, w), mode='area')
        flows = flows.permute(0, 2, 3, 1).reshape(2, b, h, w, 2)
        flows[..., 0] += grid_x
        flows[..., 1] += grid_y

        flows[..., 0] = 2 * (flows[..., 0] / (w - 1) - 0.5)
        flows[..., 1] = 2 * (flows[..., 1] / (h - 1) - 0.5)
        return flows

    def load_depth_files(self, index, size):
        depth_path = self.root/'depths/{:05}.npy'.format(index)
        depth = np.load(depth_path)
        return torch.from_numpy(depth).float()

    def __len__(self):
        return len(self.image_names)

    def load_snippet(self, begin, end, load_flow=False):
        items = {}
        items['imgs'] = torch.stack([self.load_image(i) for i in range(begin, end)], 0)
        if load_flow:
            for i in self.opt.intervals:
                flows = self.load_flow_snippet(begin, end, i)
                items[('flow_fwd', i)] = flows[0]
                items[('flow_bwd', i)] = flows[1]
        return items

    def create_video_writer(self, crop_size):
        print('=> The output video will be saved as {}'.format(self.root/'output.avi'))
        self.video_writer = cv.VideoWriter(self.root/'output.avi', cv.VideoWriter_fourcc(*'MJPG'), int(self.fps), crop_size)

    def write_images(self, imgs):
        # write torch.Tensor images into cv.VideoWriter
        imgs = ((imgs * self.std + self.mean) * 255.).detach().cpu().numpy()
        imgs = imgs.transpose(0, 2, 3, 1).astype(np.uint8)[..., ::-1]

        for i in range(imgs.shape[0]):
            self.video_writer.write(imgs[i])

    def load_image(self, index):
        img = imread(self.image_names[index]).astype(np.float32)
        if self.need_resize:
            img = imresize(img, (self.height, self.width))
        img = np.transpose(img, (2, 0, 1))
        tensor_img = (torch.from_numpy(img).float() / 255 - self.mean) / self.std
        return tensor_img

    def save_depths(self, depths, indices):
        (self.root/'depths').makedirs_p()
        for i, idx in enumerate(indices):
            np.save(self.root/'depths/{:05}.npy'.format(idx), depths[0][i].cpu().detach().numpy())

    def load_depths(self, indices):
        depths = []
        for idx in indices:
            depth = np.load(self.root/'depths/{:05}.npy'.format(idx))
            depths.append(depth)
        depths = np.stack(depths, axis=0)
        depths = torch.from_numpy(depths).float()
        return depths

    def save_errors(self, errors, indices):
        (self.root/'errors').makedirs_p()
        for i, idx in enumerate(indices):
            np.save(self.root/'errors/{:05}.npy'.format(idx), errors[i].cpu().detach().numpy())

    def load_errors(self, indices):
        errors = []
        for idx in indices:
            try:
                error = np.load(self.root/'errors/{:05}.npy'.format(idx))
            except:
                # the last frame has no corresponding error map
                error = np.zeros(errors[-1].shape)
            errors.append(error)
        errors = np.stack(errors, axis=0)
        return errors

    def load_poses(self):
        return np.load(self.root/'poses.npy')

    def save_poses(self, poses):
        np.save(self.root/'poses.npy', poses.numpy())
