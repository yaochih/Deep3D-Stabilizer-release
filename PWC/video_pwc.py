import sys, os, glob
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
from tqdm import tqdm
import models
"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""

video_dir = sys.argv[1]
#print(os.path.join(video_frames_dir, '*.png'))
interval = int(sys.argv[2])
frame_names = sorted(list(glob.glob(os.path.join(video_dir, 'images/*.png'))))
#print(len(frame_names))
#video_frames_dir = os.path.split(video_frames_txt)[0]
#frame_names = open(video_frames_txt, 'r').readlines()
#frame_names = frame_names[len(frame_names)//2:]
#frame_names = [os.path.join(video_frames_dir, name.rstrip('\n')) for name in frame_names]


pwc_model_fn = './pwc_net.pth.tar';
net = models.pwc_dc_net(pwc_model_fn)
net = net.cuda()
net.eval()

def process_batch(im_all, interval):
    im_all = [im[:, :, :3] for im in im_all]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]

    H_ = 384#int(ceil(H/divisor) * divisor)
    W_ = 640#int(ceil(W/divisor) * divisor)
    for i in range(len(im_all)):
            im_all[i] = cv2.resize(im_all[i], (W_, H_))

    for _i, _inputs in enumerate(im_all):
            im_all[_i] = im_all[_i][:, :, ::-1]
            im_all[_i] = 1.0 * im_all[_i]/255.0
            
            im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
            im_all[_i] = torch.from_numpy(im_all[_i])
            im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
            im_all[_i] = im_all[_i].float()
        
    #im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)
    im_all_fwd, im_all_bwd = [], []
    for i in range(len(im_all) - interval):
        im_all_fwd.append(torch.cat([im_all[i], im_all[i + interval]], 1).cuda())
        im_all_bwd.append(torch.cat([im_all[i + interval], im_all[i]], 1).cuda())
    im_all_fwd = torch.autograd.Variable(torch.cat(im_all_fwd, 0).cuda(), volatile=True)
    im_all_bwd = torch.autograd.Variable(torch.cat(im_all_bwd, 0).cuda(), volatile=True)

    
    def pass_model(im_all):
        flo = net(im_all)
        flo = flo * 20.0
        flo = flo.cpu().data.numpy()

        # scale the flow back to the input size 
        flo = np.swapaxes(np.swapaxes(flo, 1, 2), 2, 3) # 
        #u_ = cv2.resize(flo[:,:,0],(W,H))
        #v_ = cv2.resize(flo[:,:,1],(W,H))
        flo[:,:,:,0] *= W/ float(W_)
        flo[:,:,:,1] *= H/ float(H_)
        #flo = np.dstack((u_,v_))
        return flo

    return pass_model(im_all_fwd), pass_model(im_all_bwd)

#flows_fwd, flows_bwd = [], []
output_dir = os.path.join(video_dir, 'flows', str(interval))
if not os.path.exists(output_dir): os.makedirs(output_dir)

batch_begin, batch_end = 0, 0
while batch_end < len(frame_names):
    batch_end = min(len(frame_names), batch_begin + 12)
    frames = [imread(frame_names[j]) for j in range(batch_begin, batch_end)]
    flow_fwd, flow_bwd = process_batch(frames, interval)
    flows = np.stack([flow_fwd, flow_bwd], 0)
    for j in range(batch_begin, batch_end - interval):
        np.save(os.path.join(output_dir, '%05d.npy' % j), flows[:, j - batch_begin])
    batch_begin = batch_end - interval

