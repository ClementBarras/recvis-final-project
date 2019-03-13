import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import SupervisedDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils import get_free_gpu
from PIL import Image
from IPython.display import clear_output

from models import AOT 
from i3d import I3D

model_name = 'model_rgb_kinetics'
model_path = '../models/model_rgb.pth'
FRAMES_PER_SEQUENCE = 25

datasets_path = "../datasets/"
video_path = os.path.join(datasets_path, "UCF101_frames")
features_path = os.path.join(datasets_path, "vic/precomputed_features",  model_name)

try:
    os.mkdir(features_path)
except:
    pass

use_cuda = True

i3d = I3D(num_classes=400, modality='rgb')
aot = AOT(i3d=i3d)
i3d.load_state_dict(torch.load(model_path))
#aot.load_state_dict(torch.load(model_path))
model = i3d

if use_cuda:
    print('Using GPU')
    free_gpu_id = get_free_gpu()
    device = "cuda:{}".format(free_gpu_id)
    model.to(device)
else:
    print('Using CPU')
    
for param in model.parameters():
    param.requires_grad = False
    
def extract_frames(vid_path, idxs):
    frames = []
    for idx in idxs:
        path = os.path.join(vid_path, "frame_{}.jpg".format(idx))
        frame = Image.open(path)
        frame = transforms.Resize((224, 224))(frame)
        frames.append(transforms.ToTensor()(frame))
    return torch.stack(frames)

video_list = [f.name for f in os.scandir(video_path) if f.is_dir()]

for v, video in enumerate(video_list):
    #clear_output()
    print("{}_({}/{})".format(video,v+1, len(video_list)))
    video_folder = os.path.join(video_path, video)
    res_folder = os.path.join(features_path, video)
    #print(res_folder)
    try:
        os.mkdir(res_folder)
    except:
        pass
    frame_list = [f.name for f in os.scandir(video_folder) if f.is_file()]
    frame_count = len(frame_list)
    n_sequences = frame_count//FRAMES_PER_SEQUENCE
    if n_sequences == 0:
        idxs = list(range(1, frame_count+1)) + [frame_count for i in range(FRAMES_PER_SEQUENCE-frame_count)]
        inputs = extract_frames(video_folder, idxs)[None,:,:,:,:]
        inputs = inputs.permute((0,2,1,3,4))
        model.forward(inputs.to(device))
        features = model.features.cpu().detach().numpy().ravel()
        file_name = "frames_{}_to_{}".format(start_idx, end_idx)
        np.savetxt(os.path.join(res_folder, file_name), features, fmt="%.5f")
    else:
        for seq in range(n_sequences):
            start_idx = seq*FRAMES_PER_SEQUENCE + 1
            end_idx = start_idx + FRAMES_PER_SEQUENCE - 1
            idxs = np.arange(start_idx, end_idx+1)
            inputs = extract_frames(video_folder, idxs)[None]
            #print(inputs.shape)
            inputs = inputs.permute((0,2,1,3,4))
            model.forward(inputs.to(device))
            features = model.features.cpu().detach().numpy().ravel()
            file_name = "frames_{}_to_{}".format(start_idx, end_idx)
            np.savetxt(os.path.join(res_folder, file_name), features, fmt="%.5f")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
