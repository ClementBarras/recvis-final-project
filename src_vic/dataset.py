from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import torch

basic_transform = transforms.Compose([transforms.Resize((224, 224))])

class Sampler(object):
    def __init__(self, sampling_strat='random', n_samples=6):
        self.sampling_strat = sampling_strat
        self.n_samples = n_samples
        
    def sample(self, frame_count, sampling_strat=None, boundaries = None, shuffle=False):
        if not sampling_strat:
            sampling_strat = self.sampling_strat
        if boundaries is None:
            frame_idx_list = list(range(1, frame_count+1))
        else:
            frame_idx_list = list(range(1+boundaries[0], 1+min(frame_count, boundaries[1])))
        idxs = []
        if sampling_strat == 'random':
            idxs = random.sample(frame_idx_list, self.n_samples)
            idxs = sorted(idxs)
        elif sampling_strat == 'consecutive':
            start_idx = random.sample(list(range(len(frame_idx_list)-self.n_samples)), 1)[0]
            idxs = [frame_idx_list[start_idx + j] for j in range(self.n_samples)]
        elif sampling_strat == 'constrained_consecutive':
            window_size = int(1.5 * self.n_samples)
            start_idx = random.sample(list(range(len(frame_idx_list)-window_size)), 1)[0]
            start_idx = list(range(len(frame_idx_list)-window_size))[-1]
            window_idxs = [frame_idx_list[start_idx + j] for j in range(window_size)]
            idxs = random.sample(window_idxs, self.n_samples)
            idxs = sorted(idxs)
        if shuffle:
            shuffled = random.sample(idxs, len(idxs))
            while shuffled == idxs:
                shuffled = random.sample(idxs, len(idxs))
            idxs = shuffled
        return idxs

        
class SupervisedDataset(Dataset):
    def __init__(self, root='../datasets/UCF101_frames', transform=basic_transform, train=True,
                 video_info_path='../datasets/ucfTrainTestlist/trainlist01.txt'):
        super(SupervisedDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train
        if train:
            self.video_list, self.labels = self._get_video_list(video_info_path)
        else:
            self.video_list = self._get_video_list(video_info_path)
    
    def __getitem__(self, index):
        chosen_video = self.video_list[index]
        vid_path = os.path.join(self.root, chosen_video)
        frame_count = len(os.listdir(vid_path))
        frames = self._extract_frames(vid_path, list(range(1, frame_count+1)))
        if self.train:
            to_return = (frames, self.labels[index])
        else:
            to_return = frames
        return to_return
        
    def __len__(self):
        return len(self.video_list)
    
    def _get_video_list(self, video_info_path):
        video_list = []
        labels = []
        if isinstance(video_info_path, str):
            video_info_path = [video_info_path]
        for path in video_info_path:
            assert os.path.exists(path), 'Cannot locate file {}.'.format(path)
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip('\n')
                    if self.train:
                        vid, class_id = line.split(' ')
                        labels.append(int(class_id))
                    else:
                        vid = line
                    vid = vid.split("/")[1]
                    vid = vid.split(".")[0]
                    vid += "/"
                    video_list.append(vid)
        if labels:
            to_return = (video_list, labels)
        else:
            to_return = video_list
        return to_return
    
    def _extract_frames(self, vid_path, idxs):
        frames = []
        for idx in idxs:
            path = os.path.join(vid_path, "frame_{}.jpg".format(idx))
            frame = Image.open(path)
            if self.transform:
                frame = basic_transform(frame)
            frames.append(transforms.ToTensor()(frame))
        return torch.stack(frames)
          


class ProxyTaskDataset(Dataset):
    def __init__(self, root='../datasets/UCF101_frames', transform=basic_transform, sampling="random",
                 video_info_path='../datasets/ucfTrainTestlist/trainlist01.txt', n_sequences=3, n_samples=20, use_flow = False, flow_winwidth = 100, method="multiple"):
        super(ProxyTaskDataset, self).__init__()
        self.n_sequences = n_sequences
        self.use_flow = use_flow
        self.flow_winwidth = flow_winwidth
        #self.n_questions = n_questions
        self.root = root
        self.transform = transform
        self.method = "multiple"
        self.sampling = sampling
        self.n_samples = n_samples
        self.sampler = Sampler(sampling_strat=sampling, n_samples=n_samples)
        if use_flow:
            self.video_list, self.frames_dict = self._get_video_list(video_info_path)
        else:
            self.video_list = self._get_video_list(video_info_path)
        
    def __getitem__(self, index):
        chosen_video = self.video_list[index]
        vid_path = os.path.join(self.root, chosen_video)
        boundaries = None
        if self.use_flow:
            boundaries = self.frames_dict[chosen_video[:-1]]
        frame_count = len(os.listdir(vid_path))
        epsilon =  np.random.rand() 
        if epsilon >= 0.5:
            target = 1
        else:
            target = 0
            
        if self.method == "single":
            extract_count = 64
            if frame_count > extract_count:
                start_idx = random.sample(list(range(frame_count-extract_count)), 1)[0]
                idxs = [start_idx + j + 1 for j in range(extract_count)]
            else:
                idxs = list(range(1, frame_count+1)) + [frame_count for i in range(extract_count-frame_count)]
            frames = self._extract_frames(vid_path, idxs)
            if target == 0:
                inv_idx = torch.arange(frames.size(0)-1, -1, -1).long()
                frames = frames[inv_idx]
                
        elif self.method == "multiple":
            sequences = []
            for seq in range(self.n_sequences):
                idxs = self.sampler.sample(frame_count)
                frames = self._extract_frames(vid_path, idxs)
                sequences.append(frames)
            if target == 0:
                for i, seq in enumerate(sequences):
                    inv_idx = torch.arange(self.n_samples-1, -1, -1).long()
                    sequences[i] = seq[inv_idx]
            sequences = torch.stack(sequences)
            frames = sequences.permute((0, 2, 1, 3, 4))
        return frames, target
        
    def __len__(self):
        return len(self.video_list)
    
    def _get_video_list(self, video_info_path):
        video_list = []
        if self.use_flow:
            frames_dict = {}
        if isinstance(video_info_path, str):
            video_info_path = [video_info_path]
        for path in video_info_path:
            assert os.path.exists(path), 'Cannot locate file {}.'.format(path)
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip('\n')
                    if self.use_flow:
                        try:
                            vid, class_id, start_frame = line.split(',')
                            start_frame = int(start_frame)
                        except ValueError:
                            vid, start_frame = line, 0
                        frames_dict[vid] = (start_frame, start_frame + self.flow_winwidth)
                    else:
                        try:
                            vid, class_id = line.split(' ')
                        except ValueError:
                            vid = line
                        vid = vid.split("/")[1]
                        vid = vid.split(".")[0]
                    vid += "/"
                    video_list.append(vid)
        if self.use_flow:
            return list(set(video_list)), frames_dict
        else:
            return list(set(video_list))
    
    def _extract_frames(self, vid_path, idxs):
        frames = []
        for idx in idxs:
           # try:
            path = os.path.join(vid_path, "frame_{}.jpg".format(idx))
            frame = Image.open(path)
            if self.transform:
                frame = basic_transform(frame)
            frames.append(transforms.ToTensor()(frame))
            #except:
                #print("Unable to read frame {}".format(path))
        return torch.stack(frames)
          
        
        
    
            