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
        
    def sample(self, frame_count, shuffle=False):
        frame_idx_list = list(range(1, frame_count+1))
        idxs = []
        if self.sampling_strat == 'random':
            idxs = random.sample(frame_idx_list, self.n_samples)
            idxs = sorted(idxs)
        elif self.sampling_strat == 'consecutive':
            start_idx = random.sample(list(range(len(frame_idx_list)-self.n_samples)), 1)[0]
            idxs = [frame_idx_list[start_idx + j] for j in range(self.n_samples)]
        elif self.sampling_strat == 'constrained_consecutive':
            window_size = int(1.5 * self.n_samples)
            start_idx = random.sample(list(range(len(frame_idx_list)-window_size)), 1)[0]
            window_idxs = [frame_idx_list[start_idx + j] for j in range(window_size)]
            idxs = random.sample(window_idxs, self.n_samples)
        if shuffle:
            shuffled = random.sample(idxs, len(idxs))
            while shuffled == idxs:
                shuffled = random.sample(idxs, len(idxs))
            idxs = shuffled
        return idxs


class ProxyTaskDataset(Dataset):
    def __init__(self, root='../datasets/UCF101_frames', sampling='random', transform = basic_transform,
                 video_info_path='../datasets/ucfTrainTestlist/trainlist01.txt', n_samples=6, n_questions=6):
        super(ProxyTaskDataset, self).__init__()
        self.n_samples = n_samples
        self.n_questions = n_questions
        self.root = root
        self.transform = transform
        self.video_list = self._get_video_list(video_info_path)
        self.sampler = Sampler(sampling_strat=sampling, n_samples=n_samples)
        
    def __getitem__(self, index):
        chosen_video = self.video_list[index]
        vid_path = os.path.join(self.root, chosen_video)
        frame_count = len(os.listdir(vid_path))
        questions = []
        # Correct sequences
        target = []
        for qst in range(self.n_questions-1):
            idxs = self.sampler.sample(frame_count)
            frames = self._extract_frames(vid_path, idxs)
            questions.append(frames)
            target.append(1)
        #Incorrect sequence
        idxs = self.sampler.sample(frame_count, shuffle=True)
        frames = self._extract_frames(vid_path, idxs)
        questions.append(frames)
        target.append(0)
        temp = list(zip(questions, target))
        random.shuffle(temp)
        questions, target = zip(*temp)
        target = target.index(0)
        questions = torch.stack(questions)
        frames = questions.permute((0, 2, 1, 3, 4))
        return frames, target
        
    def __len__(self):
        return len(self.video_list)
    
    def _get_video_list(self, video_info_path):
        video_list = []
        if isinstance(video_info_path, str):
            video_info_path = [video_info_path]
        for path in video_info_path:
            assert os.path.exists(path), 'Cannot locate file {}.'.format(path)
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip('\n')
                    try:
                        vid, class_id = line.split(' ')
                    except ValueError:
                        vid = line
                    vid = vid.split("/")[1]
                    vid = vid.split(".")[0]
                    vid += "/"
                    video_list.append(vid)
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
          
        
        
    
            