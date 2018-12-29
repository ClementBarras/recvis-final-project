import cv2
import os
from torch.utils.data import Dataset
import numpy as np
import random

class Sampler(object):
    def __init__(self, sampling_strat='random', n_samples=6):
        self.sampling_strat = sampling_strat
        self.n_samples = n_samples
        
    def sample(self, frame_idx_list, shuffle=False):
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
    def __init__(self, root='../datasets/UCF-101', sampling='random',
                 video_info_path=['../datasets/ucfTrainTestlist/trainlist01.txt'], n_samples=6, n_questions=6):
        super(ProxyTaskDataset, self).__init__()
        self.n_samples = n_samples
        self.n_questions = n_questions
        self.root = root
        self.video_list = self._get_video_list(video_info_path)
        self.sampler = Sampler(sampling_strat=sampling, n_samples=n_samples)
        
    def __getitem__(self, index):
        chosen_video = self.video_list[index]
        vid_path = os.path.join(self.root, chosen_video)
        vid = Video(vid_path)
        vid.count_frames(verify_frames=True)
        frame_idx_list = vid.frame_idx_list
        questions = []
        # Correct sequences
        for qst in range(self.n_questions-1):
            idxs = self.sampler.sample(frame_idx_list)
            frames = vid.extract_frames(idxs)
            questions.append((frames, 1))
        #Incorrect sequence
        idxs = self.sampler.sample(frame_idx_list, shuffle=True)
        frames = vid.extract_frames(idxs)
        questions.append((frames, 0))
        random.shuffle(questions) # Randomize position of incorrect sequence
        vid.close()
        frames = np.array([qst[0] for qst in questions])
        frames = np.transpose(frames, (0, 4, 1, 2, 3))
        target = [qst[1] for qst in questions].index(1)
        return frames, target
        
    def __len__(self):
        return len(self.video_list)
    
    def _get_video_list(self, video_info_path):
        video_list = []
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
                    video_list.append(vid)
        return video_list
    
    
class Video(object):
    def __init__(self, vid_path):
        self.open(vid_path)
        
    def open(self, vid_path):
        assert os.path.exists(vid_path), 'Cannot locate {}'.format(vid_path)
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            self.cap = cap
            self.vid_path = vid_path
        else:
            raise IOError("Failed to open video : {}".format(vid_path))
        
    def count_frames(self, verify_frames=False):
        unverified_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not verify_frames:
            self.frame_count = unverified_frame_count
            self.frame_idx_list = list(range(unverified_frame_count))
        else:
            verified_frame_list = []
            for i in range(unverified_frame_count):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                if not self.cap.grab():
                    logging.warning('Frame {} corrupted in video {}'.format(i, vid_path))
                else:
                    verified_frame_list.append(i)
            self.frame_idx_list = verified_frame_list
            self.frame_count = len(verified_frame_list)
    
    def extract_frames(self, idxs):
        frames = []
        for idx in idxs:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.cap.read()
            assert res, "Unable to read frame {} in video {}".frame(idx, self.vid_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[0:224,0:224]
            frames.append(frame)
        return frames
            
    def close(self):
        self.cap.release()
        self.cap = None
                
            