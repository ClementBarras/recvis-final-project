import argparse
import cv2
import os
import sys
import json


def extract_frames_from_video(vid_path, out_dir):
    if not os.path.exists(vid_path):
        print("Video path {} does not exist.".format(vid_path))
        sys.exit()
    else:
        video = cv2.VideoCapture(vid_path)
        if not video.isOpened():
            print('Failed to open video : {}'.format(vid_path))
    nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video contains {} frames.'.format(nb_frames))
    idx = 1
    for i in range(nb_frames):
        found, frame = video.read()
        if found:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_name = os.path.splitext(os.path.basename(vid_path))[0]+'_f'+str(idx)+'.jpg'
            im_path = os.path.join(out_dir, im_name)
            cv2.imwrite(im_path, frame)
            idx += 1
        else:
            print('Unable to read frame.')
    video.release()
    return idx
        
def create_all_frames(root, splits_dir, out_dir):
    # Testing if directories exist
    if not os.path.exists(root):
        print('Root directory {} not found.'.format(root))
        sys.exit()
    elif not os.path.exists(splits_dir):
        print('Splits directory not found.')
        sys.exit()
    else:
        files = os.listdir(splits_dir) #get list of files in traintestsplit directory
        # Creating the output directories
        if not os.path.exists(out_dir):
            print("Creating folder {}".format(out_dir))
            os.mkdir(out_dir)
        try:
            os.mkdir(os.path.join(out_dir, 'training_set'))
            os.mkdir(os.path.join(out_dir, 'test_set'))
        except OSError:
            print('Training and test directories already exist')
        
        # Extracting the frames for the training set for each split
        train_lists = [f for f in files if f.startswith('train')]
        train_info = {}
        for f in train_lists:
            split_id = str(os.path.splitext(f)[0][-1])
            train_info[split_id] = []
            split_dir = os.path.join(out_dir, 'training_set', 'split'+split_id)
            try:
                print("Creating split directory {}".format(split_dir))
                os.mkdir(split_dir)
            except OSError:
                "Split directory already exists"
            print('Extracting frames for train split {}'.format(split_id))
            with open(os.path.join(splits_dir, f)) as f:
                lines = f.readlines()
                for line in lines:
                    vid, class_id = line.split(' ')
                    print('Processing video {}...'.format(vid))
                    vid_path = os.path.join(root, vid)
                    idx = extract_frames_from_video(vid_path, split_dir)
                    train_info[split_id].append((vid, idx, class_id))
        with open(os.path.join(out_dir, 'train_info.json'), 'w') as fp:
            json.dump(train_info, fp)
        # Extracting the frames for the test set for each split                             
        test_lists = [f for f in files if f.startswith('test')]
        test_info = {}
        for f in test_lists:
            split_id = str(os.path.splitext(f)[0][-1])
            test_info[split_id] = []
            split_dir = os.path.join(out_dir, 'test_set', 'split'+split_id)
            try:
                print("Creating split directory {}".format(split_dir))
                os.mkdir(split_dir)
            except OSError:
                "Split directory already exists"
            print('Extracting frames for test split {}'.format(split_id)) 
            with open(os.path.join(splits_dir, f)) as f:
                lines = f.readlines()
                for line in lines:
                    vid = line.strip('\n')
                    print('Processing video {}...'.format())
                    vid_path = os.path.join(root, vid)
                    idx = extract_frames_from_video(vid_path, split_dir)
                    test_info[split_id].append((vid, idx))
        with open(os.path.join(out_dir, 'test_info.json'), 'w') as fp:
            json.dump(test_info, fp)
        
       
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../datasets/UCF-101')
    parser.add_argument('--splits_dir', type=str, default='../datasets/ucfTrainTestlist')
    parser.add_argument('--out_dir', type=str, default='../datasets/UCF101_frames')
    args = parser.parse_args()
    root = args.root
    splits_dir = args.splits_dir
    out_dir = args.out_dir
    create_all_frames(root, splits_dir, out_dir)