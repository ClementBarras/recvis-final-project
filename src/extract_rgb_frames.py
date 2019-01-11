import argparse
import cv2
import os
import sys
import json


def extract_rgb_frames_from_video(vid_path, out_dir):
    if not os.path.exists(vid_path):
        print("Video path {} does not exist.".format(vid_path))
        sys.exit()
    else:
        video = cv2.VideoCapture(vid_path)
        if not video.isOpened():
            print('Failed to open video : {}'.format(vid_path))
    nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video contains {} frames.'.format(nb_frames))
    video_dir_name = os.path.splitext(os.path.basename(vid_path))[0]
    if not os.path.exists(os.path.join(out_dir, video_dir_name)):
        os.mkdir(os.path.join(out_dir, video_dir_name))
    idx = 1
    for i in range(nb_frames):
        found, frame = video.read()
        if found:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #im_name = os.path.splitext(os.path.basename(vid_path))[0]+'_f'+str(idx)+'.jpg'
            im_name = 'frame_' + str(idx) + '.jpg'
            im_path = os.path.join(out_dir, video_dir_name, im_name)
            cv2.imwrite(im_path, frame)
            idx += 1
        else:
            print('Unable to read frame.')
    video.release()
    return idx
        
def create_all_frames(root, out_dir):
    # Testing if directories exist
    if not os.path.exists(root):
        print('Root directory {} not found.'.format(root))
        sys.exit()
    else:
        # Creating the output directories
        if not os.path.exists(out_dir):
            print("Creating folder {}".format(out_dir))
            os.mkdir(out_dir)

        # Extracting the frames for the training set for each split
        subdir_list = os.listdir(root)
        for subdir in subdir_list:
            file_list = os.listdir(os.path.join(root, subdir))
            for f in file_list:
                print('Processing video {}...'.format(f))
                vid_path = os.path.join(root, subdir, f)
                idx = extract_rgb_frames_from_video(vid_path, out_dir)
       
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../datasets/UCF-101')
    parser.add_argument('--out_dir', type=str, default='../datasets/UCF101_frames')
    args = parser.parse_args()
    root = args.root
    out_dir = args.out_dir
    create_all_frames(root, out_dir)