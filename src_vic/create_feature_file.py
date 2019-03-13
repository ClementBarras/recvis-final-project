import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Creating feature file')
parser.add_argument('--output', type=str, default='../datasets/supervised_data', metavar='D',
                    help="folder where data will be stored.")
parser.add_argument('--data', type=str, default='../datasets/vic/precomputed_features', metavar='D',
                    help="folder where data is located.")
parser.add_argument('--subfolder', type=str, default='consecutive', metavar='D',
                    help="subfolder where data is located.")
parser.add_argument('--test', type=bool, default=False, metavar='D',
                    help="Whether to create train or test features")

args = parser.parse_args()

directory = os.path.join(args.data, args.subfolder)

classid = '../datasets/ucfTrainTestlist/classInd.txt'
test = args.test

if test:
    split1 = '../datasets/ucfTrainTestlist/testlist01.txt'
else:
    split1 = '../datasets/ucfTrainTestlist/trainlist01.txt'


features = []
labels = []

video_list = []
video_labels = []
video_id = []
id_dict = {}
with open(classid) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n') 
        class_id, class_name = line.split(' ')
        vid_id = int(class_id)-1     
        id_dict[class_name] = vid_id


with open(split1) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')   
        if test:
            class_name = line.split("/")[0]
            video_labels.append(id_dict[class_name])
            vid = line.split("/")[1]
            vid = vid.split(".")[0]
            video_list.append(vid)
        else:   
            vid, class_id = line.split(' ')
            video_labels.append(int(class_id)-1)     
            vid = vid.split("/")[1]
            vid = vid.split(".")[0]
            video_list.append(vid)
             
existing_folders = os.listdir(directory)
#print(video_list)
for i, vid in enumerate(video_list):
    if vid in existing_folders:
        folder_path = os.path.join(directory, vid)
        sequences = os.listdir(folder_path)
        for file in sequences:
            path = os.path.join(folder_path, file)
            try:
                with open(path) as f:
                    lines = f.readlines()
                    features.append([float(line.strip('\n')) for line in lines])
                    labels.append(video_labels[i])
                    video_id.append(i)
            except Exception:
                print('error {}'.format(file))
features = np.stack(features) 

dir_name = args.output
subfolder_name = os.path.join(dir_name, args.subfolder)
if not os.path.exists(subfolder_name):
    os.mkdir(subfolder_name)
if test:
    file_ext = 'test'
else:
    file_ext = 'train'

file_path = os.path.join(subfolder_name, 'features_{}.npy'.format(file_ext))

np.save(file_path, features)

np.save(os.path.join(subfolder_name,'labels_{}.npy'.format(file_ext)), labels)

np.save(os.path.join(subfolder_name,'video_id_{}.npy'.format(file_ext)), video_id)
