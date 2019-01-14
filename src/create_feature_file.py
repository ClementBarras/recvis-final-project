import os
import numpy as np

directory = '../datasets/precomputed_features/random_new'
split1 = '../datasets/ucfTrainTestlist/trainlist01.txt'
classid = '../datasets/ucfTrainTestlist/classInd.txt'
test = True


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
                print(file)
features = np.stack(features) 

np.save('../datasets/supervised_data/random/features_train01', features)
np.save('../datasets/supervised_data/random/labels_train01', labels)
np.save('../datasets/supervised_data/random/video_id_train01', video_id)