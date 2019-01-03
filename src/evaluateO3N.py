import argparse
import os
from utils import get_free_gpu
from dataset import ProxyTaskDataset
from torch.utils.data import DataLoader

import torch

from i3d import I3D
from odd_one_out import O3N



parser = argparse.ArgumentParser(description='Self supervised learning script')
parser.add_argument('--data', type=str, default='../datasets/UCF101_frames', metavar='D',
                    help="folder where data is located.")
parser.add_argument('--video-list-directory', type=str, default='../datasets/ucfTrainTestlist', metavar='SD',
                    help="directory where the video lists are stored")
parser.add_argument('--sampling', type=str, default='random', metavar='SA,',
                    help="Sampling strategy (random, consecutive or constrained consecutive).")
parser.add_argument('--n_questions', type=int, default=6, metavar='Q,',
                    help="Number of questions")
parser.add_argument('--n_samples', type=int, default=10, metavar='s,',
                    help="Number of samples ie. frames")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

validation_set = ProxyTaskDataset(root=args.data, video_info_path=os.path.join(args.video_list_directory, 'vallist1.txt'),                                                sampling=args.sampling, n_samples=args.n_samples, n_questions=args.n_questions)
val_loader = DataLoader(validation_set, batch_size=128, shuffle=False)

state_dict = torch.load(args.model)
i3d = I3D(num_classes=400)
model = O3N(i3d, n_questions=args.n_questions)
model.load_state_dict(state_dict)
for param in model.parameters():
    if param.requires_grad:
        param.requires_grad = False

if use_cuda:
    print('Using GPU')
    free_gpu_id = get_free_gpu()
    device = "cuda:{}".format(free_gpu_id)
    model.to(device)
else:
    print('Using CPU')

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    count = 0
    for data, target in val_loader:
        count += 1
        if use_cuda:
            data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print(count)
    validation_loss /= len(val_loader.dataset)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss

validation()
