import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ProxyTaskDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils import get_free_gpu


# Training settings
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
parser.add_argument('--batch-size', type=int, default=8, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
#from data import data_transforms

#train_split_paths = ['trainlist1.txt', 'trainlist2.txt', 'trainlist3.txt']
#val_split_paths = ['vallist1.txt', 'vallist2.txt', 'vallist3.txt']

train_set = ProxyTaskDataset(root=args.data, video_info_path=os.path.join(args.video_list_directory, 'completetrainlist.txt'),                                                sampling=args.sampling, n_samples=args.n_samples, n_questions=args.n_questions)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

validation_set = ProxyTaskDataset(root=args.data, video_info_path=os.path.join(args.video_list_directory, 'completevallist.txt'),                                                sampling=args.sampling, n_samples=args.n_samples, n_questions=args.n_questions)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from i3d import I3D
from models import O3N
i3d = I3D(num_classes=400)
i3d.load_state_dict(torch.load('../models/model_rgb.pth'))
model = O3N(i3d, n_questions=args.n_questions)
#model.load_state_dict(torch.load('adam_constrained_sampling/model_3.pth'))
layers_to_freeze = [i3d.conv3d_1a_7x7, i3d.conv3d_2b_1x1, i3d.conv3d_2c_3x3, i3d.maxPool3d_3a_3x3, i3d.mixed_3b, i3d.mixed_3c,
        i3d.mixed_4b, i3d.mixed_4c, i3d.mixed_4d, i3d.mixed_4e, i3d.mixed_4f]
for layer in layers_to_freeze:
    for param in layer.parameters():
        param.requires_grad = False
        
if use_cuda:
    print('Using GPU')
    free_gpu_id = get_free_gpu()
    device = "cuda:{}".format(free_gpu_id)
    model.to(device)
else:
    print('Using CPU')

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters())

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.float().to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
        
            
def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss

#sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = np.sqrt(.1), patience = 2)
for epoch in range(1, args.epochs + 1):
    training_loss = train(epoch)
    #for param_group in optimizer.param_groups:
       # print(param_group['lr'])
    validation_loss =  validation()
    #sched.step(validation_loss, epoch)
    model_file = args.experiment + '/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '.')
            
            
