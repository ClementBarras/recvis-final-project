from i3d import I3D
from torch import nn
import torch.nn.functional as F
import torch


class O3N(nn.Module):
    def __init__(self, i3d, n_questions):
        super(O3N, self).__init__()
        self.i3d = i3d
        self.n_questions = n_questions
        self.fc1 = nn.Linear(n_questions*1024, 128)
        self.fc2 = nn.Linear(128, n_questions)
        
    def forward(self, inputs):
        to_concat = []
        inputs = inputs.permute((1,0,2,3,4,5))
        for qst in inputs:
            out, out_logits = self.i3d(qst)
            features = self.i3d.features
            to_concat.append(features)
        x = torch.cat(to_concat, 1)
        x = x.view(-1, self.n_questions*1024)
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=0)
        return output

class SupervisedModel(nn.Module):
    def __init__(self, i3d, n_samples=10, n_classes=101):
        super(SupervisedModel, self).__init__()
        self.i3d = i3d
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, inputs):
        inputs = inputs.permute((0,2,1,3,4))
        probas = []
        n_frames = inputs.size()[2]
        n_sequences = n_frames // self.n_samples
        print(n_sequences)
        for seq in range(n_sequences):
            frames = inputs[:, :, seq*self.n_samples:(seq+1)*self.n_samples, :, :]
            out, out_logits = self.i3d(frames)
            features = self.i3d.features
            x = F.relu(self.fc1(features))
            x = F.softmax(self.fc2(x), dim=0)
            probas.append(x)
        x = torch.cat(probas, 0)
        x = torch.sum(x, dim=0, keepdim=True)
        output = F.softmax(x, dim=0)
        return output
    