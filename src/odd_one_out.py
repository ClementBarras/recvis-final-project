from i3d import I3D
from torch import nn
import torch.nn.functional as F
import torch


class O3N(nn.Module):
    def __init__(self, i3d, n_questions):
        super(O3N, self).__init__()
        self.i3d = i3d
        self.n_questions = n_questions
        self.fc1 = nn.Linear(n_questions*400, 128)
        self.fc2 = nn.Linear(128, n_questions)
        
    def forward(self, inputs):
        to_concat = []
        inputs = inputs.permute((1,0,2,3,4,5))
        for qst in inputs:
            out, out_logits = self.i3d(qst)
            to_concat.append(out)
        x = torch.cat(to_concat, 1)
        x = x.view(-1, self.n_questions*400)
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x))
        return output