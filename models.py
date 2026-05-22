import torch
import torch.nn as nn
import copy

class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def average_weights(w_list):
    w_avg = copy.deepcopy(w_list[0])
    for key in w_avg.keys():
        for i in range(1, len(w_list)):
            w_avg[key] += w_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w_list))
    return w_avg