import torch
import torch.nn as nn

class block(nn.Module):

    def __init__(self, dim):
        super(block, self).__init__()
        self.res_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.batchN_1 = nn.BatchNorm2d(dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.res_1(x)
        out = self.batchN_1(out)
        out = self.activation(out)

        return out

# Example usage