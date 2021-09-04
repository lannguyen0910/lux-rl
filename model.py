import torch.nn as nn
import torch


def single_conv5(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
        nn.ReLU()
    )


def single_conv3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
        nn.ReLU()
    )


def single_conv2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 2),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
        nn.ReLU()
    )


class Actor(nn.Module):
    def __init__(self, in_size, out_size, seed):

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.maxpool = nn.MaxPool2d(2)

        self.layer1 = single_conv3(in_size, 16)

        self.layer2_1 = single_conv5(16, 32)
        self.layer2_2 = single_conv5(32, 32)
        self.layer2_3 = single_conv3(32, 32)

        self.layer3_1 = single_conv3(32, 32)
        self.layer3_2 = single_conv3(32, 64)

        self.layer4 = single_conv5(64, 128)

        self.fc1 = nn.Sequential(
            nn.Linear(128*12*12, 64),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(64, out_size)

    def forward(self, x1):
        x1 = self.layer1(x1)

        x1 = self.layer2_1(x1)
        #print(f'conv1: {x1.data.cpu().numpy().shape}')

        x1 = self.layer2_2(x1)
        #print(f'conv2: {x1.data.cpu().numpy().shape}')

        x1 = self.layer2_3(x1)
        x1 = self.layer3_1(x1)
        #print(f'conv3: {x1.data.cpu().numpy().shape}')
        x1 = self.layer3_2(x1)
        x1 = self.layer4(x1)
        #print(f'conv4: {x1.data.cpu().numpy().shape}')

        x1 = x1.view(-1, 128*12*12)

        x = self.fc1(x1)
        #print(f'conv6: {x.data.cpu().numpy().shape}')
        out = self.fc2(x)

        return out
