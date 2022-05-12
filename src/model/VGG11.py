import torch
from torch import nn
import torch.nn.functional as F

class VGG11(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(VGG11, self).__init__()
        self.dropout_p = dropout_p

        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1_1.weight, mode='fan_in', nonlinearity='relu')

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2_1.weight, mode='fan_in', nonlinearity='relu')

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv3_1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv3_2.weight, mode='fan_in', nonlinearity='relu')

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv4_1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv4_2.weight, mode='fan_in', nonlinearity='relu')

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv5_1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv5_2.weight, mode='fan_in', nonlinearity='relu')

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2_1(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.maxpool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.maxpool5(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_p, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout_p, training=self.training)
        x = self.fc3(x)
        return x
