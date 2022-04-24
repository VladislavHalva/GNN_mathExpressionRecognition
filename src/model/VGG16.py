import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1_1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv1_2.weight, mode='fan_in', nonlinearity='relu')

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2_1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2_2.weight, mode='fan_in', nonlinearity='relu')

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv3_1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv3_2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv3_3.weight, mode='fan_in', nonlinearity='relu')

        # self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #
        # self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        # if not self.training:
        #     for comp in x.detach():
        #         fig = plt.figure(figsize=(8, 16))
        #         for i in range(comp.shape[0]):
        #             a = fig.add_subplot(8, 16, i + 1)
        #             imgplot = plt.imshow(comp[i])
        #             a.axis("off")
        #         plt.show()
        #     exit()
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        # x = F.relu(self.conv4_1(x))
        # x = F.relu(self.conv4_2(x))
        # x = F.relu(self.conv4_3(x))
        # x = self.maxpool(x)
        # x = F.relu(self.conv5_1(x))
        # x = F.relu(self.conv5_2(x))
        # x = F.relu(self.conv5_3(x))
        # x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x
