from torch import nn, flatten
from torchvision import models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.base_rn = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.base_rn.bn1
        self.relu = self.base_rn.relu
        self.maxpool = self.base_rn.maxpool
        self.layer1 = self.base_rn.layer1
        self.layer2 = self.base_rn.layer2
        self.layer3 = self.base_rn.layer3
        self.layer4 = self.base_rn.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512, 1)
        self.sigm = nn.Sigmoid()
        self.last_conv = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        self.last_conv = x
        x = self.avgpool(x)
        x = flatten(x,1)
        x = self.fc(x)
        x = self.sigm(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding="same")
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding="valid")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="valid")
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="valid")
        self.fc1 = nn.Linear(3200, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
        self.last_conv = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        self.last_conv = x
        x = self.pool(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x