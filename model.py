import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        self.inChannels = in_channels
        self.outChannels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding = kernel_size//2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
class EmotionCNN(nn.Module):
    def __init__(self, num_classes = 7):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128)
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x
    

        