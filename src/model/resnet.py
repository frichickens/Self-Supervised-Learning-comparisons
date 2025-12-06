import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_channels=3):  # Changed default to 10
        super().__init__()
        self.in_channels = 64

        # CHANGE 1: 7x7 -> 3x3, stride 2 -> 1
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # CHANGE 2: Remove maxpool entirely (too aggressive on 32x32)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layers = [3,4,6,3] for ResNet-50
        self.layer1 = self._make_layer(block, layers[0], planes=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], planes=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], planes=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, blocks, planes, stride=1):
        i_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * block.expansion:
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers.append(block(self.in_channels, planes, i_downsample=i_downsample, stride=stride))
        self.in_channels = planes * block.expansion

        for _ in range(blocks - 1):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Removed maxpool here

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# CHANGE 3 & 4 & 5: Default num_classes=10, channels=3
def ResNet50(c_in, c_out):
    """Factory: returns a ResNet-50 model for CIFAR-10/100."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=c_out, num_channels=c_in)