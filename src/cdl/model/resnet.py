import torch
from torch import nn

from cdl.model.components.quant.identity import IdentityQuant


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        self.q_in = IdentityQuant()
        self.q_mid = IdentityQuant()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
            if stride != 1 or in_planes != planes
            else nn.Sequential()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q_in(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.q_mid(out)
        return self.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_blocks: list[int], num_classes: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.in_planes = 16
        self.layer1 = self.make_layer(16, num_blocks[0], 1)
        self.layer2 = self.make_layer(32, num_blocks[1], 2)
        self.layer3 = self.make_layer(64, num_blocks[2], 2)
        self.q_out = IdentityQuant()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = []
        for s in [stride] + [1] * (num_blocks - 1):
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.stem(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.q_out(x)
        return self.fc(self.avgpool(x).flatten(1))
