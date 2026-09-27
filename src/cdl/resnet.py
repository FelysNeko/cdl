from torch import nn

from cdl.model import CdlQuantForActivation, QConv2d, QLinear


class QBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        relaxed: bool,
        bits: int,
        topk_act: int,
    ):
        super().__init__()
        self.q_in = CdlQuantForActivation(bits, relaxed, topk_act)
        self.conv1 = QConv2d(
            in_planes,
            planes,
            3,
            stride=stride,
            padding=1,
            bias=False,
            w_bits=bits,
            relaxed=relaxed,
            quantize_act=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(
            planes,
            planes,
            3,
            padding=1,
            bias=False,
            w_bits=bits,
            relaxed=relaxed,
            quantize_act=True,
            topk_act=topk_act,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = (
            nn.Sequential(
                QConv2d(
                    in_planes,
                    planes,
                    1,
                    stride=stride,
                    bias=False,
                    w_bits=bits,
                    relaxed=relaxed,
                    quantize_act=False,
                ),
                nn.BatchNorm2d(planes),
            )
            if stride != 1 or in_planes != planes
            else nn.Sequential()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.q_in(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class QResNet(nn.Module):
    def __init__(
        self,
        num_blocks: list[int],
        num_classes: int,
        relaxed: bool,
        bits: int,
        bits_edge: int,
        topk_act: int,
    ):
        super().__init__()
        self.relaxed = relaxed
        self.bits = bits
        self.topk_act = topk_act
        self.relu = nn.ReLU(inplace=True)
        self.stem = QConv2d(
            3,
            16,
            3,
            padding=1,
            bias=False,
            w_bits=bits_edge,
            relaxed=relaxed,
            quantize_act=False,
        )
        self.bn = nn.BatchNorm2d(16)
        self.in_planes = 16
        self.layer1 = self.make_layer(16, num_blocks[0], 1)
        self.layer2 = self.make_layer(32, num_blocks[1], 2)
        self.layer3 = self.make_layer(64, num_blocks[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = QLinear(
            64,
            num_classes,
            w_bits=bits_edge,
            relaxed=relaxed,
            quantize_act=True,
            topk_act=topk_act,
            act_bits=bits,
        )

    def make_layer(self, planes, num_blocks, stride):
        blocks = []
        for s in [stride] + [1] * (num_blocks - 1):
            blocks.append(
                QBasicBlock(
                    self.in_planes, planes, s, self.relaxed, self.bits, self.topk_act
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.relu(self.bn(self.stem(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(self.avgpool(x).flatten(1))


def get_cifar_resnet(
    num_layers: int,
    num_classes: int = 100,
    relaxed: bool = False,
    bits: int = 6,
    bits_edge: int = 8,
    topk_act: int = 5,
) -> QResNet:
    n = (num_layers - 2) // 6
    return QResNet([n, n, n], num_classes, relaxed, bits, bits_edge, topk_act)
