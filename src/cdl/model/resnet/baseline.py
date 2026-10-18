import torch
import torch.nn.functional as F
from flashinfer.quantization import fp4_quantization
from torch import nn


@torch.library.custom_op("cdl::nvfp4_qdq", mutates_args=())
def nvfp4_qdq(
    x: torch.Tensor, global_scale: torch.Tensor, global_scale_inv: torch.Tensor
) -> torch.Tensor:
    major, minor = torch.cuda.get_device_capability(x.device)
    quant = fp4_quantization.get_fp4_quantization_module(f"{major}{minor}")
    packed, sf = quant.fp4_quantize_sm100(x, global_scale, is_sf_swizzled_layout=False)
    out = torch.empty_like(x)
    fp4_quantization.get_fp4_kv_dequantization_module().nvfp4_kv_dequant(
        packed, sf.reshape(x.shape[0], -1), global_scale_inv, out
    )
    return out


@nvfp4_qdq.register_fake
def _(
    x: torch.Tensor, global_scale: torch.Tensor, global_scale_inv: torch.Tensor
) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=torch.bfloat16)


def quantize_dequantize(theta: torch.Tensor) -> torch.Tensor:
    k = theta.shape[-1]
    with torch.no_grad():
        x = theta.reshape(-1, k).bfloat16()
        r = -k % 16
        if r:
            x = F.pad(x, (0, r))

        amax = x.float().abs().amax()
        global_scale = torch.where(
            amax > 0,
            amax.new_full((), 448.0 * 6.0) / amax,
            amax.new_full((), torch.finfo(torch.float32).max),
        ).reshape(1)
        out = nvfp4_qdq(x, global_scale, (1.0 / global_scale).float())

    out = out.reshape(*theta.shape[:-1], -1)
    dequantized_theta = out[..., :k].to(theta.dtype)
    return theta + (dequantized_theta - theta).detach()


class BaselineQuantForActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.shape
        flat = input.permute(0, 2, 3, 1).reshape(-1, c)
        out = quantize_dequantize(flat)
        return out.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class BaselineQuantForWeight(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quantize_dequantize(input.flatten(1)).view_as(input)


class BaselineConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.weight_quant = BaselineQuantForWeight()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def make_conv2d(*args, **kwargs) -> nn.Conv2d:
    conv = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    return conv


class BaselineBasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        self.q_in = BaselineQuantForActivation()
        self.q_mid = BaselineQuantForActivation()
        self.conv1 = BaselineConv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BaselineConv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = (
            nn.Sequential(
                BaselineConv2d(in_planes, planes, 1, stride=stride, bias=False),
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


class BaselineResNet(nn.Module):
    def __init__(self, num_blocks: list[int], num_classes: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.stem = make_conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.in_planes = 16
        self.layer1 = self.make_layer(16, num_blocks[0], 1)
        self.layer2 = self.make_layer(32, num_blocks[1], 2)
        self.layer3 = self.make_layer(64, num_blocks[2], 2)
        self.q_out = BaselineQuantForActivation()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = []
        for s in [stride] + [1] * (num_blocks - 1):
            blocks.append(BaselineBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.stem(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(self.avgpool(self.q_out(x)).flatten(1))
