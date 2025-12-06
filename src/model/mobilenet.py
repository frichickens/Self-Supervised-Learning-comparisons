import torch
import torch.nn as nn


class MobileNetV1(nn.Module):
    def __init__(self, c_in, c_out):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, padding=1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(inp, inp, 3, stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                # Pointwise
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            # First layer: stride=1 instead of 2 (critical for 32x32!)
            conv_bn(c_in, 32, stride=1),        # 32×32×3 → 32×32×32

            conv_dw( 32,  64, 1),            # 32×32×32 → 32×32×64
            conv_dw( 64, 128, 2),            #          → 16×16×128
            conv_dw(128, 128, 1),            #          → 16×16×128
            conv_dw(128, 256, 2),            #          →  8×8×256
            conv_dw(256, 256, 1),            #          →  8×8×256
            conv_dw(256, 512, 2),            #          →  4×4×512

            # 5 repeated blocks with 512 channels
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),

            conv_dw(512, 1024, 2),           #                →  2×2×1024
            conv_dw(1024, 1024, 1),          #                →  2×2×1024
        )

        self.classifier = nn.Linear(1024, c_out)
        self.pool = nn.AdaptiveAvgPool2d(1)   # Works on any size → 1×1

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)              # → (B, 1024, 1, 1)
        x = x.view(x.size(0), -1)     # → (B, 1024)
        x = self.classifier(x)
        return x
