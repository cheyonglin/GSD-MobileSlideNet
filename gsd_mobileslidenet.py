import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8):
    return int((v + divisor / 2) // divisor * divisor)


# =========================================================
# Conv
# =========================================================
class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, g=1, act=True):
        p = (k - 1) // 2

        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                groups=g,
                bias=False
            ),
            nn.BatchNorm2d(out_ch)
        ]

        if act:
            layers.append(nn.Hardswish(inplace=True))

        super().__init__(*layers)


# =========================================================
# ECA Attention
# =========================================================
class ECALayer(nn.Module):
    def __init__(self, ch, k_size=3):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.pool(x)

        y = self.conv(
            y.squeeze(-1).transpose(-1, -2)
        )

        y = self.sigmoid(
            y.transpose(-1, -2).unsqueeze(-1)
        )

        return x * y


# =========================================================
# LMSDBC Block
# =========================================================
class LMSDBC(nn.Module):

    def __init__(self, ch, dilation=2):
        super().__init__()

        hidden = ch // 3

        self.reduce = ConvBNAct(ch, hidden, 1)

        self.dw = ConvBNAct(
            hidden,
            hidden,
            3,
            g=hidden
        )

        self.dw_dilated = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=hidden,
                bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True)
        )

        self.eca = ECALayer(hidden)

        self.expand = ConvBNAct(
            hidden,
            ch,
            1,
            act=False
        )

    def forward(self, x):

        out = self.reduce(x)

        out = self.dw(out) + self.dw_dilated(out)

        out = self.eca(out)

        out = self.expand(out)

        return out + x


# =========================================================
# Lite InvertedResidual
# =========================================================
class InvertedResidual(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        stride,
        expand_ratio=3
    ):
        super().__init__()

        hidden = int(in_ch * expand_ratio)

        self.use_res = (
            stride == 1 and in_ch == out_ch
        )

        self.block = nn.Sequential(
            ConvBNAct(in_ch, hidden, 1),

            ConvBNAct(
                hidden,
                hidden,
                3,
                stride,
                g=hidden
            ),

            ConvBNAct(
                hidden,
                out_ch,
                1,
                act=False
            ),
        )

    def forward(self, x):

        out = self.block(x)

        if self.use_res:
            out = out + x

        return out


# =========================================================
# GSD Head
# =========================================================
class Head(nn.Module):

    def __init__(self, in_ch, num_classes):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten(1)

        self.feat = nn.Linear(in_ch, 128)

        self.cls = nn.Linear(in_ch, num_classes)

    def forward(self, x):

        x = self.pool(x)

        x = self.flatten(x)

        feat = self.feat(x)

        cls = self.cls(x)

        return feat, cls


# =========================================================
# EMA Teacher
# =========================================================
class EMA:

    def __init__(self, model, decay=0.999):

        self.model = model

        self.decay = decay

        self.shadow = {}

        for name, param in model.named_parameters():

            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):

        for name, param in self.model.named_parameters():

            if param.requires_grad:

                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1 - self.decay) * param.data
                )

    def apply_shadow(self):

        for name, param in self.model.named_parameters():

            if param.requires_grad:
                param.data = self.shadow[name]


# =========================================================
# GSD-MobileSlideNet
# =========================================================
class GSD_MobileSlideNet(nn.Module):

    def __init__(
        self,
        num_classes=3,
        width_mult=0.75
    ):
        super().__init__()

        c16 = _make_divisible(16 * width_mult)
        c24 = _make_divisible(24 * width_mult)
        c40 = _make_divisible(40 * width_mult)
        c80 = _make_divisible(80 * width_mult)
        c112 = _make_divisible(112 * width_mult)
        c160 = _make_divisible(160 * width_mult)

        # Stem
        self.stem = ConvBNAct(
            3,
            c16,
            3,
            2
        )

        # Stage1
        self.stage1 = nn.Sequential(
            InvertedResidual(c16, c24, 2, 3),
            InvertedResidual(c24, c24, 1, 2),
        )

        # Stage2
        self.stage2 = nn.Sequential(
            InvertedResidual(c24, c40, 2, 3),
            InvertedResidual(c40, c40, 1, 2),
        )

        # Stage3
        self.stage3 = nn.Sequential(
            ConvBNAct(c40, c80, 1),
            LMSDBC(c80, 2),
            LMSDBC(c80, 4),
        )

        # Stage4
        self.stage4 = nn.Sequential(
            ConvBNAct(c80, c112, 2),
            LMSDBC(c112, 2),
        )

        # Stage5
        self.stage5 = nn.Sequential(
            InvertedResidual(c112, c160, 2, 3),
        )

        # Multi-stage Heads
        self.h1 = Head(c24, num_classes)
        self.h2 = Head(c40, num_classes)
        self.h3 = Head(c80, num_classes)
        self.h4 = Head(c112, num_classes)
        self.h5 = Head(c160, num_classes)

        self.w = nn.Parameter(torch.ones(5))

    def forward(self, x):

        feats = []
        logits = []

        x = self.stem(x)

        x = self.stage1(x)
        f1, l1 = self.h1(x)
        feats.append(f1)
        logits.append(l1)

        x = self.stage2(x)
        f2, l2 = self.h2(x)
        feats.append(f2)
        logits.append(l2)

        x = self.stage3(x)
        f3, l3 = self.h3(x)
        feats.append(f3)
        logits.append(l3)

        x = self.stage4(x)
        f4, l4 = self.h4(x)
        feats.append(f4)
        logits.append(l4)

        x = self.stage5(x)
        f5, l5 = self.h5(x)
        feats.append(f5)
        logits.append(l5)

        w = torch.softmax(self.w, dim=0)

        final_logits = (
            w[0] * l1 +
            w[1] * l2 +
            w[2] * l3 +
            w[3] * l4 +
            w[4] * l5
        )

        final_feat = f5

        return logits, feats, final_logits, final_feat


# =========================================================
# Test
# =========================================================
if __name__ == "__main__":

    model = GSD_MobileSlideNet(
        num_classes=5,
        width_mult=0.75
    )

    x = torch.randn(1, 3, 224, 224)

    logits, feats, final_logits, final_feat = model(x)

