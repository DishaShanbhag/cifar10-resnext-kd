"""Model definitions, copied from the training notebooks (notebooks/0*_*.ipynb, section 8).

Final model: ResNeXt-29 (cardinality 4, bottleneck width 24, stage widths 280/528/1040, blocks 3/3/3)
with an SLA joint (class x rotation) head used only during training. 4,998,738 trainable parameters.
Teacher (training only): SE-WRN-28-10 + SLA head.
"""
import torch, torch.nn as nn, torch.nn.functional as F

FINAL_CFG = dict(ARCH="resnext", CARDINALITY=4, BOTTLENECK_WIDTH=24, WIDTHS=(280, 528, 1040), BLOCKS=(3, 3, 3),
                 DROPOUT=0.0, SLA=True, SLA_ROTATIONS=4)

# ---- Models (same code as the training notebooks): ResNet, ResNeXt-29, Shake-Shake, chosen with CFG["ARCH"] ----
class BasicBlock(nn.Module):
    """Post-activation basic block (our original ResNet)."""
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)

class CifarResNet(nn.Module):
    def __init__(self, widths=(64, 128, 256), blocks=(3, 3, 3), num_classes=10, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, (w, nb) in enumerate(zip(widths, blocks)):
            for j in range(nb):
                layers.append(BasicBlock(cin, w, 2 if (j == 0 and i > 0) else 1)); cin = w
        self.layers = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for b in self.layers: nn.init.zeros_(b.bn2.weight)
    def forward(self, x):
        return self.fc(self.drop(self.features(x)))
    def features(self, x):
        return torch.flatten(F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1), 1)

class ResNeXtBlock(nn.Module):
    """ResNeXt bottleneck: 1x1 conv -> grouped 3x3 conv (cardinality groups) -> 1x1 conv, plus skip (Xie et al. 2017)."""
    def __init__(self, cin, inner, cout, stride, cardinality):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, inner, 1, bias=False); self.bn1 = nn.BatchNorm2d(inner)
        self.conv2 = nn.Conv2d(inner, inner, 3, stride, 1, groups=cardinality, bias=False); self.bn2 = nn.BatchNorm2d(inner)
        self.conv3 = nn.Conv2d(inner, cout, 1, bias=False); self.bn3 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x), inplace=True)

class ResNeXt29(nn.Module):
    """CIFAR ResNeXt-29: 3x3 stem (64) + 3 stages x 3 blocks; output widths 256/512/1024;
    inner width = cardinality * bottleneck_width, doubled at each stage. (2, 64) is the standard 2x64d."""
    def __init__(self, cardinality=2, bottleneck_width=64, blocks=(3, 3, 3), out_widths=(256, 512, 1024),
                 num_classes=10, dropout=0.0, stem=64, strides=(1, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem), nn.ReLU(inplace=True))
        layers, cin = [], stem
        for i, (cout, nb) in enumerate(zip(out_widths, blocks)):
            inner = cardinality * bottleneck_width * (2 ** i)
            for j in range(nb):
                layers.append(ResNeXtBlock(cin, inner, cout, strides[i] if j == 0 else 1, cardinality)); cin = cout
        self.layers = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout); self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for b in self.layers: nn.init.zeros_(b.bn3.weight)
    def forward(self, x):
        return self.fc(self.drop(self.features(x)))
    def features(self, x):
        return torch.flatten(F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1), 1)

class SE(nn.Module):
    """Squeeze-excitation (Hu et al. 2018): channel weights from global average pooling. Teachers only."""
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1, self.fc2 = nn.Conv2d(c, c // r, 1), nn.Conv2d(c // r, c, 1)
    def forward(self, x):
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)))))

class WBlock(nn.Module):
    """Pre-activation wide residual block (Zagoruyko & Komodakis 2016)."""
    def __init__(self, cin, cout, stride, drop, se_r=0):
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2, self.conv2 = nn.BatchNorm2d(cout), nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(drop)
        self.skip = None if (stride == 1 and cin == cout) else nn.Conv2d(cin, cout, 1, stride, bias=False)
        if se_r: self.se = SE(cout, se_r)
    def forward(self, x):
        o = F.relu(self.bn1(x))
        sc = x if self.skip is None else self.skip(o)
        o = self.conv2(self.dropout(F.relu(self.bn2(self.conv1(o)))))
        if hasattr(self, "se"): o = self.se(o)
        return o + sc

class WRN(nn.Module):
    """WideResNet for CIFAR (Zagoruyko & Komodakis 2016). k = widen factor (WRN-28-10: 36.5M params) or explicit stage widths."""
    def __init__(self, depth=28, k=10, dropout=0.0, num_classes=10, se_r=0):
        super().__init__()
        n = (depth - 4) // 6
        w = [16] + (list(k) if isinstance(k, (tuple, list)) else [16 * k, 32 * k, 64 * k])   # k: widen factor, or the 3 stage widths
        self.stem = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        layers, cin = [], w[0]
        for i in range(3):
            for j in range(n):
                layers.append(WBlock(cin, w[i + 1], 2 if (j == 0 and i > 0) else 1, dropout, se_r)); cin = w[i + 1]
        self.layers = nn.Sequential(*layers); self.bn = nn.BatchNorm2d(cin)
        self.drop = nn.Dropout(0.0); self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    def features(self, x):
        return torch.flatten(F.adaptive_avg_pool2d(F.relu(self.bn(self.layers(self.stem(x)))), 1), 1)
    def forward(self, x):
        return self.fc(self.drop(self.features(x)))

def _build_base(c):
    """Build any of our architectures from a config dict (also rebuilds teachers from their saved cfg)."""
    arch = c.get("ARCH", "resnet")
    if arch == "resnet":
        return CifarResNet(c["WIDTHS"], c["BLOCKS"], dropout=c.get("DROPOUT", 0.0))
    if arch == "resnext":
        return ResNeXt29(c.get("CARDINALITY", 2), c.get("BOTTLENECK_WIDTH", 64), c.get("BLOCKS", (3, 3, 3)),
                         c.get("WIDTHS", (256, 512, 1024)), dropout=c.get("DROPOUT", 0.0),
                         strides=tuple(c.get("STRIDES", (1, 2, 2))))   # (1, 1, 2) = higher-resolution student, same params
    if arch in ("wrn", "sewrn"):     # sewrn = WRN with SE blocks: a training-only teacher
        return WRN(c["DEPTH"], c["WIDEN"], c.get("DROPOUT", 0.0), se_r=c.get("SE_REDUCTION", 16) if arch == "sewrn" else 0)
    raise ValueError(f"unknown ARCH {arch}")

def build_model(c):
    """Base network; with SLA on, adds the joint (class x rotation) head used only during training."""
    net = _build_base(c)
    if c.get("SLA", False):
        net.fc_joint = nn.Linear(net.fc.in_features, 10 * c.get("SLA_ROTATIONS", 4))
    return net

def forward_both(net, x):
    """Returns (10-class self-distillation head logits, joint class x rotation head logits)."""
    f = net.drop(net.features(x))
    return net.fc(f), net.fc_joint(f)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
