# ============================================================================================================
# STANDALONE: colour + size TTA on the 0.917 SOUP and the 0.915 SINGLE model (no training). One cell, fresh runtime OK.
# Works on Colab (mounts Drive MyDrive/cifar10_resnet_challenge), Kaggle, HPC. Chosen on VAL only (> +0.2 pt).
# ============================================================================================================
import os, re, glob, time, json, copy
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
try:
    import google.colab  # noqa
    IN_COLAB = True
    from google.colab import drive; drive.mount("/content/drive")
    ROOT, WORK_DIR = "/content/drive/MyDrive/cifar10_resnet_challenge", "/content/work"
except ImportError:
    IN_COLAB = False
    if os.path.exists("/kaggle/working/cifar10_resnet_challenge"): ROOT, WORK_DIR = "/kaggle/working/cifar10_resnet_challenge", "/kaggle/working/work"
    else: ROOT = os.environ.get("SOUP_ROOT", "/scratch/ds8223/cifar10_resnet_challenge"); WORK_DIR = os.environ.get("SOUP_WORK", "/scratch/ds8223/kd_work")
BASE = "kd_sewrn_only_student_resnext29_slasd_400ep_v1"
RUN_DIR = os.path.join(ROOT, "runs", BASE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if device.type == "cpu" or torch.cuda.get_device_capability()[0] >= 8 else None
print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (slow)")

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



# ---- data, split, base model ----
best = torch.load(os.path.join(RUN_DIR, "best.pt"), map_location=device, weights_only=False); CFG = best["cfg"]
CACHE_DIR, SPLIT_DIR = os.path.join(ROOT, "data", "cache"), os.path.join(ROOT, "data", "splits")
X_full = np.load(os.path.join(CACHE_DIR, "train_x.npy")); y_full = np.load(os.path.join(CACHE_DIR, "train_y.npy"))
X_test = np.load(os.path.join(CACHE_DIR, "test_x.npy")); test_ids = np.load(os.path.join(CACHE_DIR, "test_ids.npy"))
SAMPLE_SUB = pd.read_csv(os.path.join(CACHE_DIR, "sample_submission.csv"))
sp = np.load(os.path.join(SPLIT_DIR, f"split_seed{CFG['SPLIT_SEED']}_val{CFG['VAL_FRAC']}_test{CFG['TEST_FRAC']}_n{len(y_full)}.npz"))
def to_dev(x): return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
Xva, yva = to_dev(X_full[sp["val"]]), torch.from_numpy(y_full[sp["val"]]).to(device)
Xte_int, yte_int = to_dev(X_full[sp["test"]]), torch.from_numpy(y_full[sp["test"]]).to(device)
Xkaggle = to_dev(X_test)
MEAN, STD = best["norm"]["mean"].to(device), best["norm"]["std"].to(device)
def normalize(x_u8): return ((x_u8.float() / 255 - MEAN) / STD).contiguous(memory_format=torch.channels_last)
net = build_model(CFG).to(device).to(memory_format=torch.channels_last); net.load_state_dict(best["model"]); net.eval()
acc = lambda s, y: (s.argmax(1) == y).float().mean().item()
DATA = {"val": Xva, "test": Xte_int, "kaggle": Xkaggle}

@torch.no_grad()
def flip_probs(m, X, bs=500):                       # exactly how the 0.915 / 0.917 submissions were scored
    out = []
    for i in range(0, len(X), bs):
        xb = normalize(X[i:i + bs])
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            out.append((m(xb).float().softmax(1) + m(xb.flip(3)).float().softmax(1)) / 2)
    return torch.cat(out)

# ---- the two models: 0.915 single and the 0.917 uniform soup of 3 ----
single = net
SOUP3 = [BASE, BASE + "_ft100_lr0.02", BASE + "_tail50_noheavyaug"]
sds = [torch.load(os.path.join(ROOT, "runs", r, "best.pt"), map_location=device, weights_only=False)["model"] for r in SOUP3]
soup = copy.deepcopy(single)
soup.load_state_dict({k: (sum(s[k].float() for s in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].is_floating_point() else sds[0][k]
                      for k in sds[0]}); soup.eval(); del sds
print(f"0.915 single: val {acc(flip_probs(single, Xva), yva):.4f} | internal {acc(flip_probs(single, Xte_int), yte_int):.4f}  (expected 0.9852 / 0.9812)")
print(f"0.917 soup:   val {acc(flip_probs(soup, Xva), yva):.4f} | internal {acc(flip_probs(soup, Xte_int), yte_int):.4f}  (expected 0.9868 / 0.9824)")
print(f"params {count_params(soup):,}")

def zoom(x, s):
    if s == 1.0: return x
    n = int(round(32 * s))
    y = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False, antialias=s < 1)
    if s > 1:
        o = (n - 32) // 2; return y[:, :, o:o + 32, o:o + 32]
    p = 32 - n; return F.pad(y, (p // 2, p - p // 2, p // 2, p - p // 2), mode="reflect")

import itertools
import torchvision.transforms.functional as TF

COLOR = {"orig": lambda u: u,
         "gray": lambda u: TF.rgb_to_grayscale(u, 3),
         "sat0.5": lambda u: TF.adjust_saturation(u, 0.5), "sat1.5": lambda u: TF.adjust_saturation(u, 1.5),
         "hue-0.05": lambda u: TF.adjust_hue(u, -0.05), "hue+0.05": lambda u: TF.adjust_hue(u, 0.05)}
SIZES = [1.0, 0.875, 1.125]

@torch.no_grad()
def cs_probs(m, X, col, s, bs=500):
    out = []
    for i in range(0, len(X), bs):
        u = COLOR[col](X[i:i + bs])                                   # uint8, colour change in pixel space
        xb = zoom((u.float() / 255 - MEAN) / STD, s).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            out.append((m(xb).float().softmax(1) + m(xb.flip(3)).float().softmax(1)) / 2)
    return torch.cat(out)

SET_C = {"flip only": [("orig", 1.0)],
         "+ gray": [("orig", 1.0), ("gray", 1.0)],
         "+ saturation 0.5/1.5": [("orig", 1.0), ("sat0.5", 1.0), ("sat1.5", 1.0)],
         "+ hue -/+0.05": [("orig", 1.0), ("hue-0.05", 1.0), ("hue+0.05", 1.0)],
         "+ all colours": [(c, 1.0) for c in COLOR],
         "+ all colours + sizes 0.875/1.125": [(c, 1.0) for c in COLOR] + [("orig", 0.875), ("orig", 1.125)],
         "all colours x all sizes": list(itertools.product(COLOR, SIZES))}
need = sorted({v for vs in SET_C.values() for v in vs})
acc = lambda p, y: (p.argmax(1) == y).float().mean().item()
DATA = {"val": Xva, "test": Xte_int, "kaggle": Xkaggle}
for mname, m in [("0.917 SOUP", soup), ("0.915 SINGLE", single)]:
    m.eval(); t0 = time.time()
    CC = {k: {v: cs_probs(m, X, *v) for v in need} for k, X in DATA.items()}
    PC = lambda k, st: sum(CC[k][v] for v in SET_C[st])
    print(f"\n===== {mname} =====")
    for v in need:
        if v[1] == 1.0: print(f"  single view {v[0]:9s}: val {acc(CC['val'][v], yva):.4f} | internal test {acc(CC['test'][v], yte_int):.4f}")
    rows = [(st, len(v), acc(PC("val", st), yva), acc(PC("test", st), yte_int)) for st, v in SET_C.items()]
    dfc = pd.DataFrame(rows, columns=["setting", "views", "val", "internal_test"]); print(dfc.round(4).to_string(index=False))
    b0 = dfc.iloc[0]; cc = dfc.iloc[1:].sort_values(["val", "views"], ascending=[False, True]).iloc[0]
    print(f"best on VAL: {cc.setting}: val {cc.val:.4f} (flip only {b0.val:.4f}) | internal {cc.internal_test:.4f} "
          f"(flip only {b0.internal_test:.4f}, {int(round((cc.internal_test - b0.internal_test) * 2500)):+d} images) | {time.time() - t0:.0f}s")
    if cc.val > b0.val + 0.002:
        kp = PC("kaggle", cc.setting).argmax(1).cpu().numpy(); k0 = PC("kaggle", "flip only").argmax(1).cpu().numpy()
        sub = pd.DataFrame({"id": test_ids.astype(int), "label": kp.astype(int)})
        assert len(sub) == len(SAMPLE_SUB) and (sub["id"].values == SAMPLE_SUB["id"].values).all() and sub["label"].between(0, 9).all()
        tag = re.sub(r"[^a-z0-9.]+", "_", f"{mname}_colsize_{cc.setting}".lower()).strip("_") + f"_val{cc.val:.4f}_test{cc.internal_test:.4f}"
        sub.to_csv(os.path.join(RUN_DIR, f"submission_{tag}.csv"), index=False)
        print(f"DECISION: use it -> {os.path.join(RUN_DIR, f'submission_{tag}.csv')} | Kaggle predictions changed: {(kp != k0).sum()} of {len(kp)}")
    else:
        print("DECISION: keep flip only (no colour/size setting beats it on VAL by > 0.2 pt). No file written.")
