# ============================================================================================================
# big_soup.py : average the WEIGHTS of every model fine-tuned from the 0.915 student (one network, no training).
# Run in a notebook cell:   exec(open("big_soup.py").read())
# Pool = the 0.915 model, ..._ft100_lr0.02, ..._tail50_noheavyaug, and every runs/..._ing*/ best.pt and snap_e*.pt.
# Compares the uniform soup and a 2-fold cross-fitted greedy soup (built on VAL) against the current 0.917 soup.
# Writes a submission (and the averaged network) only if one beats the 0.917 soup on VAL by > 0.2 pt (5 of 2,500).
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

# ---- the pool of ingredients (all fine-tuned from the 0.915 weights) ----
CURRENT3 = [BASE, BASE + "_ft100_lr0.02", BASE + "_tail50_noheavyaug"]
POOL = {}
for r in CURRENT3:
    p = os.path.join(ROOT, "runs", r, "best.pt")
    if os.path.exists(p): POOL[r] = torch.load(p, map_location=device, weights_only=False)["model"]
    else: print("missing:", p)
for p in sorted(glob.glob(os.path.join(ROOT, "runs", BASE + "_ing*", "*.pt"))):
    if os.path.basename(p) == "last.pt": continue
    POOL[os.path.relpath(p, os.path.join(ROOT, "runs"))] = torch.load(p, map_location=device, weights_only=False)["model"]
print(f"{len(POOL)} ingredients:", *POOL, sep="\n  ")

def make_net(names):
    m = copy.deepcopy(net); ref = POOL[names[0]]
    m.load_state_dict({k: (sum(POOL[n][k].float() for n in names) / len(names)) if ref[k].is_floating_point() else ref[k] for k in ref})
    m.eval(); return m
def probs_of(names, splits=("val",)):
    m = make_net(names); return {s: flip_probs(m, DATA[s]) for s in splits}

t0 = time.time()
CUR = probs_of([n for n in CURRENT3 if n in POOL], ("val", "test", "kaggle"))           # the 0.917 soup
VALP = {n: probs_of([n])["val"] for n in POOL}
for n in POOL: print(f"  single {n[-60:]:60s} val {acc(VALP[n], yva):.4f}")

g = torch.Generator().manual_seed(0); idx = torch.randperm(len(yva), generator=g).to(device); h = len(idx) // 2
FOLDS = [(idx[:h], idx[h:]), (idx[h:], idx[:h])]
def greedy_on(ix):
    order = sorted(POOL, key=lambda n: acc(VALP[n][ix], yva[ix]), reverse=True)
    chosen = [order[0]]; best_a = acc(VALP[order[0]][ix], yva[ix])
    for n in order[1:]:
        a = acc(probs_of(chosen + [n])["val"][ix], yva[ix])
        if a >= best_a: chosen.append(n); best_a = a
    return chosen
cv_correct = 0
for a, b in FOLDS:
    gch = greedy_on(a); cv_correct += (probs_of(gch)["val"][b].argmax(1) == yva[b]).sum().item()
cv_greedy = cv_correct / len(yva)
G = greedy_on(torch.arange(len(yva), device=device)); GR = probs_of(G, ("val", "test", "kaggle"))
U = probs_of(list(POOL), ("val", "test", "kaggle"))
print(f"(done in {time.time() - t0:.0f}s)")

cur_val, cur_test = acc(CUR["val"], yva), acc(CUR["test"], yte_int)
print(f"\nCURRENT 0.917 SOUP (3): val {cur_val:.4f} | internal test {cur_test:.4f}   (expected 0.9868 / 0.9824)")
print(f"UNIFORM BIG SOUP ({len(POOL)}): val {acc(U['val'], yva):.4f} | internal test {acc(U['test'], yte_int):.4f}")
print(f"GREEDY SOUP ({len(G)}): cross-fitted val {cv_greedy:.4f} | internal test {acc(GR['test'], yte_int):.4f}")
print("  greedy ingredients:", G)

name, v, R, members = max([("uniform", acc(U["val"], yva), U, list(POOL)), ("greedy", cv_greedy, GR, G)], key=lambda c: c[1])
if v > cur_val + 0.002:
    kp = R["kaggle"].argmax(1).cpu().numpy(); kb = CUR["kaggle"].argmax(1).cpu().numpy()
    sub = pd.DataFrame({"id": test_ids.astype(int), "label": kp.astype(int)})
    assert len(sub) == len(SAMPLE_SUB) and (sub["id"].values == SAMPLE_SUB["id"].values).all() and sub["label"].between(0, 9).all()
    tag = f"bigsoup_{name}_{len(members)}_val{v:.4f}_test{acc(R['test'], yte_int):.4f}"
    drive_sub = os.path.join(RUN_DIR, f"submission_{tag}.csv"); sub.to_csv(drive_sub, index=False)
    torch.save({"model": make_net(members).state_dict(), "cfg": CFG, "norm": best["norm"], "soup_of": members}, os.path.join(RUN_DIR, f"best_{tag}.pt"))
    os.makedirs(os.path.join(WORK_DIR, "bigsoup"), exist_ok=True); local_sub = os.path.join(WORK_DIR, "bigsoup", "submission.csv"); sub.to_csv(local_sub, index=False)
    print(f"DECISION: use the {name} big soup. saved {drive_sub}\nKaggle predictions that differ from the 0.917 soup: {(kp != kb).sum()} of {len(kp)}")
    if IN_COLAB:
        from google.colab import files; files.download(local_sub)
else:
    print("DECISION: keep the 0.917 soup (nothing beats it on VAL by > 0.2 pt). No file written.")
