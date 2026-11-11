# ---- Soup + BatchNorm recompute with TRAINING-MATCHED inputs (the BN step of SWA, Sparse Model Soups, REPAIR) ----
# Paste in the same notebook, after the soup cell (uses its variables). No training, no test images: train split only.
# Why it differs from the earlier Precise BN: in training, BN saw batches of [CutMixed, augmented image x 4 rotations]
# (SLA). The earlier Precise BN fed upright images only. Here the inputs follow the training pipeline.
import copy, os, re, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T

for _n in ["ROOT", "BASE", "RUN_DIR", "WORK_DIR", "CFG", "best", "X_full", "y_full", "sp", "to_dev", "normalize", "MEAN", "STD",
           "build_model", "count_params", "flip_probs", "Xva", "yva", "Xte_int", "yte_int", "Xkaggle", "test_ids", "SAMPLE_SUB",
           "device", "AMP_DTYPE"]:
    assert _n in globals(), f"missing {_n}: run the soup cell (setup part) first in this notebook"

# ---- training split (the 45k the model was trained on) ----
tr_idx = sp["train"] if "train" in sp.files else np.setdiff1d(np.arange(len(y_full)), np.concatenate([sp["val"], sp["test"]]))
Xtr = to_dev(X_full[tr_idx])
assert len(set(tr_idx.tolist()) & set(np.concatenate([sp["val"], sp["test"]]).tolist())) == 0
B, N_ROT = CFG["BATCH_SIZE"], CFG.get("SLA_ROTATIONS", 4)
MIX_PROB = CFG.get("MIX_PROB", CFG.get("CUTMIX_P", 0.0))
print(f"train images {len(Xtr)} | batch {B} | rotations {N_ROT} | CutMix prob {MIX_PROB} | TrivialAug {CFG.get('TRIVIAL_AUG')} "
      f"| grayscale p {CFG.get('GRAYSCALE_P', 0.0)} | cutout {CFG.get('CUTOUT')} | mixup share {CFG.get('MIXUP_SHARE', 0.0)}")

# ---- the 0.915 training augmentation (copied from the training notebook) ----
_trivial = T.TrivialAugmentWide()
def photometric(x_u8):
    if CFG.get("TRIVIAL_AUG", False):
        g = CFG.get("TA_GROUP", 8); x_u8 = torch.cat([_trivial(c) for c in x_u8.split(g)])
    p = CFG.get("GRAYSCALE_P", 0.0)
    if p > 0:
        gray = (x_u8.float() * torch.tensor([0.299, 0.587, 0.114], device=x_u8.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
        gray = gray.round().clamp(0, 255).to(torch.uint8).expand(-1, 3, -1, -1)
        m = (torch.rand(len(x_u8), device=x_u8.device) < p)[:, None, None, None]; x_u8 = torch.where(m, gray, x_u8)
    return x_u8

def train_augment(x_u8):
    x_u8 = photometric(x_u8)
    x = (x_u8.float() / 255 - MEAN) / STD
    n, dev = x.size(0), x.device
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    dy = torch.randint(0, 9, (n,), device=dev); dx = torch.randint(0, 9, (n,), device=dev); ar = torch.arange(32, device=dev)
    x = xp[torch.arange(n, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None],
           (dy[:, None] + ar)[:, None, :, None], (dx[:, None] + ar)[:, None, None, :]]
    flip = torch.rand(n, device=dev) < 0.5; x = torch.where(flip[:, None, None, None], x.flip(3), x)
    if CFG.get("CUTOUT", 0) > 0:
        cy = torch.randint(0, 32, (n,), device=dev); cx = torch.randint(0, 32, (n,), device=dev); h = CFG["CUTOUT"] // 2
        m = ((ar[None, :, None] - cy[:, None, None]).abs() < h) & ((ar[None, None, :] - cx[:, None, None]).abs() < h)
        x = x * (~m)[:, None].float()
    return x

def cutmix_x(x):
    lam = np.random.beta(1.0, 1.0); idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.size(2), x.size(3); r = np.sqrt(1.0 - lam); ch, cw = int(H * r), int(W * r)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2, x1, x2 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw // 2, 0), min(cx + cw // 2, W)
    x = x.clone(); x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]; return x

@torch.no_grad()
def recompute_bn(model, aug, cutmix, rot, seed=0):
    """Same weights; BN running mean/var replaced by an exact average over one pass of the 45k train images."""
    m = copy.deepcopy(model)
    bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    for b in bns: b.reset_running_stats(); b.momentum = None            # None -> exact cumulative average
    m.train(); torch.manual_seed(seed); np.random.seed(seed)
    perm = torch.randperm(len(Xtr), device=device)
    for i in range(0, len(Xtr) - B + 1, B):                               # full batches only, as in training
        xu = Xtr[perm[i:i + B]]
        x = train_augment(xu) if aug else (xu.float() / 255 - MEAN) / STD
        if cutmix and MIX_PROB > 0 and np.random.rand() < MIX_PROB: x = cutmix_x(x)
        if rot: x = torch.cat([torch.rot90(x, k, dims=(2, 3)) for k in range(N_ROT)])
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            m(x.contiguous(memory_format=torch.channels_last))
    m.eval(); return m

acc = lambda s, y: (s.argmax(1) == y).float().mean().item()
def score(m, kaggle=False):
    r = {"val": acc(flip_probs(m, Xva), yva), "test": acc(flip_probs(m, Xte_int), yte_int)}
    if kaggle: r["kaggle_pred"] = flip_probs(m, Xkaggle).argmax(1).cpu().numpy()
    return r

# ---- models: the 0.915 single model and the 0.917 uniform soup of 3 ----
single = build_model(CFG).to(device).to(memory_format=torch.channels_last); single.load_state_dict(best["model"]); single.eval()
SOUP3 = [BASE, BASE + "_ft100_lr0.02", BASE + "_tail50_noheavyaug"]
sds = [torch.load(os.path.join(ROOT, "runs", r, "best.pt"), map_location=device, weights_only=False)["model"] for r in SOUP3]
soup = copy.deepcopy(single)
soup.load_state_dict({k: (sum(s[k].float() for s in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].is_floating_point() else sds[0][k]
                      for k in sds[0]}); soup.eval()
print(f"params {count_params(soup):,} (BN recompute changes buffers only, not parameters)")

VARIANTS = {
    "A. aug + CutMix + 4 rotations (training-matched)": dict(aug=True, cutmix=True, rot=True),
    "B. aug + 4 rotations (no CutMix)":                 dict(aug=True, cutmix=False, rot=True),
    "C. clean + 4 rotations":                           dict(aug=False, cutmix=False, rot=True),
}
t0 = time.time(); rows = []
s0 = score(single); rows.append(("0.915 single, saved BN", s0["val"], s0["test"]))
sA = score(recompute_bn(single, **VARIANTS["A. aug + CutMix + 4 rotations (training-matched)"]))
rows.append(("0.915 single, recompute A (sanity check)", sA["val"], sA["test"]))
base = score(soup, kaggle=True); rows.append(("SOUP, averaged BN (the 0.917 file)", base["val"], base["test"]))
res = {}
for name, kw in VARIANTS.items():
    for seed in ((0, 1) if name.startswith("A") else (0,)):
        r = score(recompute_bn(soup, seed=seed, **kw), kaggle=True); res[(name, seed)] = r
        rows.append((f"SOUP, recompute {name} seed {seed}", r["val"], r["test"]))
df = pd.DataFrame(rows, columns=["model", "val", "internal_test"])
print(df.assign(val=df.val.round(4), internal_test=df.internal_test.round(4)).to_string(index=False))
print(f"(done in {time.time() - t0:.0f}s)  expected soup averaged BN: 0.9868 / 0.9824 | single: 0.9852 / 0.9812")

# ---- decision on VAL only: must beat the soup (averaged BN) by > 0.2 pt (5 of 2,500 images) ----
(kbest, rbest) = max(res.items(), key=lambda kv: kv[1]["val"])
print(f"\nbest on VAL: {kbest[0]} seed {kbest[1]}: val {rbest['val']:.4f} | internal test {rbest['test']:.4f} "
      f"| change vs soup: val {100 * (rbest['val'] - base['val']):+.2f} pt, internal {int(round((rbest['test'] - base['test']) * 2500)):+d} images")
if rbest["val"] > base["val"] + 0.002:
    name, seed = kbest
    m = recompute_bn(soup, seed=seed, **VARIANTS[name])
    kp = flip_probs(m, Xkaggle).argmax(1).cpu().numpy()
    sub = pd.DataFrame({"id": test_ids.astype(int), "label": kp.astype(int)})
    assert len(sub) == len(SAMPLE_SUB) and (sub["id"].values == SAMPLE_SUB["id"].values).all() and sub["label"].between(0, 9).all()
    tag = f"soup3_bn{name[0]}_s{seed}_val{rbest['val']:.4f}_test{rbest['test']:.4f}"
    torch.save({"model": m.state_dict(), "cfg": CFG, "norm": best["norm"], "soup_of": SOUP3, "bn": name},
               os.path.join(RUN_DIR, f"best_{tag}.pt"))
    sub.to_csv(os.path.join(RUN_DIR, f"submission_{tag}.csv"), index=False)
    os.makedirs(os.path.join(WORK_DIR, "soupbn"), exist_ok=True); local = os.path.join(WORK_DIR, "soupbn", "submission.csv"); sub.to_csv(local, index=False)
    print(f"DECISION: use it. saved {os.path.join(RUN_DIR, f'submission_{tag}.csv')}\n"
          f"Kaggle predictions that differ from the 0.917 soup: {(kp != base['kaggle_pred']).sum()} of {len(kp)}")
    if globals().get("IN_COLAB"):
        from google.colab import files; files.download(local)
else:
    print("DECISION: keep the 0.917 soup (no recompute beats it on VAL by > 0.2 pt). No file written.")
