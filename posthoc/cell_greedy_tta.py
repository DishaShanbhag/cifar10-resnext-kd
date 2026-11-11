# ---- Greedy TTA on the 0.917 SOUP: pool of views, chosen on one half of VAL, scored on the other half (cross-fitted) ----
# Paste after the standalone cell (uses soup, normalize, zoom, MEAN, STD, data). No training. Every view = itself + its flip.
import numpy as np, torch, torch.nn.functional as F
def _shift(x, dy, dx, p=4):
    if dy == 0 and dx == 0: return x
    xp = F.pad(x, (p, p, p, p), mode="reflect"); return xp[:, :, p - dy:p - dy + 32, p - dx:p - dx + 32]
def _gblur(x, s):
    r = 2; k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * s ** 2)); k = k / k.sum(); c = x.size(1)
    x = F.pad(x, (r, r, r, r), mode="reflect")
    return F.conv2d(F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
def _bilat(x01, sr, ss=1.0, r=2):
    N, C, H, W = x01.shape; k = 2 * r + 1
    pt = F.unfold(F.pad(x01, (r, r, r, r), mode="reflect"), k).view(N, C, k * k, H * W); ce = x01.view(N, C, 1, H * W)
    ax = torch.arange(-r, r + 1, device=x01.device, dtype=x01.dtype)
    w = torch.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * ss ** 2)).view(1, 1, k * k, 1) * torch.exp(-((pt - ce) ** 2).sum(1, keepdim=True) / (2 * sr ** 2))
    return ((pt * w).sum(2) / w.sum(2)).view(N, C, H, W)

VIEWS = {   # each takes uint8 images, returns normalised float images
    "orig":        lambda u: normalize(u),
    "shift +2,0":  lambda u: _shift(normalize(u), 2, 0),  "shift -2,0": lambda u: _shift(normalize(u), -2, 0),
    "shift 0,+2":  lambda u: _shift(normalize(u), 0, 2),  "shift 0,-2": lambda u: _shift(normalize(u), 0, -2),
    "blur 0.5":    lambda u: _gblur(normalize(u).float(), 0.5),
    "sharpen 0.5": lambda u: (lambda x: x + 0.5 * (x - _gblur(x, 1.0)))(normalize(u).float()),
    "bilat 0.05":  lambda u: (_bilat(u.float() / 255, 0.05) - MEAN) / STD,
    "zoom 0.875":  lambda u: zoom(normalize(u).float(), 0.875),
    "zoom 1.125":  lambda u: zoom(normalize(u).float(), 1.125),
}
@torch.no_grad()
def vprobs(m, X, name, bs=500):
    out = []
    for i in range(0, len(X), bs):
        xb = VIEWS[name](X[i:i + bs]).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            out.append((m(xb).float().softmax(1) + m(xb.flip(3)).float().softmax(1)) / 2)
    return torch.cat(out)

soup.eval(); t0 = time.time()
V = {k: {n: vprobs(soup, X, n) for n in VIEWS} for k, X in {"val": Xva, "test": Xte_int, "kaggle": Xkaggle}.items()}
acc = lambda p, y: (p.argmax(1) == y).float().mean().item()
for n in VIEWS: print(f"  view {n:12s}: val {acc(V['val'][n], yva):.4f} | internal test {acc(V['test'][n], yte_int):.4f}")
print(f"({time.time() - t0:.0f}s)")

def greedy(ix):
    """Forward selection on VAL images ix, starting from 'orig' (= the current flip TTA); add a view only if it STRICTLY helps."""
    chosen = ["orig"]; cur = V["val"]["orig"][ix].clone(); best = acc(cur, yva[ix])
    while True:
        cands = [(acc(cur + V["val"][n][ix], yva[ix]), n) for n in VIEWS if n not in chosen]
        if not cands: break
        a, n = max(cands)
        if a <= best: break
        chosen.append(n); cur = cur + V["val"][n][ix]; best = a
    return chosen
P = lambda k, names: sum(V[k][n] for n in names)

# cross-fitted estimate: 5 random 50/50 splits of VAL, choose on one half, score on the other (both ways)
cv = []
for seed in range(5):
    g = torch.Generator().manual_seed(seed); idx = torch.randperm(len(yva), generator=g).to(device); h = len(idx) // 2
    correct = 0
    for a, b in [(idx[:h], idx[h:]), (idx[h:], idx[:h])]:
        ch = greedy(a); correct += (P("val", ch)[b].argmax(1) == yva[b]).sum().item()
    cv.append(correct / len(yva))
cv_acc = float(np.mean(cv)); flip_val = acc(V["val"]["orig"], yva); flip_test = acc(V["test"]["orig"], yte_int)
final = greedy(torch.arange(len(yva), device=device))
fin_test = acc(P("test", final), yte_int)
print(f"\nflip only (the 0.917 soup):       val {flip_val:.4f} | internal test {flip_test:.4f}")
print(f"greedy TTA, cross-fitted val:     {cv_acc:.4f}  (5 splits: {' '.join(f'{c:.4f}' for c in cv)})")
print(f"greedy TTA chosen on all of VAL:  {final} -> internal test {fin_test:.4f} ({int(round((fin_test - flip_test) * 2500)):+d} images vs flip)")
if cv_acc > flip_val + 0.002 and len(final) > 1:
    kp = P("kaggle", final).argmax(1).cpu().numpy(); k0 = V["kaggle"]["orig"].argmax(1).cpu().numpy()
    sub = pd.DataFrame({"id": test_ids.astype(int), "label": kp.astype(int)})
    assert len(sub) == len(SAMPLE_SUB) and (sub["id"].values == SAMPLE_SUB["id"].values).all() and sub["label"].between(0, 9).all()
    path = os.path.join(RUN_DIR, f"submission_soup3_greedyTTA_{len(final)}views_cv{cv_acc:.4f}_test{fin_test:.4f}.csv"); sub.to_csv(path, index=False)
    print(f"DECISION: use it -> {path} | Kaggle predictions changed vs soup: {(kp != k0).sum()} of {len(kp)}")
else:
    print("DECISION: keep flip only (greedy TTA does not beat it on cross-fitted VAL by > 0.2 pt). No file written.")
