# ---- Zoom TTA on the 0.917 SOUP: add scaled-in / scaled-out views (one network, no training) ----
# Paste after the soup and BN cells (uses soup, normalize, Xva, yva, Xte_int, yte_int, Xkaggle, ...).
# Zoom in  (s > 1): upscale bilinear, centre-crop 32x32.  Zoom out (s < 1): downscale, reflect-pad back to 32x32.
# Every view also flipped. Setting CHOSEN ON VAL ONLY (must beat flip-only soup by > 0.2 pt); internal test = honest check.
import os, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
for _n in ["soup", "normalize", "Xva", "yva", "Xte_int", "yte_int", "Xkaggle", "test_ids", "SAMPLE_SUB", "RUN_DIR", "WORK_DIR", "device", "AMP_DTYPE"]:
    assert _n in globals(), f"missing {_n}: run the soup and BN cells first"

def zoom(x, s):
    if s == 1.0: return x
    n = int(round(32 * s))
    y = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False, antialias=s < 1)
    if s > 1:
        o = (n - 32) // 2; return y[:, :, o:o + 32, o:o + 32]
    p = 32 - n; return F.pad(y, (p // 2, p - p // 2, p // 2, p - p // 2), mode="reflect")

@torch.no_grad()
def view_probs(m, X, s, bs=500):
    out = []
    for i in range(0, len(X), bs):
        xb = zoom(normalize(X[i:i + bs]), s).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            out.append((m(xb).float().softmax(1) + m(xb.flip(3)).float().softmax(1)) / 2)
    return torch.cat(out)

SCALES = [0.75, 0.875, 1.0, 1.125, 1.25]
SETTINGS = {
    "flip only (the 0.917 soup)":      [1.0],
    "+ zoom in 1.125":                 [1.0, 1.125],
    "+ zoom out 0.875":                [1.0, 0.875],
    "+ zoom in/out 1.125, 0.875":      [1.0, 1.125, 0.875],
    "+ zoom in 1.125, 1.25":           [1.0, 1.125, 1.25],
    "+ zoom out 0.875, 0.75":          [1.0, 0.875, 0.75],
    "+ all (0.75 ... 1.25)":           [0.75, 0.875, 1.0, 1.125, 1.25],
}
soup.eval(); t0 = time.time()
DATA = {"val": Xva, "test": Xte_int, "kaggle": Xkaggle}
C = {k: {s: view_probs(soup, X, s) for s in SCALES} for k, X in DATA.items()}
acc = lambda p, y: (p.argmax(1) == y).float().mean().item()
P = lambda k, st: sum(C[k][s] for s in SETTINGS[st])
for s in SCALES: print(f"  single view scale {s:5.3f}: val {acc(C['val'][s], yva):.4f} | internal test {acc(C['test'][s], yte_int):.4f}")
rows = [(st, len(v), acc(P("val", st), yva), acc(P("test", st), yte_int)) for st, v in SETTINGS.items()]
df = pd.DataFrame(rows, columns=["setting", "scales", "val", "internal_test"])
print(df.round(4).to_string(index=False)); print(f"({time.time() - t0:.0f}s)")

base = df.iloc[0]; cand = df.iloc[1:].sort_values(["val", "scales"], ascending=[False, True]).iloc[0]
print(f"\nbest on VAL: {cand.setting}: val {cand.val:.4f} (soup {base.val:.4f}) | internal {cand.internal_test:.4f} "
      f"(soup {base.internal_test:.4f}, {int(round((cand.internal_test - base.internal_test) * 2500)):+d} images)")
if cand.val > base.val + 0.002:
    kp = P("kaggle", cand.setting).argmax(1).cpu().numpy(); k0 = P("kaggle", base.setting).argmax(1).cpu().numpy()
    sub = pd.DataFrame({"id": test_ids.astype(int), "label": kp.astype(int)})
    assert len(sub) == len(SAMPLE_SUB) and (sub["id"].values == SAMPLE_SUB["id"].values).all() and sub["label"].between(0, 9).all()
    tag = "soup3_zoomtta_" + "_".join(str(s) for s in SETTINGS[cand.setting]) + f"_val{cand.val:.4f}_test{cand.internal_test:.4f}"
    sub.to_csv(os.path.join(RUN_DIR, f"submission_{tag}.csv"), index=False)
    os.makedirs(os.path.join(WORK_DIR, "zoomtta"), exist_ok=True); local = os.path.join(WORK_DIR, "zoomtta", "submission.csv"); sub.to_csv(local, index=False)
    print(f"DECISION: use it -> {os.path.join(RUN_DIR, f'submission_{tag}.csv')} | Kaggle predictions changed vs soup: {(kp != k0).sum()} of {len(kp)}")
    if globals().get("IN_COLAB"):
        from google.colab import files; files.download(local)
else:
    print("DECISION: keep the 0.917 soup (no zoom setting beats it on VAL by > 0.2 pt). No file written.")
