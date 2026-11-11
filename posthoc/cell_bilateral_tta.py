# ---- Bilateral-filter TTA (edge-preserving smoothing) on the 0.917 SOUP and the 0.915 SINGLE model (no training) ----
# Paste after the soup and BN cells (uses soup, single, MEAN, STD, Xva, yva, Xte_int, yte_int, Xkaggle, ...).
# Filter on pixels in [0,1], 5x5 window, spatial sigma 1.0; range sigma = how different a neighbour's colour may be and
# still be averaged (small = only near-identical pixels are smoothed, edges kept). Every view also flipped.
# Setting CHOSEN ON VAL ONLY (must beat flip-only by > 0.2 pt); internal test = honest check.
import os, re, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
for _n in ["soup", "single", "MEAN", "STD", "Xva", "yva", "Xte_int", "yte_int", "Xkaggle", "test_ids", "SAMPLE_SUB", "RUN_DIR"]:
    assert _n in globals(), f"missing {_n}: run the soup and BN cells first"

def bilateral(x, sigma_s=1.0, sigma_r=0.1, r=2):
    """x: N,3,H,W in [0,1]. Joint (all 3 channels) range weight, Gaussian spatial weight."""
    N, C, H, W = x.shape; k = 2 * r + 1
    patches = F.unfold(F.pad(x, (r, r, r, r), mode="reflect"), k).view(N, C, k * k, H * W)
    centre = x.view(N, C, 1, H * W)
    ax = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    ws = torch.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma_s ** 2)).view(1, 1, k * k, 1)
    wr = torch.exp(-((patches - centre) ** 2).sum(1, keepdim=True) / (2 * sigma_r ** 2))
    w = ws * wr
    return ((patches * w).sum(2) / w.sum(2)).view(N, C, H, W)

OPS = {"none": None, "bilat r0.05": 0.05, "bilat r0.10": 0.10, "bilat r0.20": 0.20}
SET_L = {"flip only": ["none"],
         "+ bilat 0.05": ["none", "bilat r0.05"], "+ bilat 0.10": ["none", "bilat r0.10"], "+ bilat 0.20": ["none", "bilat r0.20"],
         "+ bilat 0.05 + 0.10": ["none", "bilat r0.05", "bilat r0.10"]}

@torch.no_grad()
def bil_probs(m, X, op, bs=500):
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].float() / 255
        if OPS[op] is not None: x = bilateral(x, 1.0, OPS[op])
        xb = ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            out.append((m(xb).float().softmax(1) + m(xb.flip(3)).float().softmax(1)) / 2)
    return torch.cat(out)

acc = lambda p, y: (p.argmax(1) == y).float().mean().item()
DATA = {"val": Xva, "test": Xte_int, "kaggle": Xkaggle}
for mname, m in [("0.917 SOUP", soup), ("0.915 SINGLE", single)]:
    m.eval(); t0 = time.time()
    CL = {k: {op: bil_probs(m, X, op) for op in OPS} for k, X in DATA.items()}
    PL = lambda k, st: sum(CL[k][op] for op in SET_L[st])
    print(f"\n===== {mname} =====")
    for op in OPS: print(f"  single view {op:12s}: val {acc(CL['val'][op], yva):.4f} | internal test {acc(CL['test'][op], yte_int):.4f}")
    rows = [(st, len(v), acc(PL("val", st), yva), acc(PL("test", st), yte_int)) for st, v in SET_L.items()]
    dfl = pd.DataFrame(rows, columns=["setting", "views", "val", "internal_test"]); print(dfl.round(4).to_string(index=False))
    b0 = dfl.iloc[0]; cl = dfl.iloc[1:].sort_values(["val", "views"], ascending=[False, True]).iloc[0]
    print(f"best on VAL: {cl.setting}: val {cl.val:.4f} (flip only {b0.val:.4f}) | internal {cl.internal_test:.4f} "
          f"(flip only {b0.internal_test:.4f}, {int(round((cl.internal_test - b0.internal_test) * 2500)):+d} images) | {time.time() - t0:.0f}s")
    if cl.val > b0.val + 0.002:
        kp = PL("kaggle", cl.setting).argmax(1).cpu().numpy(); k0 = PL("kaggle", "flip only").argmax(1).cpu().numpy()
        sub = pd.DataFrame({"id": test_ids.astype(int), "label": kp.astype(int)})
        assert len(sub) == len(SAMPLE_SUB) and (sub["id"].values == SAMPLE_SUB["id"].values).all() and sub["label"].between(0, 9).all()
        tag = re.sub(r"[^a-z0-9.]+", "_", f"{mname}_bilateral_{cl.setting}".lower()).strip("_") + f"_val{cl.val:.4f}_test{cl.internal_test:.4f}"
        sub.to_csv(os.path.join(RUN_DIR, f"submission_{tag}.csv"), index=False)
        print(f"DECISION: use it -> {os.path.join(RUN_DIR, f'submission_{tag}.csv')} | Kaggle predictions changed: {(kp != k0).sum()} of {len(kp)}")
    else:
        print("DECISION: keep flip only (no bilateral setting beats it on VAL by > 0.2 pt). No file written.")
