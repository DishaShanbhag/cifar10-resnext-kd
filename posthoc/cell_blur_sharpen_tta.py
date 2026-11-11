# ---- Blur / sharpen TTA on the 0.917 SOUP and the 0.915 SINGLE model (one network each, no training) ----
# Paste after the soup, BN and zoom cells (uses soup, single, normalize, DATA, acc, ...).
# blur: Gaussian (sigma), sharpen: unsharp mask x + a*(x - blur(x)). Every view also flipped.
# Setting CHOSEN ON VAL ONLY (must beat flip-only by > 0.2 pt); internal test = honest check.
import os, re, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
for _n in ["soup", "single", "normalize", "Xva", "yva", "Xte_int", "yte_int", "Xkaggle", "test_ids", "SAMPLE_SUB", "RUN_DIR", "WORK_DIR"]:
    assert _n in globals(), f"missing {_n}: run the soup and BN cells first"

def gblur(x, sigma):
    r = 2; k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    c = x.size(1); x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)

OPS = {"none": lambda x: x,
       "blur s0.5": lambda x: gblur(x, 0.5), "blur s0.8": lambda x: gblur(x, 0.8),
       "sharpen a0.5": lambda x: x + 0.5 * (x - gblur(x, 1.0)), "sharpen a1.0": lambda x: x + 1.0 * (x - gblur(x, 1.0))}
SET_B = {"flip only": ["none"],
         "+ blur 0.5": ["none", "blur s0.5"], "+ blur 0.8": ["none", "blur s0.8"],
         "+ sharpen 0.5": ["none", "sharpen a0.5"], "+ sharpen 1.0": ["none", "sharpen a1.0"],
         "+ blur 0.5 + sharpen 0.5": ["none", "blur s0.5", "sharpen a0.5"],
         "+ blur 0.8 + sharpen 1.0": ["none", "blur s0.8", "sharpen a1.0"]}

@torch.no_grad()
def op_probs(m, X, op, bs=500):
    out = []
    for i in range(0, len(X), bs):
        xb = OPS[op](normalize(X[i:i + bs]).float()).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            out.append((m(xb).float().softmax(1) + m(xb.flip(3)).float().softmax(1)) / 2)
    return torch.cat(out)

acc = lambda p, y: (p.argmax(1) == y).float().mean().item()
DATA = {"val": Xva, "test": Xte_int, "kaggle": Xkaggle}
for mname, m in [("0.917 SOUP", soup), ("0.915 SINGLE", single)]:
    m.eval(); t0 = time.time()
    CB = {k: {op: op_probs(m, X, op) for op in OPS} for k, X in DATA.items()}
    PB = lambda k, st: sum(CB[k][op] for op in SET_B[st])
    print(f"\n===== {mname} =====")
    for op in OPS: print(f"  single view {op:13s}: val {acc(CB['val'][op], yva):.4f} | internal test {acc(CB['test'][op], yte_int):.4f}")
    rows = [(st, len(v), acc(PB("val", st), yva), acc(PB("test", st), yte_int)) for st, v in SET_B.items()]
    dfb = pd.DataFrame(rows, columns=["setting", "views", "val", "internal_test"]); print(dfb.round(4).to_string(index=False))
    b0 = dfb.iloc[0]; cb = dfb.iloc[1:].sort_values(["val", "views"], ascending=[False, True]).iloc[0]
    print(f"best on VAL: {cb.setting}: val {cb.val:.4f} (flip only {b0.val:.4f}) | internal {cb.internal_test:.4f} "
          f"(flip only {b0.internal_test:.4f}, {int(round((cb.internal_test - b0.internal_test) * 2500)):+d} images) | {time.time() - t0:.0f}s")
    if cb.val > b0.val + 0.002:
        kp = PB("kaggle", cb.setting).argmax(1).cpu().numpy(); k0 = PB("kaggle", "flip only").argmax(1).cpu().numpy()
        sub = pd.DataFrame({"id": test_ids.astype(int), "label": kp.astype(int)})
        assert len(sub) == len(SAMPLE_SUB) and (sub["id"].values == SAMPLE_SUB["id"].values).all() and sub["label"].between(0, 9).all()
        tag = re.sub(r"[^a-z0-9.]+", "_", f"{mname}_blurtta_{cb.setting}".lower()).strip("_") + f"_val{cb.val:.4f}_test{cb.internal_test:.4f}"
        sub.to_csv(os.path.join(RUN_DIR, f"submission_{tag}.csv"), index=False)
        print(f"DECISION: use it -> {os.path.join(RUN_DIR, f'submission_{tag}.csv')} | Kaggle predictions changed: {(kp != k0).sum()} of {len(kp)}")
    else:
        print("DECISION: keep flip only (no blur/sharpen setting beats it on VAL by > 0.2 pt). No file written.")
