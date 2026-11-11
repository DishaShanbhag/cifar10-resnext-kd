# ---- TTA test on the 0.915 model: more views of the SAME single network ----
# Every setting is scored on VAL and on INTERNAL TEST. The setting is CHOSEN ON VAL ONLY; the internal test is the honest check.
# Views: horizontal flip x small shifts (reflect padding, the same padding the training crops used).
best = torch.load(BEST_CKPT, map_location=device, weights_only=False)
net = build_model(best["cfg"]).to(device).to(memory_format=torch.channels_last)
net.load_state_dict(best["model"]); net.eval()
MEAN, STD = best["norm"]["mean"].to(device), best["norm"]["std"].to(device)   # exact normalisation of that run
has_joint = getattr(net, "fc_joint", None) is not None
print(f"loaded {BEST_CKPT}\nparams {count_params(net):,} | saved val {best['val_acc']:.4f} | SLA head: {has_joint}")

def shift(x, dy, dx, p=4):
    """Move the image content by (dy, dx) pixels; borders filled by reflection as in training."""
    if dy == 0 and dx == 0: return x
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    return xp[:, :, p - dy:p - dy + 32, p - dx:p - dx + 32].contiguous(memory_format=torch.channels_last)

@torch.no_grad()
def view_probs(X, dy, dx, flip, bs=500):
    """Softmax of the 10-class head (sd) and of SLA aggregation (ag) for one view of every image."""
    sd, ag = [], []
    for i in range(0, len(X), bs):
        xb = normalize(X[i:i + bs])
        if flip: xb = xb.flip(3)
        xb = shift(xb, dy, dx)
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            s = net(xb).float().softmax(1)
            a = _ag_logits(net, xb).float().softmax(1) if has_joint else s
        sd.append(s); ag.append(a)
    return torch.cat(sd), torch.cat(ag)

POS = {
    "flip only (current)":        [(0, 0)],
    "flip + shift 1px (5 pos)":   [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
    "flip + shift 2px (5 pos)":   [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)],
    "flip + shift 2px (9 pos)":   [(dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2)],
}
needed = sorted({p for v in POS.values() for p in v})
DATA = {"val": (Xva, yva), "test": (Xte_int, yte_int), "kaggle": (Xkaggle, None)}
t0 = time.time()
cache = {name: {(dy, dx, fl): view_probs(X, dy, dx, fl) for (dy, dx) in needed for fl in (False, True)}
         for name, (X, _) in DATA.items()}
print(f"computed {len(needed) * 2} views x 3 sets in {time.time() - t0:.0f}s")

def probs(name, setting, mode):
    views = [cache[name][(dy, dx, fl)] for (dy, dx) in POS[setting] for fl in (False, True)]
    if mode == "sd":   return sum(v[0] for v in views)
    if mode == "ag":   return sum(v[1] for v in views)
    return sum(0.5 * (v[0] + v[1]) for v in views)          # sdag

modes = ("sd", "ag", "sdag") if has_joint else ("sd",)
rows = []
for setting in POS:
    for md in modes:
        r = {"setting": setting, "mode": md, "views": 2 * len(POS[setting]) * (5 if md == "sdag" else 4 if md == "ag" else 1)}
        for name in ("val", "test"):
            r[name] = (probs(name, setting, md).argmax(1) == DATA[name][1]).float().mean().item()
        rows.append(r)
res = pd.DataFrame(rows)

# baseline = what the 0.915 submission used: flip TTA, mode chosen by the section-12 rule (sd unless ag/sdag wins VAL by > 0.2 pt)
b = res[res.setting == "flip only (current)"].set_index("mode")
alt = max([m for m in modes if m != "sd"], key=lambda m: b.loc[m, "val"], default="sd")
base_mode = alt if alt != "sd" and b.loc[alt, "val"] > b.loc["sd", "val"] + 0.002 else "sd"
base = res[(res.setting == "flip only (current)") & (res["mode"] == base_mode)].iloc[0]

# choice on VAL only: a new setting must beat the baseline on VAL by > 0.2 pt (5 of 2,500 images), else keep the baseline
cand = res.sort_values(["val", "views"], ascending=[False, True]).iloc[0]
choice = cand if cand.val > base.val + 0.002 else base

print(res.assign(val=res.val.round(4), test=res.test.round(4)).to_string(index=False))
print(f"\nBASELINE (0.915 submission): flip only, mode '{base_mode}': val {base.val:.4f} | internal test {base.test:.4f}")
print(f"CHOSEN ON VAL: {choice.setting}, mode '{choice['mode']}': val {choice.val:.4f} | internal test {choice.test:.4f}"
      f" -> internal test change {100 * (choice.test - base.test):+.2f} pt ({int(round((choice.test - base.test) * 2500)):+d} images of 2,500)")

# submission for the chosen setting (only written if it is not the baseline)
if choice.setting != base.setting or choice["mode"] != base["mode"]:
    kp = probs("kaggle", choice.setting, choice["mode"]).argmax(1).cpu().numpy()
    kb = probs("kaggle", base.setting, base["mode"]).argmax(1).cpu().numpy()
    sub = pd.DataFrame({"id": test_ids.astype(int), "label": kp.astype(int)})
    assert len(sub) == len(SAMPLE_SUB) and (sub["id"].values == SAMPLE_SUB["id"].values).all() and sub["label"].between(0, 9).all()
    tag = re.sub(r"[^a-z0-9]+", "_", f"{choice.setting}_{choice['mode']}".lower()).strip("_")
    drive_sub = os.path.join(RUN_DIR, f"submission_tta_{tag}_val{choice.val:.4f}_test{choice.test:.4f}.csv")
    sub.to_csv(drive_sub, index=False)
    os.makedirs(os.path.join(WORK_DIR, "tta"), exist_ok=True)
    local_sub = os.path.join(WORK_DIR, "tta", "submission.csv"); sub.to_csv(local_sub, index=False)
    print(f"\nsaved {drive_sub}\nKaggle predictions that differ from the 0.915 submission: {(kp != kb).sum()} of {len(kp)}")
    if IN_COLAB:
        from google.colab import files; files.download(local_sub)
else:
    print("\nNo setting beat the baseline on VAL by more than 0.2 pt -> keep the 0.915 submission; no new file written.")
