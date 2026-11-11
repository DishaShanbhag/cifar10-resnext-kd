# ---- Teacher-guided last-layer fine-tune of the 0.917 SOUP (one network; only the final 10-class layer changes) ----
# Paste after the soup, BN and option-3 cells (uses soup, teachers[0] = SE-WRN, Ts, train_augment, Xtr, ytr, flip_probs).
# Backbone frozen (eval mode, BN untouched). The fc layer learns to match the SE-WRN teacher on augmented TRAIN images.
# Decision on VAL only (> soup + 0.2 pt); internal test is the honest check. ~1 min per epoch on an L4 (estimate).
import os, math, time, copy, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
for _n in ["soup", "teachers", "Ts", "train_augment", "Xtr", "ytr", "flip_probs", "Xkaggle", "test_ids", "SAMPLE_SUB", "RUN_DIR", "WORK_DIR"]:
    assert _n in globals(), f"missing {_n}: run the soup, BN and option-3 cells first"
teacher, T_t = teachers[0], Ts[0]                       # SE-WRN-28-10 + SLA, calibrated temperature from option-3 cell
HEAD_EPOCHS, HEAD_LR, ALPHA, BSZ = 4, 0.01, 0.9, 512
torch.manual_seed(0); np.random.seed(0)
acc = lambda p, y: (p.argmax(1) == y).float().mean().item()

stu = copy.deepcopy(soup).eval()
for p in stu.parameters(): p.requires_grad_(False)
for p in stu.fc.parameters(): p.requires_grad_(True)
fc0 = copy.deepcopy(stu.fc.state_dict())
opt = torch.optim.SGD(stu.fc.parameters(), lr=HEAD_LR, momentum=0.9, nesterov=True, weight_decay=5e-4)
steps = HEAD_EPOCHS * math.ceil(len(Xtr) / BSZ); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
ce = nn.CrossEntropyLoss(label_smoothing=0.1)

base_v, base_t = acc(flip_probs(stu, Xva), yva), acc(flip_probs(stu, Xte_int), yte_int)
rows = [("soup (start)", 0, base_v, base_t)]; states = {0: copy.deepcopy(fc0)}
for ep in range(1, HEAD_EPOCHS + 1):
    t0 = time.time(); perm = torch.randperm(len(Xtr), device=device)
    for i in range(0, len(Xtr), BSZ):
        idx = perm[i:i + BSZ]; xb = train_augment(Xtr[idx]).contiguous(memory_format=torch.channels_last); yb = ytr[idx]
        with torch.no_grad(), torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            f = stu.drop(stu.features(xb)).float()
            tp = F.softmax(teacher(xb).float() / T_t, 1)
        lo = stu.fc(f)
        loss = ALPHA * F.kl_div(F.log_softmax(lo, 1), tp, reduction="batchmean") + (1 - ALPHA) * ce(lo, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    v, t = acc(flip_probs(stu, Xva), yva), acc(flip_probs(stu, Xte_int), yte_int)
    rows.append((f"head fine-tune epoch {ep}", ep, v, t)); states[ep] = copy.deepcopy(stu.fc.state_dict())
    print(f"epoch {ep}: val {v:.4f} | internal test {t:.4f} | {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows, columns=["model", "epoch", "val", "internal_test"]); print(df.round(4).to_string(index=False))

# decision on VAL only: best epoch must beat the soup by > 0.2 pt (5 of 2,500)
b = df.iloc[1:].sort_values(["val", "epoch"], ascending=[False, True]).iloc[0]
print(f"best on VAL: epoch {int(b.epoch)}: val {b.val:.4f} (soup {base_v:.4f}) | internal {b.internal_test:.4f} (soup {base_t:.4f}, "
      f"{int(round((b.internal_test - base_t) * 2500)):+d} images)")
if b.val > base_v + 0.002:
    stu.fc.load_state_dict(states[int(b.epoch)]); stu.eval()
    kp = flip_probs(stu, Xkaggle).argmax(1).cpu().numpy(); k0 = flip_probs(soup, Xkaggle).argmax(1).cpu().numpy()
    sub = pd.DataFrame({"id": test_ids.astype(int), "label": kp.astype(int)})
    assert len(sub) == len(SAMPLE_SUB) and (sub["id"].values == SAMPLE_SUB["id"].values).all() and sub["label"].between(0, 9).all()
    tag = f"soup3_headkd_e{int(b.epoch)}_val{b.val:.4f}_test{b.internal_test:.4f}"
    torch.save({"model": stu.state_dict(), "cfg": CFG, "norm": best["norm"]}, os.path.join(RUN_DIR, f"best_{tag}.pt"))
    sub.to_csv(os.path.join(RUN_DIR, f"submission_{tag}.csv"), index=False)
    os.makedirs(os.path.join(WORK_DIR, "headkd"), exist_ok=True); local = os.path.join(WORK_DIR, "headkd", "submission.csv"); sub.to_csv(local, index=False)
    print(f"DECISION: use it -> {os.path.join(RUN_DIR, f'submission_{tag}.csv')} | Kaggle predictions changed vs soup: {(kp != k0).sum()} of {len(kp)}")
    if globals().get("IN_COLAB"):
        from google.colab import files; files.download(local)
else:
    print("DECISION: keep the 0.917 soup (no epoch beats it on VAL by > 0.2 pt). No file written.")
