# ---- OPTION 3: fine-tune the 0.915 model with a 4-model TEACHER ENSEMBLE (SE-WRN + 0.915 + ft100 + tail50) ----
# Paste in the same notebook AFTER the soup cell and the BN-recompute cell (uses their variables: model code, data,
# Xtr, train_augment, flip_probs, ...). Same recipe as the 0.915 run: SLA+SD, KD alpha 0.9, ASAM 0.5, EMA 0.999,
# TrivialAug + crop/flip + cutout + CutMix 50%, label smoothing 0.1, SGD-Nesterov, cosine. Only the teacher changes.
# GATE: stops before training unless the teacher ensemble beats the soup on VAL by > 0.2 pt.
import os, math, time, copy, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

for _n in ["Xtr", "tr_idx", "y_full", "train_augment", "flip_probs", "build_model", "forward_both", "normalize", "CFG", "best", "ROOT", "BASE",
           "Xva", "yva", "Xte_int", "yte_int", "device", "AMP_DTYPE", "MEAN", "STD"]:
    assert _n in globals(), f"missing {_n}: run the soup cell and the BN-recompute cell first in this notebook"

ytr = torch.from_numpy(y_full[tr_idx]).to(device)                        # labels of the 45k train images (tr_idx from the BN cell)
RUN_NAME = BASE + "_ing3kd"; OUT = os.path.join(ROOT, "runs", RUN_NAME); os.makedirs(OUT, exist_ok=True)
EPOCHS, LR, SEED = 20, 0.01, 31
GATE_VAL = 0.9888                                   # soup val 0.9868 + 0.2 pt
TEACHERS = ["teacher_sewrn28_10_sla250_v1", BASE, BASE + "_ft100_lr0.02", BASE + "_tail50_noheavyaug"]
BS, N_ROT = CFG["BATCH_SIZE"], CFG.get("SLA_ROTATIONS", 4)
KD_ALPHA, KD_T, SD_W = CFG["KD_ALPHA"], CFG["KD_T"], CFG.get("SLA_SD_WEIGHT", 1.0)
RHO, ASAM_ETA = CFG["SAM_RHO"], 0.01
MIX_P, CM_ALPHA = CFG.get("MIX_PROB", CFG.get("CUTMIX_P", 0.0)), CFG.get("CUTMIX_ALPHA", 1.0)
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"recipe: KD alpha {KD_ALPHA}, T {KD_T} | SD weight {SD_W} | ASAM rho {RHO} (adaptive {CFG.get('SAM_ADAPTIVE')}) | EMA {CFG['EMA_DECAY']} "
      f"| CutMix {MIX_P} | LS {CFG['LABEL_SMOOTHING']} | wd {CFG['WEIGHT_DECAY']} | batch {BS} | {EPOCHS} epochs, LR {LR}")

# ---- 1. teachers: load, calibrate each on VAL (temperature), equal-weight average, one ensemble temperature ----
@torch.no_grad()
def logits_of(m, X, flip=False, bs=500):
    out = []
    for i in range(0, len(X), bs):
        xb = normalize(X[i:i + bs]); xb = xb.flip(3) if flip else xb
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None): out.append(m(xb).float())
    return torch.cat(out)
def fit_T(lp, y, lo=0.1, hi=10.0):
    grid = torch.exp(torch.linspace(math.log(lo), math.log(hi), 300, device=lp.device))
    return float(grid[torch.stack([F.cross_entropy(lp / t, y) for t in grid]).argmin()])

teachers, Ts, pv, pt = [], [], [], []
for rn in TEACHERS:
    ck = torch.load(os.path.join(ROOT, "runs", rn, "best.pt"), map_location=device, weights_only=False)
    t = build_model(ck["cfg"]).to(device).to(memory_format=torch.channels_last); t.load_state_dict(ck["model"]); t.eval()
    for p in t.parameters(): p.requires_grad_(False)
    T = fit_T(logits_of(t, Xva), yva)
    v = 0.5 * ((logits_of(t, Xva) / T).softmax(1) + (logits_of(t, Xva, True) / T).softmax(1))
    s = 0.5 * ((logits_of(t, Xte_int) / T).softmax(1) + (logits_of(t, Xte_int, True) / T).softmax(1))
    teachers.append(t); Ts.append(T); pv.append(v); pt.append(s)
    print(f"  teacher {rn[-45:]:45s} T {T:.3f} | val {(v.argmax(1) == yva).float().mean():.4f} | internal test {(s.argmax(1) == yte_int).float().mean():.4f}")
ens_v, ens_t = sum(pv) / len(pv), sum(pt) / len(pt)
ENS_T = fit_T(torch.log(ens_v.clamp_min(1e-12)), yva, 0.3, 3.0)
ev, et = (ens_v.argmax(1) == yva).float().mean().item(), (ens_t.argmax(1) == yte_int).float().mean().item()
print(f"TEACHER ENSEMBLE (equal weights, nothing chosen on val): val {ev:.4f} | internal test {et:.4f} | ensemble T {ENS_T:.3f}")
if ev <= GATE_VAL:
    raise RuntimeError(f"GATE: ensemble val {ev:.4f} is not above {GATE_VAL} (soup val + 0.2 pt). A student copying it is "
                       "unlikely to beat the soup. Stopping before training (nothing lost).")
print("GATE passed -> training")

@torch.no_grad()
def teacher_probs(xb):
    p = 0
    with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
        for t, T in zip(teachers, Ts): p = p + F.softmax(t(xb).float() / T, 1) / len(teachers)
    return F.softmax(torch.log(p.clamp_min(1e-12)) / (ENS_T * KD_T), 1)

# ---- 2. student = the 0.915 weights, same optimiser / schedule / EMA as the soup ingredients ----
model = build_model(CFG).to(device).to(memory_format=torch.channels_last); model.load_state_dict(best["model"])
assert abs((flip_probs(model.eval(), Xva).argmax(1) == yva).float().mean().item() - 0.9852) < 0.0015, "start point is not the 0.915 model"
decay = [p for p in model.parameters() if p.ndim > 1]; no_decay = [p for p in model.parameters() if p.ndim <= 1]
opt = torch.optim.SGD([{"params": decay, "weight_decay": CFG["WEIGHT_DECAY"]}, {"params": no_decay, "weight_decay": 0.0}],
                      lr=LR, momentum=CFG.get("MOMENTUM", 0.9), nesterov=True)
steps = EPOCHS * math.ceil(len(Xtr) / BS); warm = math.ceil(len(Xtr) / BS)
sched = torch.optim.lr_scheduler.SequentialLR(opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warm),
                                                    torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps - warm, 0.0)], [warm])
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(CFG["EMA_DECAY"]), use_buffers=True)
ce = nn.CrossEntropyLoss(label_smoothing=CFG["LABEL_SMOOTHING"])
params = [p for p in model.parameters() if p.requires_grad]

def cutmix(x, y):
    lam = np.random.beta(CM_ALPHA, CM_ALPHA); idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.size(2), x.size(3); r = math.sqrt(1.0 - lam); ch, cw = int(H * r), int(W * r)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2, x1, x2 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw // 2, 0), min(cx + cw // 2, W)
    x = x.clone(); x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    return x, y, y[idx], 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

def sla_loss(xb, yb, mixed, ya, yb2, lam, tp):
    B = len(yb); rot = torch.arange(N_ROT, device=device).repeat_interleave(B)
    xr = torch.cat([torch.rot90(xb, k, dims=(2, 3)) for k in range(N_ROT)]).contiguous(memory_format=torch.channels_last)
    with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
        sd_lo, j_lo = forward_both(model, xr)
    sd_lo, j_lo = sd_lo.float(), j_lo.float()
    lj = (lam * ce(j_lo, ya.repeat(N_ROT) * N_ROT + rot) + (1 - lam) * ce(j_lo, yb2.repeat(N_ROT) * N_ROT + rot)) if mixed \
        else ce(j_lo, yb.repeat(N_ROT) * N_ROT + rot)
    jv = j_lo.view(N_ROT, B, 10, N_ROT); agg = torch.stack([jv[k, :, :, k] for k in range(N_ROT)]).mean(0)
    sd0 = sd_lo[:B]
    sd = F.kl_div(F.log_softmax(sd0, 1), F.softmax(agg.detach(), 1), reduction="batchmean")
    kd = F.kl_div(F.log_softmax(sd0 / KD_T, 1), tp, reduction="batchmean") * KD_T ** 2
    return sd0, lj + SD_W * (KD_ALPHA * kd + (1 - KD_ALPHA) * sd)

# ---- 3. train ----
best_val, hist = -1.0, []
for ep in range(EPOCHS):
    model.train(); t0 = time.time(); perm = torch.randperm(len(Xtr), device=device); tot = 0.0
    for i in range(0, len(Xtr), BS):
        idx = perm[i:i + BS]; yb = ytr[idx]
        xb = train_augment(Xtr[idx]).contiguous(memory_format=torch.channels_last)
        mixed = MIX_P > 0 and np.random.rand() < MIX_P; ya = yb2 = None; lam = 1.0
        if mixed: xb, ya, yb2, lam = cutmix(xb, yb)
        tp = teacher_probs(xb.contiguous(memory_format=torch.channels_last))
        opt.zero_grad(set_to_none=True); _, loss = sla_loss(xb, yb, mixed, ya, yb2, lam, tp); loss.backward()
        with torch.no_grad():                                   # ASAM (Kwon et al. 2021), as in the 0.915 run
            if CFG.get("SAM_ADAPTIVE", True):
                tw = [(p.abs() + ASAM_ETA) if p.grad is not None else None for p in params]
                gn = torch.norm(torch.stack([(t * p.grad).norm() for t, p in zip(tw, params) if t is not None])) + 1e-12
                eps = [t * t * p.grad * (RHO / gn) if t is not None else None for t, p in zip(tw, params)]
            else:
                gn = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])) + 1e-12
                eps = [p.grad * (RHO / gn) if p.grad is not None else None for p in params]
            for p, e in zip(params, eps):
                if e is not None: p.add_(e)
        opt.zero_grad(set_to_none=True); _, loss2 = sla_loss(xb, yb, mixed, ya, yb2, lam, tp); loss2.backward()
        with torch.no_grad():
            for p, e in zip(params, eps):
                if e is not None: p.sub_(e)
        opt.step(); sched.step(); ema.update_parameters(model); tot += loss.item() * len(idx)
    net = ema.module.eval()
    va = (flip_probs(net, Xva).argmax(1) == yva).float().mean().item()
    te = (flip_probs(net, Xte_int).argmax(1) == yte_int).float().mean().item() if (ep + 1) % 5 == 0 or ep + 1 == EPOCHS else None
    if va > best_val:
        best_val = va
        torch.save({"model": net.state_dict(), "cfg": CFG, "epoch": ep + 1, "val_acc": va, "norm": best["norm"], "is_ema": True,
                    "teachers": TEACHERS}, os.path.join(OUT, "best.pt"))
    if (ep + 1) % 5 == 0: torch.save({"model": net.state_dict(), "epoch": ep + 1}, os.path.join(OUT, f"snap_e{ep + 1:03d}.pt"))
    sec = time.time() - t0; hist.append({"epoch": ep + 1, "loss": tot / len(Xtr), "val": va, "internal_test": te, "sec": sec})
    pd.DataFrame(hist).to_csv(os.path.join(OUT, "history.csv"), index=False)
    print(f"ep {ep + 1:2d}/{EPOCHS} | loss {tot / len(Xtr):.4f} | val ema {va:.4f}{' *' if va == best_val else '  '} | "
          f"{sec:.0f}s | eta {(EPOCHS - ep - 1) * sec / 60:.0f} min" + (f" | INTERNAL TEST {te:.4f} (0.915: 0.9812, soup: 0.9824)" if te is not None else ""), flush=True)
print(f"DONE: best val {best_val:.4f} -> {os.path.join(OUT, 'best.pt')}. Now rerun the soup cell (it picks up _ing3kd automatically).")
