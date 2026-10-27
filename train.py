"""UPAR 2027 Track 1 - training script v2 (made for Kaggle).

Environment variables (all optional):
  TAG        name of this run (default "run"); results go to BASE/out_<TAG>
  ARCH       convnext_tiny (default) or convnext_small
  AUG        base (default) or strong  (strong = more blur / low resolution / JPEG / brightness)
  POSW       strength of the extra weight on rare "yes" answers: (no/yes)**POSW, default 0.5
  POSCAP     largest allowed extra weight, default 10
  HOLDOUT    market / peta / pa100k : leave that photo collection OUT of training and test on it.
             This imitates "photos from a place the model has never seen" (experiment mode).
  HOLD_N     how many held-out photos to test on (default 10000)
  EPOCHS     passes over the training photos (default 4)
  BATCH      photos per step (default 64; use 32 if you run out of GPU memory)
  SUBSET     if > 0, use only that many photos: a quick test of the whole pipeline
  USE_VAL    1 = also train on the validation photos (allowed; for the final run)
  TUNE_THR   1 = tune a cutoff per attribute on the validation photos (default 0, not trusted under domain shift)
  BASE       folder for outputs (default /kaggle/working)
  DATA_BASE  folder that holds the downloaded starter kit (default = BASE)
"""
import os
import io
import sys
import json
import math
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader

BASE = os.environ.get("BASE", "/kaggle/working")


def _find_data_base():
    """Where the starter kit with the photos lives (the setup cell normally tells us)."""
    env = os.environ.get("DATA_BASE")
    if env:
        return env
    for p in ["/kaggle/temp", "/tmp", BASE]:
        if os.path.isdir(os.path.join(p, "UPAR-Challenge-2027", "data", "annotations")):
            return p
    return BASE


DATA_BASE = _find_data_base()
REPO = os.path.join(DATA_BASE, "UPAR-Challenge-2027")
DATA = os.path.join(REPO, "data")
ANN = os.path.join(DATA, "annotations", "task1")
TAG = os.environ.get("TAG", "run")
OUT = os.path.join(BASE, "out_" + TAG)
ARCH = os.environ.get("ARCH", "convnext_tiny")
AUG = os.environ.get("AUG", "base")
POSW = float(os.environ.get("POSW", 0.5))
POSCAP = float(os.environ.get("POSCAP", 10))
HOLDOUT = os.environ.get("HOLDOUT", "").strip().lower()
HOLD_N = int(os.environ.get("HOLD_N", 10000))
EPOCHS = int(os.environ.get("EPOCHS", 4))
BATCH = int(os.environ.get("BATCH", 64))
SUBSET = int(os.environ.get("SUBSET", 0))
USE_VAL = int(os.environ.get("USE_VAL", 0))
TUNE_THR = int(os.environ.get("TUNE_THR", 0))
SANITY_N = int(os.environ.get("SANITY_N", 0))   # with USE_VAL: keep this many photos aside as a bug check
CLIP_LR = float(os.environ.get("CLIP_LR", 4e-5))
H, W = 256, 128
STRONG = AUG == "strong"
if ARCH.startswith("clip"):
    MEAN, STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
else:
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CONFIG = f"{TAG}: arch={ARCH} aug={AUG} posw={POSW} holdout={HOLDOUT or '-'} epochs={EPOCHS} use_val={USE_VAL}"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "|", CONFIG, flush=True)


# ----------------------------------------------------------------- data
def build_index():
    by2, by1 = {}, {}
    for root, _, files in os.walk(DATA):
        if os.sep + "annotations" in root:
            continue
        parent = os.path.basename(root)
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            p = os.path.join(root, f)
            by2.setdefault(parent + "/" + f, p)
            by1.setdefault(f, []).append(p)
    return by2, by1


def resolve(img, by2, by1):
    img = str(img).replace("\\", "/")
    for cand in (os.path.join(DATA, img), os.path.join(REPO, img)):
        if os.path.exists(cand):
            return cand
    parts = img.split("/")
    if len(parts) >= 2 and (parts[-2] + "/" + parts[-1]) in by2:
        return by2[parts[-2] + "/" + parts[-1]]
    cands = by1.get(parts[-1], [])
    for c in cands:
        if any(seg and seg in c for seg in parts[:-1]):
            return c
    return cands[0] if cands else None


def dataset_of(s):
    s = s.lower()
    if "market" in s:
        return "market"
    if "pa100k" in s or "pa-100k" in s or "release_data" in s:
        return "pa100k"
    if "peta" in s:
        return "peta"
    return "other"


def read_split(split, by2, by1):
    df = pd.read_csv(os.path.join(ANN, split, "gt.csv"))
    cols = list(df.columns)
    df.columns = ["image"] + cols[1:]
    names = cols[1:]
    assert len(names) == 40, f"expected 40 attributes, found {len(names)}"
    paths = [resolve(i, by2, by1) for i in df["image"]]
    bad = [i for i, p in zip(df["image"], paths) if p is None]
    print(f"{split}: {len(df)} rows, {len(bad)} images not found", flush=True)
    if len(bad) > 0.01 * len(df):
        print("Examples of rows in gt.csv:", list(df["image"][:3]), flush=True)
        print("Examples of missing:", bad[:3], flush=True)
        raise SystemExit("Too many images not found.")
    keep = [i for i, p in enumerate(paths) if p is not None]
    arr = df.iloc[keep, 1:].values.astype(np.float32)
    mask = ((arr == 0) | (arr == 1)).astype(np.float32)  # anything else = "unknown"
    y = np.where(mask > 0, arr, 0).astype(np.float32)
    paths = [paths[i] for i in keep]
    ds = [dataset_of(str(df["image"].iloc[i]) + " " + paths[j]) for j, i in enumerate(keep)]
    return names, paths, y, mask, ds


def degrade(img):
    """Strong augmentation: imitate other cameras (low resolution, blur, JPEG compression)."""
    w, h = img.size
    if random.random() < 0.5:
        f = random.uniform(0.25, 0.8)
        img = img.resize((max(8, int(w * f)), max(16, int(h * f))),
                         random.choice([Image.BILINEAR, Image.BICUBIC, Image.BOX]))
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if random.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(25, 85))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


class PedDS(Dataset):
    def __init__(self, paths, y, m, train, strong=False):
        self.paths, self.y, self.m, self.train, self.strong = paths, y, m, train, strong
        if strong:
            self.cj = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.25, hue=0.0)
        else:
            self.cj = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.0)
        self.tn = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
        self.erase = T.RandomErasing(p=0.3 if strong else 0.25, scale=(0.02, 0.15), value=0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (W, H), (128, 128, 128))
        if self.train:
            if self.strong:
                img = degrade(img)
                pw, ph = random.randint(0, 16), random.randint(0, 32)
            else:
                if random.random() < 0.3:  # imitate blurry / low-resolution cameras
                    f = random.uniform(0.35, 0.8)
                    w, h = img.size
                    img = img.resize((max(8, int(w * f)), max(16, int(h * f))), Image.BILINEAR)
                pw, ph = random.randint(0, 12), random.randint(0, 24)
            img = img.resize((W + pw, H + ph), Image.BILINEAR)
            x0, y0 = random.randint(0, pw), random.randint(0, ph)
            img = img.crop((x0, y0, x0 + W, y0 + H))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.cj(img)
            t = self.erase(self.tn(img))
        else:
            t = self.tn(img.resize((W, H), Image.BILINEAR))
        return t, torch.from_numpy(self.y[i]), torch.from_numpy(self.m[i])


# ----------------------------------------------------------------- model
def make_model(arch):
    if arch == "convnext_tiny":
        m = torchvision.models.convnext_tiny(weights="DEFAULT")  # ImageNet weights (allowed)
    elif arch == "convnext_small":
        m = torchvision.models.convnext_small(weights="DEFAULT")
    elif arch == "clip_vit_b16":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from clipvit import CLIPViT, load_openai_visual
        m = CLIPViT(img_size=(H, W), n_out=40)
        sd = torch.load(os.path.join(BASE, "clip_b16_visual.pt"), map_location="cpu")
        extra = load_openai_visual(m, {k: v.float() for k, v in sd.items()})
        m.proj.requires_grad_(False)
        print("CLIP weights loaded; unused keys:", list(extra)[:5], flush=True)
        return m
    else:
        raise SystemExit(f"unknown ARCH {arch}")
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 40)
    return m


@torch.no_grad()
def predict_logits(model, loader, tta=True):
    model.eval()
    outs = []
    for x, _, _ in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            lg = model(x).float()
            if tta:
                lg = (lg + model(torch.flip(x, dims=[3])).float()) / 2
        outs.append(lg.cpu())
    return torch.cat(outs).numpy()


# ----------------------------------------------------------------- metrics (approximation of the official ones)
def challenge_metrics(prob, y, m, ds, thr=0.5):
    p_all = prob > thr
    g_all = y > 0.5
    k_all = m > 0.5
    res = []
    for d in sorted(set(ds)):
        idx = np.array([n == d for n in ds])
        p, g, k = p_all[idx], g_all[idx], k_all[idx]
        accs = []
        for j in np.where(k.any(0))[0]:
            v = k[:, j]
            pj, gj = p[v, j], g[v, j]
            tp, fn = (pj & gj).sum(), (~pj & gj).sum()
            tn, fp = (~pj & ~gj).sum(), (pj & ~gj).sum()
            tnr = tn / max(tn + fp, 1)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else tnr
            accs.append((tpr + tnr) / 2)
        mA = float(np.mean(accs))
        pa, ga = p & k, g & k
        inter, ps, gs = (pa & ga).sum(1), pa.sum(1), ga.sum(1)
        prec = np.where(ps > 0, inter / np.maximum(ps, 1), 0.0)
        rec = np.where(gs > 0, inter / np.maximum(gs, 1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)
        res.append((mA, float(f1.mean()), float(prec.mean()), float(rec.mean())))
    mA = float(np.mean([r[0] for r in res]))
    iF1 = float(np.mean([r[1] for r in res]))
    return {"mA": mA, "F1": iF1, "Avg": 2 * mA * iF1 / max(mA + iF1, 1e-9),
            "P": float(np.mean([r[2] for r in res])), "R": float(np.mean([r[3] for r in res]))}


def sweep(prob, y, m, ds, label):
    """Print the scores for a range of global cutoffs, and remember them in results.txt."""
    lines = []
    best = (-1, 0.5)
    for t in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]:
        r = challenge_metrics(prob, y, m, ds, thr=t)
        lines.append(f"RESULT {label} cutoff={t:.2f} Avg={r['Avg']:.4f} mA={r['mA']:.4f} "
                     f"InstF1={r['F1']:.4f} Prec={r['P']:.4f} Rec={r['R']:.4f}")
        if r["Avg"] > best[0]:
            best = (r["Avg"], t)
    lines.append(f"RESULT {label} BEST cutoff={best[1]:.2f} Avg={best[0]:.4f}")
    print("\n".join(lines), flush=True)
    with open(os.path.join(BASE, "results.txt"), "a") as f:
        f.write("\n".join(lines) + "\n")
    return best[1]


def calibrate(prob, y, m):
    """Optional: pick a decision threshold per attribute on the validation photos."""
    grid = np.arange(0.25, 0.751, 0.05)
    thr = np.full(40, 0.5)
    g_all, k_all = y > 0.5, m > 0.5
    for j in range(40):
        v = k_all[:, j]
        if v.sum() == 0 or g_all[v, j].sum() == 0:
            continue
        gj = g_all[v, j]
        best, bt = -1, 0.5
        for t in grid:
            pj = prob[v, j] > t
            tp, fn = (pj & gj).sum(), (~pj & gj).sum()
            tn, fp = (~pj & ~gj).sum(), (pj & ~gj).sum()
            ba = (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            sc = (ba + f1) / 2
            if sc > best:
                best, bt = sc, float(t)
        thr[j] = bt
    return thr


# ----------------------------------------------------------------- main
def main():
    t_start = time.time()
    print("indexing image files...", flush=True)
    by2, by1 = build_index()
    names, tr_p, tr_y, tr_m, tr_d = read_split("train", by2, by1)
    _, va_p, va_y, va_m, va_d = read_split("val", by2, by1)

    if HOLDOUT:
        all_p, all_y, all_m, all_d = tr_p + va_p, np.concatenate([tr_y, va_y]), np.concatenate([tr_m, va_m]), tr_d + va_d
        hold = np.array([d == HOLDOUT for d in all_d])
        if not hold.any():
            raise SystemExit(f"HOLDOUT={HOLDOUT} matches no photos; use market, peta or pa100k")
        rs = np.random.RandomState(1)
        hidx = rs.permutation(np.where(hold)[0])[:HOLD_N]
        fidx = np.where(~hold)[0]
        fit_p, fit_y, fit_m = [all_p[i] for i in fidx], all_y[fidx], all_m[fidx]
        va_p, va_y, va_m, va_d = [all_p[i] for i in hidx], all_y[hidx], all_m[hidx], [all_d[i] for i in hidx]
        print(f"EXPERIMENT: training on {len(fit_p)} photos WITHOUT '{HOLDOUT}', testing on {len(va_p)} '{HOLDOUT}' photos",
              flush=True)
    else:
        if USE_VAL:
            fit_p, fit_y, fit_m = tr_p + va_p, np.concatenate([tr_y, va_y]), np.concatenate([tr_m, va_m])
            if SANITY_N:  # keep a few photos aside, only to notice bugs (not a fair test)
                all_d = tr_d + va_d
                perm = np.random.RandomState(2).permutation(len(fit_p))
                si, ki = perm[:SANITY_N], perm[SANITY_N:]
                va_p, va_y, va_m, va_d = [fit_p[i] for i in si], fit_y[si], fit_m[si], [all_d[i] for i in si]
                fit_p, fit_y, fit_m = [fit_p[i] for i in ki], fit_y[ki], fit_m[ki]
        else:
            fit_p, fit_y, fit_m = tr_p, tr_y, tr_m
    if SUBSET:
        rs = np.random.RandomState(1)
        fi = rs.permutation(len(fit_p))[:SUBSET]
        vi = rs.permutation(len(va_p))[: max(SUBSET // 2, 200)]
        fit_p, fit_y, fit_m = [fit_p[i] for i in fi], fit_y[fi], fit_m[fi]
        va_p, va_y, va_m, va_d = [va_p[i] for i in vi], va_y[vi], va_m[vi], [va_d[i] for i in vi]

    # rare "yes" answers get extra weight (the score treats every attribute equally)
    pos = (fit_y * fit_m).sum(0)
    neg = ((1 - fit_y) * fit_m).sum(0)
    pw = np.clip((neg / np.maximum(pos, 1)) ** POSW, 1.0, POSCAP)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=dev)
    print("training photos:", len(fit_p), "| test photos:", len(va_p), flush=True)

    nw = min(os.cpu_count() or 2, 4)
    tl = DataLoader(PedDS(fit_p, fit_y, fit_m, True, STRONG), batch_size=BATCH, shuffle=True, num_workers=nw,
                    pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    vl = DataLoader(PedDS(va_p, va_y, va_m, False), batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)

    model = make_model(ARCH).to(dev)
    if ARCH.startswith("clip"):
        opt = torch.optim.AdamW(model.param_groups(lr=CLIP_LR, decay=0.75, head_lr=1e-3), betas=(0.9, 0.98), eps=1e-6)
    else:
        if dev == "cuda":
            model = model.to(memory_format=torch.channels_last)
        head = list(model.classifier[2].parameters())
        head_ids = {id(p) for p in head}
        body = [p for p in model.parameters() if id(p) not in head_ids]
        opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
    base_lrs = [g["lr"] for g in opt.param_groups]
    total = EPOCHS * len(tl)
    warm = min(300, total // 10 + 1)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    os.makedirs(OUT, exist_ok=True)
    wpath = os.path.join(OUT, "weights.pt")
    best, step = -1.0, 0
    for ep in range(EPOCHS):
        model.train()
        run, n = 0.0, 0
        t0 = time.time()
        for x, y, mk in tl:
            lrf = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lrf
            x, y, mk = x.to(dev, non_blocking=True), y.to(dev), mk.to(dev)
            if dev == "cuda" and not ARCH.startswith("clip"):
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                lg = model(x)
            loss = (lossf(lg.float(), y) * mk).sum() / mk.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            n += 1
            step += 1
            if n % 200 == 0:
                print(f"  epoch {ep + 1} step {n}/{len(tl)} loss {run / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if USE_VAL and not HOLDOUT and not SANITY_N:
            print(f"== epoch {ep + 1}/{EPOCHS}: loss {run / max(n, 1):.4f} | {time.time() - t0:.0f}s", flush=True)
        else:
            prob = 1 / (1 + np.exp(-predict_logits(model, vl, tta=False)))
            r = challenge_metrics(prob, va_y, va_m, va_d)
            print(f"== epoch {ep + 1}/{EPOCHS}: loss {run / max(n, 1):.4f} | test mA {r['mA']:.4f} instF1 {r['F1']:.4f} "
                  f"Avg {r['Avg']:.4f} | {time.time() - t0:.0f}s", flush=True)
        if not HOLDOUT:
            if USE_VAL or r["Avg"] > best:
                best = -1.0 if USE_VAL else r["Avg"]
                torch.save({k: v.detach().half().cpu() for k, v in model.state_dict().items()}, wpath)
                print("   saved weights.pt", flush=True)

    if HOLDOUT:
        # the final-epoch model on the never-seen photo collection, for a range of cutoffs
        prob = 1 / (1 + np.exp(-predict_logits(model, vl, tta=True)))
        sweep(prob, va_y, va_m, va_d, CONFIG)
    else:
        if not USE_VAL:
            model.load_state_dict({k: v.float().to(dev) for k, v in torch.load(wpath).items()})
        if (not USE_VAL) or SANITY_N:
            prob = 1 / (1 + np.exp(-predict_logits(model, vl, tta=True)))
            sweep(prob, va_y, va_m, va_d, CONFIG + " (same-collection check, optimistic)")
            if TUNE_THR and not USE_VAL:
                thr = calibrate(prob, va_y, va_m)
                json.dump([float(t) for t in thr], open(os.path.join(OUT, "thresholds.json"), "w"))
        json.dump({"arch": ARCH, "norm": "clip" if ARCH.startswith("clip") else "imagenet", "H": H, "W": W,
                   "config": CONFIG}, open(os.path.join(OUT, "model.json"), "w"))
        print("files in", OUT, ":", sorted(os.listdir(OUT)), flush=True)
    print(f"\nDONE in {(time.time() - t_start) / 60:.1f} min  ({CONFIG})", flush=True)


if __name__ == "__main__":
    main()
