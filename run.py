"""UPAR 2027 Track 1 - Pedestrian Attribute Recognition (Codabench entry point, v3).
Averages one or more models (listed in config.json: ConvNeXt and/or CLIP ViT), with mirror-image averaging
and a global cutoff. Needs only torch / torchvision / numpy / Pillow (all in the evaluation image)."""
import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
H, W = 256, 128
BATCH_SIZE = 128

NORMS = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
         "clip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])}

_models = []   # list of (model, mean, std, channels_last)
_dev = None
_bias = 0.0
_pool = ThreadPoolExecutor(max_workers=16)
_seen = 0
_t0 = None


def _build(arch):
    if arch == "clip_vit_b16":
        from clipvit import CLIPViT
        return CLIPViT(img_size=(H, W), n_out=40)
    ctor = {"convnext_tiny": torchvision.models.convnext_tiny,
            "convnext_small": torchvision.models.convnext_small}[arch]
    m = ctor(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 40)
    return m


def load_model():
    global _models, _dev, _bias
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    _models = []
    for ent in cfg["models"]:
        m = _build(ent["arch"])
        sd = torch.load(os.path.join(HERE, ent["weights"]), map_location="cpu", weights_only=True)
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        m.eval().to(_dev)
        cl = _dev == "cuda" and not ent["arch"].startswith("clip")
        if cl:
            m = m.to(memory_format=torch.channels_last)
        mean, std = NORMS[ent.get("norm", "imagenet")]
        _models.append((m,
                        torch.tensor(mean, device=_dev).view(1, 3, 1, 1),
                        torch.tensor(std, device=_dev).view(1, 3, 1, 1),
                        cl))
    thr = float(np.clip(float(cfg.get("threshold", 0.5)), 0.05, 0.95))
    _bias = float(np.log(thr / (1.0 - thr)))  # "probability > thr" is the same as "logit > bias"
    print(f"[run.py] {len(_models)} model(s) loaded on {_dev}, cutoff {thr}", flush=True)


def _load(path):
    try:
        img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:  # never crash the whole run on one bad image
        print(f"[run.py] could not read {path}: {e}", flush=True)
        return np.full((H, W, 3), 128, dtype=np.uint8)


@torch.inference_mode()
def predict_batch(samples):
    global _seen, _t0
    if not _models:
        load_model()
    if _t0 is None:
        _t0 = time.time()
    arrs = list(_pool.map(_load, [s["image_path"] for s in samples]))
    x01 = torch.from_numpy(np.stack(arrs)).to(_dev).permute(0, 3, 1, 2).float() / 255.0
    total = 0
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(_dev == "cuda")):
        for m, mean, std, cl in _models:
            x = ((x01 - mean) / std).contiguous()
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
            xf = torch.flip(x, dims=[3])
            total = total + (m(x).float() + m(xf).float()) / 2.0
    logits = total / len(_models)
    probs = torch.sigmoid(logits - _bias)
    probs = torch.nan_to_num(probs, nan=0.5).clamp(0.0, 1.0)
    _seen += len(samples)
    if _seen % 2048 < len(samples):
        print(f"[run.py] {_seen} images, {time.time() - _t0:.1f}s", flush=True)
    return probs.cpu().numpy().tolist()
