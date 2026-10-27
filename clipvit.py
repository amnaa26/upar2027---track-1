"""Minimal CLIP ViT image encoder (OpenAI / open_clip layout) with a 40-way head.
Used for fine-tuning (train.py) and for inference (run.py). Needs only torch."""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d, 4 * d)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(4 * d, d)),
        ]))

    def forward(self, x):  # x: (tokens, batch, width)
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class CLIPViT(nn.Module):
    def __init__(self, img_size=(224, 224), patch=16, width=768, layers=12, heads=12, embed_dim=512, n_out=40):
        super().__init__()
        self.grid = (img_size[0] // patch, img_size[1] // patch)
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros(self.grid[0] * self.grid[1] + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.zeros(width, embed_dim))
        self.head = nn.Linear(width, n_out) if n_out else None

    def features(self, x):
        """CLIP image embedding before the final projection: (batch, width)."""
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).transpose(0, 1)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_post(x.transpose(0, 1)[:, 0])

    def forward(self, x):
        return self.head(self.features(x))

    def param_groups(self, lr, decay, head_lr, wd=0.05):
        """Layer-wise learning rates: early layers change slowly, the head changes fast."""
        n = len(self.blocks)

        def layer_id(name):
            if name.startswith("blocks."):
                return int(name.split(".")[1]) + 1
            if name.startswith(("conv1", "class_embedding", "positional_embedding", "ln_pre")):
                return 0
            return n + 1  # ln_post, proj

        groups = {}
        for name, p in self.named_parameters():
            no_wd = p.ndim <= 1 or "embedding" in name
            this_lr = head_lr if name.startswith("head.") else lr * decay ** (n + 1 - layer_id(name))
            g = groups.setdefault((this_lr, no_wd), {"params": [], "lr": this_lr, "weight_decay": 0.0 if no_wd else wd})
            g["params"].append(p)
        return list(groups.values())


def load_openai_visual(model, sd):
    """Load an OpenAI/open_clip `visual` state dict, resizing the position embedding to the model's patch grid."""
    sd = {k.replace("transformer.resblocks.", "blocks."): v for k, v in sd.items()}
    pe = sd["positional_embedding"].float()
    cls_pe, grid_pe = pe[:1], pe[1:]
    g = int(round(grid_pe.shape[0] ** 0.5))
    gh, gw = model.grid
    if (gh, gw) != (g, g):
        grid_pe = grid_pe.reshape(1, g, g, -1).permute(0, 3, 1, 2)
        grid_pe = F.interpolate(grid_pe, size=(gh, gw), mode="bicubic", align_corners=False)
        grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(gh * gw, -1)
    sd["positional_embedding"] = torch.cat([cls_pe, grid_pe], dim=0)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("head.")]
    assert not missing, f"CLIP weights not loaded for: {missing[:5]}"
    return unexpected
