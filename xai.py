"""
Explainable AI untuk RetinaSense — Attention Rollout & LeGrad pada DINOv2.

Diadaptasi dari notebook penelitian (versi notebook memakai matplotlib +
val_dataset). Di sini difokuskan untuk SATU citra PIL yang sudah dipraproses
oleh RetinaPredictor, dan mengembalikan gambar (base64 PNG) siap tampil:

    base            : citra setelah praproses (crop border + resize 448)
    <method>.map    : heatmap berwarna (jet) di atas latar gelap
    <method>.overlay: heatmap di-blend ke atas base

Catatan performa: jalur attention-capture memakai implementasi attention
eksplisit (bukan SDPA), dan LeGrad melakukan beberapa backward. Di CPU ini
makan beberapa detik per citra (ViT-L lebih berat) — jadi XAI dihitung
LAZY per citra lewat endpoint /api/explain, bukan untuk seluruh batch.
"""

import base64
import io
import math
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from infer import IMAGENET_MEAN, IMAGENET_STD

try:
    from scipy.ndimage import binary_erosion
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ==============================================================
# Attention capture — buka attn_map yang disembunyikan jalur SDPA
# ==============================================================

def _patched_attn_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
    self.attn_map = attn                         # (B, heads, N, N)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    return self.proj(x)                          # drop di-skip (eval, p=0)


def enable_attention_capture(model):
    for blk in model.backbone.blocks:
        if not hasattr(blk.attn, "_orig_forward"):
            blk.attn._orig_forward = blk.attn.forward
        blk.attn.forward = types.MethodType(_patched_attn_forward, blk.attn)


def disable_attention_capture(model):
    for blk in model.backbone.blocks:
        if hasattr(blk.attn, "_orig_forward"):
            blk.attn.forward = blk.attn._orig_forward
        if hasattr(blk.attn, "attn_map"):
            del blk.attn.attn_map


# ==============================================================
# Metode
# ==============================================================

@torch.no_grad()
def attention_rollout(model, x, head_fuse="mean", add_residual=True):
    _ = model.extract_features(x)                # trigger forward → isi attn_map
    N = model.backbone.blocks[0].attn.attn_map.shape[-1]
    eye = torch.eye(N, device=x.device)
    result = eye
    for blk in model.backbone.blocks:
        A = blk.attn.attn_map[0]                  # (heads, N, N)
        A = A.mean(0) if head_fuse == "mean" else A.max(0).values
        if add_residual:
            A = A + eye
            A = A / A.sum(dim=-1, keepdim=True)
        result = A @ result
    mask = result[0, 1:]                          # baris CLS → patch
    g = int(math.sqrt(mask.numel()))
    m = mask.reshape(g, g)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return F.interpolate(m[None, None], size=x.shape[-2:],
                         mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def legrad(model, x, class_idx=None, starting_depth=None):
    x = x.clone().requires_grad_(True)
    blocks = model.backbone.blocks
    L = len(blocks)
    sd = L // 2 if starting_depth is None else max(0, min(starting_depth, L - 1))

    feat_dim = model.classifier[0].normalized_shape[0]
    use_patch = feat_dim == 2 * model.backbone.embed_dim

    def layer_feature(X):                         # X: (1,N,C) output block (pra final-norm)
        Xn = model.backbone.norm(X)
        cls = Xn[:, 0]
        return torch.cat([cls, Xn[:, 1:].mean(1)], dim=1) if use_patch else cls

    block_outputs = []
    handles = [blk.register_forward_hook(lambda m, i, o: block_outputs.append(o))
               for blk in blocks]
    try:
        with torch.enable_grad():
            logits_final = model.classifier(model.extract_features(x))
            if class_idx is None:
                class_idx = int(logits_final.argmax(1).item())
            attn_maps = [blk.attn.attn_map for blk in blocks]

            accum = None
            for l in range(sd, L):
                score_l = model.classifier(layer_feature(block_outputs[l]))[0, class_idx]
                grad = torch.autograd.grad(score_l, attn_maps[l], retain_graph=True)[0]
                grad = grad.clamp(min=0).mean(1)[:, 0, 1:]   # ReLU, mean head, CLS→patch
                g = int(math.sqrt(grad.shape[-1]))
                m = grad.reshape(1, 1, g, g)
                accum = m if accum is None else accum + m
    finally:
        for h in handles:
            h.remove()

    accum = F.interpolate(accum, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    accum = (accum - accum.min()) / (accum.max() - accum.min() + 1e-8)
    return accum.detach().cpu().numpy(), class_idx


# ==============================================================
# Masking foreground + normalisasi (agar background hitam tak diwarnai)
# ==============================================================

def _denorm(img_t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img_t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _foreground_mask(base, erode_frac=0.03):
    fg = base.mean(-1) > 0.10                     # retina vs latar hitam
    if _HAS_SCIPY:
        it = max(1, int(erode_frac * base.shape[0]))
        fg = binary_erosion(fg, iterations=it)    # kikis tepi → buang cincin batas
    return fg


def _norm_in_mask(m, fg, pct=0):
    if not fg.any():
        fg = np.ones_like(fg)
    v = m[fg]
    mn = np.clip((m - v.min()) / (v.max() - v.min() + 1e-8), 0, 1)
    valid = fg.copy()
    if pct > 0:
        thr = np.percentile(mn[fg], pct)
        valid = valid & (mn >= thr)
    return mn, valid


# ==============================================================
# Render → PNG base64
# ==============================================================

# jet: Blue → Cyan → Green → Yellow → Red (konsisten dgn legend di UI)
_JET = np.array([[29, 78, 216], [6, 182, 212], [34, 197, 94], [234, 179, 8], [239, 68, 68]], dtype=np.float32)


def _colormap(v):
    """v: HxW dalam [0,1] → HxWx3 uint8 (jet)."""
    x = np.clip(v, 0, 1) * (len(_JET) - 1)
    i = np.floor(x).astype(int)
    i = np.clip(i, 0, len(_JET) - 2)
    f = (x - i)[..., None]
    c = _JET[i] * (1 - f) + _JET[i + 1] * f
    return c.astype(np.uint8)


def _to_datauri(arr_uint8):
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _render(base, mn, valid, bg=(15, 23, 42), alpha=0.55):
    """base: HxWx3 float[0,1]; mn: HxW value; valid: HxW bool.
    Kembalikan (map_datauri, overlay_datauri)."""
    color = _colormap(mn)                          # HxWx3 uint8
    base_u8 = (base * 255).astype(np.uint8)
    v3 = valid[..., None]

    # map: warna di area valid, latar gelap di luar
    bg_arr = np.array(bg, dtype=np.uint8)
    heat_map = np.where(v3, color, bg_arr)

    # overlay: blend warna ke base, intensitas blend ∝ nilai
    a = (mn * alpha)[..., None] * valid[..., None]
    over = (base_u8 * (1 - a) + color * a).astype(np.uint8)

    return _to_datauri(heat_map), _to_datauri(over)


# ==============================================================
# Entry point
# ==============================================================

def explain(predictor, pil_image, class_idx=None, pct=60,
            methods=("attention_rollout", "legrad")):
    """Hitung XAI untuk satu citra PIL. Kembalikan dict berisi data-URI PNG."""
    model = predictor.model
    model.eval()
    enable_attention_capture(model)
    try:
        x = predictor._to_tensor(pil_image).unsqueeze(0).to(predictor.device)
        base = _denorm(x[0])
        fg = _foreground_mask(base)

        out = {"base": _to_datauri((base * 255).astype(np.uint8)), "methods": {}}
        pred_id = None

        if "attention_rollout" in methods:
            roll = attention_rollout(model, x)
            mn, valid = _norm_in_mask(roll, fg, pct)
            mp, ov = _render(base, mn, valid)
            out["methods"]["attention_rollout"] = {"map": mp, "overlay": ov}

        if "legrad" in methods:
            lg_raw, pred_id = legrad(model, x, class_idx=class_idx)
            mn, valid = _norm_in_mask(lg_raw, fg, pct)
            mp, ov = _render(base, mn, valid)
            out["methods"]["legrad"] = {"map": mp, "overlay": ov}

        out["explain_class"] = class_idx if class_idx is not None else pred_id
        return out
    finally:
        disable_attention_capture(model)
