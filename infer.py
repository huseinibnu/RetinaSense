"""
Inferensi lokal — DINOv2 gradual fine-tuning.

Dipakai dua cara:

    # 1. sebagai pustaka
    from infer import RetinaPredictor
    p = RetinaPredictor("checkpoints/dinov2_aptos_stage2b_best.pth", dataset="aptos")
    print(p.predict("foto.jpg"))

    # 2. sebagai CLI, satu berkas atau satu folder
    python infer.py --dataset aptos \
                    --checkpoint checkpoints/dinov2_aptos_stage2b_best.pth \
                    --input data/test_images/ --out hasil.csv

Praproses di sini HARUS sama persis dengan notebook. Kalau berbeda,
hasilnya meleset tanpa memunculkan error apa pun.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ==============================================================
# PROFIL DATASET
# ==============================================================

PROFILES = {
    "aptos": {
        "task_type": "ordinal",
        "modality": "fundus",
        "image_size": 448,
        "classes": ["anodr", "bmilddr", "cmoderatedr", "dseveredr", "eproliferativedr"],
        "display": ["Tidak ada DR", "DR ringan", "DR sedang", "DR berat", "DR proliferatif"],
        # FALLBACK saja. Sumber kebenaran thresholds ada DI DALAM checkpoint
        # (key "thresholds"), karena nilainya beda per-model (ViT-B vs ViT-L)
        # dan berubah tiap retrain. RetinaPredictor akan menimpanya otomatis
        # bila checkpoint menyimpannya. Nilai di bawah hanya dipakai kalau
        # checkpoint lama belum punya "thresholds".
        "thresholds": [0.7818, 1.8557, 2.3243, 2.977],
        "refer_from": 2,
    },
    "octid": {
        "task_type": "categorical",
        "modality": "oct",
        "image_size": 448,
        "classes": ["ANormal", "ARMD", "CSR", "Diabetic_retinopathy", "Macular_Hole"],
        "display": ["Normal", "ARMD", "CSR", "Retinopati diabetik", "Macular hole"],
        "thresholds": None,
        "refer_from": None,
    },
}


# ==============================================================
# ARSITEKTUR — identik dengan notebook
# ==============================================================

class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, num_classes, dropout=0.1, use_patch_tokens=True):
        super().__init__()
        self.backbone = backbone
        self.use_patch_tokens = use_patch_tokens

        feat_dim = backbone.embed_dim * (2 if use_patch_tokens else 1)
        self.feat_dim = feat_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def extract_features(self, x):
        out = self.backbone.forward_features(x)
        cls = out["x_norm_clstoken"]
        if not self.use_patch_tokens:
            return cls
        return torch.cat([cls, out["x_norm_patchtokens"].mean(dim=1)], dim=1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))


class CropBlackBorders:
    """Buang bingkai hitam fundus. TIDAK dipakai untuk OCT B-scan."""

    def __init__(self, tol=7):
        self.tol = tol

    def __call__(self, img):
        gray = np.array(img.convert("L"))
        mask = gray > self.tol
        if mask.sum() == 0:
            return img
        ys, xs = np.where(mask)
        return img.crop((int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1))


# ==============================================================
# PREDICTOR
# ==============================================================

class RetinaPredictor:
    """Muat model SEKALI, lalu panggil predict() sebanyak yang perlu.

    Membuat ulang objek ini per permintaan adalah kesalahan paling
    mahal yang bisa dilakukan: memuat backbone + bobot makan 5-15
    detik, sementara inferensinya sendiri puluhan milidetik.
    """

    def __init__(self, checkpoint, dataset="aptos", device=None,
                 amp=True, compile_model=False, cpu_threads=None,
                 backbone_name="dinov2_vitb14",
                 thresholds=None, refer_from=None):
        if dataset not in PROFILES:
            raise ValueError(f"dataset tidak dikenal: {dataset}")

        self.cfg = dict(PROFILES[dataset])
        self.dataset = dataset
        self.backbone_name = backbone_name

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.amp = amp and self.device.type == "cuda"

        if self.device.type == "cuda":
            # TF32: matmul lebih cepat di Ampere ke atas, selisih
            # numeriknya tidak berarti untuk inferensi.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Ukuran input tetap -> biarkan cuDNN memilih algoritma tercepat.
            torch.backends.cudnn.benchmark = True
        elif cpu_threads:
            torch.set_num_threads(cpu_threads)

        # V1 = ViT-B/14 (dinov2_vitb14), V2 = ViT-L/14 (dinov2_vitl14).
        # embed_dim menyesuaikan otomatis (768 vs 1024), jadi classifier
        # ikut menyesuaikan lewat backbone.embed_dim.
        backbone = torch.hub.load("facebookresearch/dinov2", backbone_name,
                                  verbose=False)
        self.model = DINOv2Classifier(backbone, num_classes=len(self.cfg["classes"]))

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        self.model.to(self.device).eval()

        # Simpan metadata checkpoint -- berguna untuk kartu model.
        self.checkpoint_path = str(checkpoint)
        self.epoch = int(ckpt.get("epoch", 0))
        self.val_metrics = {
            k: float(v) for k, v in ckpt.get("val_metrics", {}).items()
            if isinstance(v, (int, float, np.floating))
        }

        # --- Thresholds ordinal mengikuti MODEL, bukan hardcode global ---
        # Nilai thresholds beda antara ViT-B vs ViT-L dan berubah tiap retrain.
        # Prioritas: checkpoint  >  override per-model (registry app.py)  >  PROFILES.
        #   - checkpoint : paling benar, otomatis ikut retrain (belum tersedia skrg)
        #   - per-model  : hardcode base vs large untuk sementara
        #   - PROFILES   : fallback terakhir
        self.thresholds_source = "default"
        if self.cfg.get("task_type") == "ordinal":
            if ckpt.get("thresholds") is not None:
                self.cfg["thresholds"] = [float(t) for t in ckpt["thresholds"]]
                self.thresholds_source = "checkpoint"
            elif thresholds is not None:
                self.cfg["thresholds"] = [float(t) for t in thresholds]
                self.thresholds_source = "model"

            if ckpt.get("refer_from") is not None:
                self.cfg["refer_from"] = int(ckpt["refer_from"])
            elif refer_from is not None:
                self.cfg["refer_from"] = int(refer_from)
        self.thresholds = self.cfg.get("thresholds")
        self.refer_from = self.cfg.get("refer_from")

        if compile_model:
            # Kompilasi pertama makan ~1 menit, sesudahnya 10-30% lebih
            # cepat. Hanya sepadan kalau prosesnya panjang.
            self.model = torch.compile(self.model)

        self.transform = self._build_transform()
        self.flips = self._tta_flips()

    # ---------- praproses ----------

    def _build_transform(self):
        steps = []
        if self.cfg["modality"] == "fundus":
            steps.append(CropBlackBorders())
        steps += [
            transforms.Resize((self.cfg["image_size"], self.cfg["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        return transforms.Compose(steps)

    def _tta_flips(self):
        # Fundus: kiri/kanan dan atas/bawah sama-sama wajar -> 4 view.
        # OCT   : atas-bawah mustahil secara anatomis -> 2 view.
        return [[3], [2], [2, 3]] if self.cfg["modality"] == "fundus" else [[3]]

    def _to_tensor(self, item):
        if isinstance(item, (str, Path)):
            img = Image.open(item)
        elif isinstance(item, Image.Image):
            img = item
        else:
            raise TypeError(f"tipe masukan tidak didukung: {type(item)}")
        return self.transform(img.convert("RGB"))

    # ---------- inferensi ----------

    @torch.inference_mode()
    def _forward(self, batch, tta):
        """inference_mode lebih ketat daripada no_grad: ia juga mematikan
        pencatatan versi tensor, jadi sedikit lebih cepat dan hemat."""
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            logits = self.model(batch).float()
            n_view = 1
            if tta:
                for dims in self.flips:
                    logits = logits + self.model(torch.flip(batch, dims=dims)).float()
                    n_view += 1
                logits = logits / n_view
        return torch.softmax(logits, dim=1).cpu().numpy(), n_view

    def predict(self, images, tta=True, batch_size=8):
        """Terima satu path/PIL atau daftarnya. Selalu kembalikan list dict."""
        single = not isinstance(images, (list, tuple))
        items = [images] if single else list(images)

        results = []
        t0 = time.perf_counter()

        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            # Batching adalah percepatan terbesar: satu gambar tidak
            # pernah mengisi GPU. Dari 1 ke 8 biasanya 4-6x throughput.
            batch = torch.stack([self._to_tensor(c) for c in chunk]).to(
                self.device, non_blocking=True
            )
            probs, n_view = self._forward(batch, tta)

            for j, p in enumerate(probs):
                src = chunk[j]
                results.append(self._interpret(
                    p, n_view,
                    name=(os.path.basename(str(src)) if isinstance(src, (str, Path)) else None)
                ))

        total_ms = (time.perf_counter() - t0) * 1000.0
        for r in results:
            r["latency_ms_avg"] = round(total_ms / max(len(results), 1), 1)

        return results[0] if single else results

    def _interpret(self, probs, n_view, name=None):
        cfg = self.cfg
        order = np.argsort(probs)[::-1]
        argmax = int(order[0])

        out = {
            "file": name,
            "dataset": self.dataset,
            "task_type": cfg["task_type"],
            "probs": {cfg["classes"][i]: round(float(p), 6) for i, p in enumerate(probs)},
            "argmax": argmax,
            "argmax_label": cfg["display"][argmax],
            "confidence": float(probs[argmax]),
            "margin": float(probs[order[0]] - probs[order[1]]),
            "entropy": float(-np.sum(probs * np.log(probs + 1e-12))),
            "n_view": n_view,
        }

        if cfg["task_type"] == "ordinal":
            expected = float(np.dot(probs, np.arange(len(probs))))
            grade = int(np.digitize(expected, cfg["thresholds"]))
            out.update({
                "expected_grade": round(expected, 4),
                "grade": grade,
                "grade_label": cfg["display"][grade],
                "refer": bool(grade >= cfg["refer_from"]),
            })
        else:
            out["grade"] = argmax
            out["grade_label"] = cfg["display"][argmax]

        return out


# ==============================================================
# CLI
# ==============================================================

def collect_images(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)


def main():
    ap = argparse.ArgumentParser(description="Inferensi retina lokal")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="aptos", choices=list(PROFILES))
    ap.add_argument("--input", required=True, help="berkas citra atau folder")
    ap.add_argument("--out", default=None, help="tulis hasil ke CSV")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--cpu-threads", type=int, default=None)
    args = ap.parse_args()

    files = collect_images(args.input)
    if not files:
        raise SystemExit(f"tidak ada citra di {args.input}")

    print(f"Memuat model ({args.dataset}) ...")
    t0 = time.perf_counter()
    p = RetinaPredictor(args.checkpoint, dataset=args.dataset, device=args.device,
                        compile_model=args.compile, cpu_threads=args.cpu_threads)
    print(f"  siap dalam {time.perf_counter() - t0:.1f}s  device={p.device}  "
          f"epoch={p.epoch}")

    t0 = time.perf_counter()
    results = p.predict(files, tta=not args.no_tta, batch_size=args.batch_size)
    dt = time.perf_counter() - t0

    print(f"\n{len(files)} citra dalam {dt:.1f}s  "
          f"({dt / len(files) * 1000:.0f} ms/citra, {results[0]['n_view']} view)\n")

    for r in results[:20]:
        extra = ""
        if r["task_type"] == "ordinal":
            extra = f"  exp={r['expected_grade']:.2f}  rujuk={'ya' if r['refer'] else 'tidak'}"
        print(f"  {str(r['file'])[:44]:<44} {r['grade_label']:<18} "
              f"conf={r['confidence']:.3f}{extra}")
    if len(results) > 20:
        print(f"  ... dan {len(results) - 20} lainnya")

    if args.out:
        cols = ["file", "grade", "grade_label", "confidence", "entropy"]
        if results[0]["task_type"] == "ordinal":
            cols += ["expected_grade", "refer"]
        cols += list(results[0]["probs"].keys())

        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in results:
                row = {k: r.get(k) for k in cols if k in r}
                row.update(r["probs"])
                w.writerow(row)
        print(f"\nDitulis ke {args.out}")


if __name__ == "__main__":
    main()
