"""
RetinaSense — backend Flask.

Arsitektur, praproses, dan logika TTA TIDAK didefinisikan di sini.
Semuanya diimpor dari infer.py supaya hanya ada satu sumber kebenaran.

Model registry: tiap modality punya 2 varian.
    CFP (Funduscopy / APTOS)   : V1 = ViT-B/14, V2 = ViT-L/14   (ordinal, DR grading)
    OCT (OCTID)                : V1 = ViT-B/14, V2 = ViT-L/14   (categorical, multi-penyakit)

Jalankan:
    pip install -r requirements.txt
    python app.py
    buka http://127.0.0.1:5000

Checkpoint diambil dari CKPT_DIR (default: ./checkpoints). Kalau checkpoint
masih di folder demo lama, jalankan dengan:
    CKPT_DIR="C:/Users/gic12/PycharmProjects/DINO_Demo/checkpoints" python app.py
"""

import io
import json
import os

from PIL import Image
from flask import Flask, jsonify, request, render_template

from infer import PROFILES, RetinaPredictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = r"C:\Users\gic12\PycharmProjects\DINO_Demo\checkpoints"

MAX_UPLOAD_MB = 20
MAX_BATCH = 16

# ==============================================================
# REGISTRY MODEL — key UI -> (dataset, varian backbone, checkpoint)
# ==============================================================

MODELS = {
    "cfp_v1": {
        "modality": "cfp",
        "dataset": "aptos",
        "backbone": "dinov2_vitb14",
        "checkpoint": "dinov2_aptos_stage2b_best.pth",
        "name": "CFP Model V1",
        "subtitle": "Diabetic Retinopathy Grading",
        "arch": "DINOv2 · ViT-B/14",
        "recommended_tta": True,
        # Thresholds ordinal hardcode per-model (sementara, sampai bisa disimpan
        # di checkpoint). ViT-B dari fit_thresholds().
        "thresholds": [0.7818, 1.8557, 2.3243, 2.977],
        "refer_from": 2,
    },
    "cfp_v2": {
        "modality": "cfp",
        "dataset": "aptos",
        "backbone": "dinov2_vitl14",
        "checkpoint": "dinov2_large_aptos_stage2b_best.pth",
        "name": "CFP Model V2",
        "subtitle": "Diabetic Retinopathy Grading (Large)",
        "arch": "DINOv2 · ViT-L/14",
        "recommended_tta": True,
        # TODO: ganti dengan hasil fit_thresholds() untuk ViT-L (nilai berbeda
        # dari ViT-B). Sementara memakai nilai ViT-B sebagai placeholder.
        "thresholds": [1.0328, 1.649, 2.2655, 2.9468],
        "refer_from": 2,
    },
    "oct_v1": {
        "modality": "oct",
        "dataset": "octid",
        "backbone": "dinov2_vitb14",
        "checkpoint": "dinov2_octid_stage2b_best.pth",
        "name": "OCT Model V1",
        "subtitle": "Klasifikasi Penyakit Retina (OCT)",
        "arch": "DINOv2 · ViT-B/14",
        "recommended_tta": True,
    },
    "oct_v2": {
        "modality": "oct",
        "dataset": "octid",
        "backbone": "dinov2_vitl14",
        "checkpoint": "dinov2_large_octid_stage2b_best.pth",
        "name": "OCT Model V2",
        "subtitle": "Klasifikasi Penyakit Retina (OCT, Large)",
        "arch": "DINOv2 · ViT-L/14",
        "recommended_tta": True,
    },
}

# Label kelas yang ditampilkan di UI (Bahasa Indonesia).
DISPLAY = {
    "aptos": ["Tidak ada DR", "DR ringan", "DR sedang", "DR berat", "DR proliferatif"],
    "octid": ["Normal", "ARMD", "CSR", "Retinopati diabetik", "Macular hole"],
}

DETAIL = {
    "aptos": [
        "Tidak tampak tanda retinopati diabetik",
        "NPDR ringan — mikroaneurisma saja",
        "NPDR sedang",
        "NPDR berat",
        "Retinopati diabetik proliferatif",
    ],
    "octid": [
        "Tidak ada patologi terdeteksi",
        "Age-related macular degeneration",
        "Central serous retinopathy",
        "Edema makula diabetik pada OCT",
        "Defek fovea seluruh ketebalan",
    ],
}


# ==============================================================
# REGISTRY — model dimuat saat pertama dipakai, lalu ditahan
# ==============================================================

_loaded = {}


def ckpt_path(model_key):
    return os.path.join(CKPT_DIR, MODELS[model_key]["checkpoint"])


def get_predictor(model_key):
    if model_key in _loaded:
        return _loaded[model_key]

    cfg = MODELS[model_key]
    path = ckpt_path(model_key)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {path}. "
            f"Salin file .pth ke folder '{CKPT_DIR}' atau set CKPT_DIR."
        )

    print(f"[{model_key}] memuat {cfg['checkpoint']} ({cfg['backbone']}) ...")
    p = RetinaPredictor(path, dataset=cfg["dataset"], backbone_name=cfg["backbone"],
                        thresholds=cfg.get("thresholds"), refer_from=cfg.get("refer_from"))
    _loaded[model_key] = p
    print(f"[{model_key}] siap. device={p.device} epoch={p.epoch}")
    return p


def load_card_file(name):
    """Baca model_card_<name>.json dari CKPT_DIR, lalu fallback ke root folder."""
    for base in (CKPT_DIR, BASE_DIR):
        path = os.path.join(base, f"model_card_{name}.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def resolve_card(key):
    """key bisa model (cfp_v1) atau dataset (aptos).
    Untuk model: coba kartu spesifik model dulu, lalu kartu dataset."""
    if key in MODELS:
        return load_card_file(key) or load_card_file(MODELS[key]["dataset"])
    return load_card_file(key)


# ==============================================================
# FLASK
# ==============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ---------- halaman ----------

@app.get("/")
def page_dashboard():
    return render_template("dashboard.html", active="dashboard")


@app.get("/analysis")
def page_analysis():
    return render_template("analysis.html", active="analysis")


@app.get("/history")
def page_history():
    return render_template("history.html", active="history")


@app.get("/dataset")
def page_dataset():
    return render_template("dataset.html", active="dataset")


@app.get("/model-info")
def page_model_info():
    return render_template("model_info.html", active="model_info")


@app.get("/help")
def page_help():
    return render_template("help.html", active="help")


# ---------- API ----------

@app.get("/api/models")
def api_models():
    out = {}
    for key, cfg in MODELS.items():
        ds = cfg["dataset"]
        prof = PROFILES[ds]
        n_view = 4 if cfg["modality"] == "cfp" else 2
        out[key] = {
            "name": cfg["name"],
            "subtitle": cfg["subtitle"],
            "arch": cfg["arch"],
            "modality": cfg["modality"],
            "dataset": ds,
            "task_type": prof["task_type"],
            "image_size": prof["image_size"],
            "classes": prof["classes"],
            "display": DISPLAY[ds],
            "detail": DETAIL[ds],
            "tta_views": n_view,
            "recommended_tta": cfg["recommended_tta"],
            "available": os.path.exists(ckpt_path(key)),
            "loaded": key in _loaded,
            "checkpoint": cfg["checkpoint"],
            "has_card": resolve_card(key) is not None,
        }

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return jsonify({"models": out, "device": device})


@app.get("/api/card/<key>")
def api_card(key):
    if key not in MODELS and key not in PROFILES:
        return jsonify({"error": "model/dataset tidak dikenal"}), 404
    card = resolve_card(key)
    if card is None:
        return jsonify({"error": f"model_card_{key}.json belum ada."}), 404
    return jsonify(card)


def _run_one(model_key, file_storage, use_tta):
    """Jalankan inferensi untuk satu berkas, kembalikan dict hasil (atau error)."""
    cfg = MODELS[model_key]
    ds = cfg["dataset"]
    prof = PROFILES[ds]

    try:
        img = Image.open(io.BytesIO(file_storage.read()))
        img.load()
    except Exception:
        return {"filename": file_storage.filename, "error": "berkas bukan citra yang bisa dibaca"}

    p = get_predictor(model_key)
    r = p.predict(img, tta=use_tta)

    out = {
        "filename": file_storage.filename,
        "model": model_key,
        "modality": cfg["modality"],
        "dataset": ds,
        "task_type": r["task_type"],
        "classes": prof["classes"],
        "display": DISPLAY[ds],
        "detail": DETAIL[ds],
        "probs": [r["probs"][c] for c in prof["classes"]],
        "argmax": r["argmax"],
        "confidence": r["confidence"],
        "margin": r["margin"],
        "entropy": r["entropy"],
        "n_view": r["n_view"],
        "latency_ms": r["latency_ms_avg"],
        "grade": r["grade"],
        # XAI dihitung lazy lewat /api/explain saat citra dilihat.
        "explainability": {"mock": False, "methods": ["attention_rollout", "legrad"]},
    }

    if r["task_type"] == "ordinal":
        out.update({
            "expected_grade": r["expected_grade"],
            # thresholds per-model (dari checkpoint bila ada), bukan hardcode global
            "thresholds": p.cfg.get("thresholds"),
            "thresholds_source": getattr(p, "thresholds_source", "default"),
            "refer": r["refer"],
            "refer_from": p.cfg.get("refer_from"),
        })

    return out


@app.post("/api/predict")
def api_predict():
    model_key = request.form.get("model", "cfp_v1")
    if model_key not in MODELS:
        return jsonify({"error": f"model tidak dikenal: {model_key}"}), 400
    if "image" not in request.files:
        return jsonify({"error": "tidak ada berkas 'image'"}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "berkas kosong"}), 400

    use_tta = request.form.get("tta", "true").lower() != "false"

    try:
        result = _run_one(model_key, f, use_tta)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.exception("inferensi gagal")
        return jsonify({"error": f"inferensi gagal: {e}"}), 500

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.post("/api/predict_batch")
def api_predict_batch():
    model_key = request.form.get("model", "cfp_v1")
    if model_key not in MODELS:
        return jsonify({"error": f"model tidak dikenal: {model_key}"}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "tidak ada berkas 'images'"}), 400
    if len(files) > MAX_BATCH:
        return jsonify({"error": f"maksimum {MAX_BATCH} citra per batch"}), 400

    use_tta = request.form.get("tta", "true").lower() != "false"

    import time
    t0 = time.perf_counter()
    results = []
    try:
        for f in files:
            if not f.filename:
                continue
            results.append(_run_one(model_key, f, use_tta))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.exception("inferensi batch gagal")
        return jsonify({"error": f"inferensi gagal: {e}"}), 500

    total_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return jsonify({
        "status": "success",
        "model": model_key,
        "tta": use_tta,
        "count": len(results),
        "processing_ms": total_ms,
        "results": results,
    })


@app.post("/api/explain")
def api_explain():
    """Explainable AI untuk SATU citra: Attention Rollout + LeGrad.
    Dihitung lazy (per citra yang sedang dilihat), bukan untuk seluruh batch."""
    model_key = request.form.get("model", "cfp_v1")
    if model_key not in MODELS:
        return jsonify({"error": f"model tidak dikenal: {model_key}"}), 400
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "tidak ada berkas 'image'"}), 400

    class_idx = request.form.get("class_idx")
    class_idx = int(class_idx) if (class_idx is not None and class_idx != "") else None
    try:
        pct = int(request.form.get("pct", 60))
    except ValueError:
        pct = 60

    methods = request.form.get("methods", "attention_rollout,legrad").split(",")
    methods = [m.strip() for m in methods if m.strip()]

    f = request.files["image"]
    try:
        img = Image.open(io.BytesIO(f.read()))
        img.load()
    except Exception:
        return jsonify({"error": "berkas bukan citra yang bisa dibaca"}), 400

    try:
        import time
        import xai
        p = get_predictor(model_key)
        t0 = time.perf_counter()
        out = xai.explain(p, img, class_idx=class_idx, pct=pct, methods=methods)
        out["filename"] = f.filename
        out["processing_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return jsonify(out)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.exception("XAI gagal")
        return jsonify({"error": f"XAI gagal: {e}"}), 500


@app.get("/api/health")
def api_health():
    import torch
    return jsonify({
        "ok": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "ckpt_dir": CKPT_DIR,
        "loaded": list(_loaded.keys()),
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"berkas melebihi {MAX_UPLOAD_MB} MB"}), 413


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Checkpoint dir: {CKPT_DIR}")
    for key, cfg in MODELS.items():
        p = ckpt_path(key)
        status = "ADA   " if os.path.exists(p) else "HILANG"
        print(f"  {key:8s} {status}  {cfg['checkpoint']}")
    app.run(host="127.0.0.1", port=5000, debug=False)
