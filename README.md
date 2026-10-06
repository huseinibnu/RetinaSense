# RetinaSense

Analisis citra retina berbasis **DINOv2** (CFP/Funduscopy & OCT) dengan penjelasan
visual (Explainable AI). UI baru untuk demo DINO.

**Tech stack:** Flask · HTML · Vanilla JS · Tailwind (CDN) · Lucide icons · Inter.

> ⚠️ **Prototipe penelitian.** Bukan alat bantu diagnosis, belum divalidasi klinis,
> tidak menggantikan penilaian dokter mata.

---

## Menjalankan

```bash
pip install -r requirements.txt

python app.py
# buka http://127.0.0.1:5000
```

Catatan: Flask dijalankan dengan `debug=False`, jadi **template di-cache** — setiap
mengubah HTML/templates, restart server. Perubahan JSON kartu model dibaca live.

---

## Model (registry di `app.py`)

| key      | modality | varian   | checkpoint                            | task        |
|----------|----------|----------|---------------------------------------|-------------|
| `cfp_v1` | CFP      | ViT-B/14 | `dinov2_aptos_stage2b_best.pth`       | ordinal     |
| `cfp_v2` | CFP      | ViT-L/14 | `dinov2_large_aptos_stage2b_best.pth` | ordinal     |
| `oct_v1` | OCT      | ViT-B/14 | `dinov2_octid_stage2b_best.pth`       | categorical |
| `oct_v2` | OCT      | ViT-L/14 | `dinov2_large_octid_stage2b_best.pth` | categorical |

- **CFP / APTOS** → grading retinopati diabetik (ordinal 0–4), output utama flag
  **rujukan (referable DR, grade ≥ 2)**.
- **OCT / OCTID** → klasifikasi 5 penyakit: Normal, ARMD, CSR, Retinopati diabetik, Macular hole.

Model dimuat lazy saat pertama dipakai, lalu ditahan di memori.

---

## Halaman

| Route          | Halaman           | Isi                                                       |
|----------------|-------------------|-----------------------------------------------------------|
| `/`            | Dashboard         | Ringkasan, status model, aktivitas terakhir               |
| `/analysis`    | **Analysis**      | Workflow utama (satu halaman)                             |
| `/history`     | History           | Riwayat analisis (localStorage)                           |
| `/dataset`     | Dataset           | Info APTOS & OCTID                                        |
| `/model-info`  | Model Information  | Kartu per-model: arsitektur, metrik, performa per kelas   |
| `/help`        | Help              | Alur analisis + glosarium istilah (ramah non-teknis)      |

**Alur Analysis:** Modality → Model (V1/V2) → TTA → Upload (1–16 citra) → Analisis →
Hasil (Citra · Prediksi · Penjelasan) + navigasi antar-citra + tab Batch Summary.

---

## API

| Endpoint              | Metode | Keterangan                                                     |
|-----------------------|--------|----------------------------------------------------------------|
| `/api/models`         | GET    | Registry model + device                                        |
| `/api/card/<key>`     | GET    | Kartu model; `key` = model (`cfp_v1`…) atau dataset (`aptos`)   |
| `/api/predict`        | POST   | Inferensi satu citra                                           |
| `/api/predict_batch`  | POST   | Inferensi banyak citra (maks 16)                               |
| `/api/explain`        | POST   | **XAI** satu citra: Attention Rollout + LeGrad (lazy)          |
| `/api/health`         | GET    | Status device & model termuat                                 |

`/api/predict*` form: `model`, `tta`, `image`/`images`.
`/api/explain` form: `model`, `image`, opsional `class_idx`, `pct`, `methods`.
Response XAI berisi PNG base64: `base`, dan `methods.<nama>.{map, overlay}`.

---

## Explainable AI (`xai.py`)

Dua metode, dihitung **lazy per citra** lewat `/api/explain` (bukan untuk seluruh batch):

- **Attention Rollout** — *di mana* model memusatkan perhatian (telusur attention antar-lapisan).
- **LeGrad** — bagian yang paling *memengaruhi keputusan* untuk kelas yang diprediksi (berbasis gradien).

Heatmap dihitung pada citra **setelah praproses** (crop border + resize 448) dan
dikembalikan bersama citra base-nya, sehingga overlay presisi. Panel UI: Asli ·
Heatmap · Overlay, dengan tab metode, penjelasan singkat, dan klik → lightbox.

`scipy` opsional (memperhalus foreground mask); ada fallback bila tidak terpasang.

---

## Struktur folder

```
RetinaSense/
├── app.py                      # Flask: halaman + API + registry model
├── infer.py                    # Praproses + DINOv2Classifier + RetinaPredictor (ditto demo lama)
├── xai.py                      # Attention Rollout & LeGrad → PNG base64
├── requirements.txt
├── templates/
│   ├── base.html               # layout + top navbar (sticky, blur saat scroll)
│   ├── dashboard.html  analysis.html  history.html
│   ├── dataset.html    model_info.html  help.html
└── static/
    ├── css/app.css             # token + navbar + nav underline + heatmap legend
    ├── js/
    │   ├── state.js            # state terpusat + API + history(localStorage)
    │   ├── upload.js           # dropzone + thumbnail
    │   ├── results.js          # prediksi, probabilitas, XAI, lightbox, batch summary
    │   └── analysis.js         # orchestration workflow Analysis
    └── assets/
        ├── icons/              # favicon / logo
        ├── modality/           # cfp.jpg, oct.jpg (kartu modality)
        └── examples/cfp|oct/   # contoh citra per kelas
```

### Kartu model (Model Information)

Letakkan JSON di `checkpoints/` (atau root). Per-model lebih spesifik, dan jatuh
ke per-dataset bila tidak ada:

```
model_card_cfp_v1.json   model_card_cfp_v2.json
model_card_oct_v1.json   model_card_oct_v2.json
model_card_aptos.json    model_card_octid.json   (fallback)
```

Field yang dibaca UI: `dataset`, `task_type`, `backbone`, `image_size`, `stage`,
`intended_use`, `data.classes`, `test.with_tta` (accuracy/bal_acc/f1/auroc/qwk),
`test.per_class` (precision/recall/f1/support), `limitations`.

---

## Status

- ✅ Layout top-navbar modern, 6 halaman
- ✅ Single & batch inference, dukungan ViT-B **dan** ViT-L
- ✅ Prediksi: mode ordinal (grade + flag rujukan) vs categorical (argmax), progress-bar probabilitas
- ✅ **Explainable AI nyata**: Attention Rollout + LeGrad (overlay presisi, lightbox, penjelasan non-teknis)
- ✅ History (localStorage), Model Information per-model dengan performa per kelas

### Penyesuaian dari design-doc awal
- Kelas OCT mengikuti **OCTID** (Normal/ARMD/CSR/Ret. diabetik/Macular hole), bukan DME/CNV.
- APTOS **ordinal** → output utama flag **rujukan (grade ≥ 2)**.
- DICOM belum didukung (PIL) → JPG/PNG saja.
- "Patch Masking" diganti **LeGrad** sebagai metode XAI kedua.
