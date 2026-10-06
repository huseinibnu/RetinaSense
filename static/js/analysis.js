/* analysis.js — mengikat semuanya: render berbasis state, pilih model,
   panggil /api/predict_batch, simpan history, tampilkan hasil. */

(async function () {
  const RS = window.RS, S = RS.state;

  await RS.loadModels();

  const DEFAULT_TTA = { cfp: 4, oct: 2 };

  // ---------- render ----------
  function renderModalityCards() {
    document.querySelectorAll('.modality-card').forEach(card => {
      const on = card.dataset.modality === S.modality;
      card.classList.toggle('border-primary-600', on);
      card.classList.toggle('border-2', on);
      card.classList.toggle('bg-primary-50', on);
      card.classList.toggle('border-line', !on);
      card.querySelector('.check').classList.toggle('opacity-0', !on);
    });
  }

  function renderModelCards() {
    const wrap = document.getElementById('modelCards');
    if (!S.modality) { wrap.innerHTML = `<p class="text-[13px] text-ink-500 col-span-full">Pilih modality dulu.</p>`; return; }
    const models = RS.modelsFor(S.modality);
    wrap.innerHTML = models.map(m => {
      const on = m.key === S.model;
      const disabled = !m.available;
      return `<button data-model="${m.key}" ${disabled?'disabled':''}
        class="model-card text-left rounded-card border p-4 transition
               ${on?'border-2 border-primary-600 bg-primary-50':'border-line bg-surface'}
               ${disabled?'opacity-50 cursor-not-allowed':''}">
        <div class="flex items-start justify-between">
          <div class="w-9 h-9 rounded-lg bg-primary-100 text-primary-700 flex items-center justify-center mb-3">
            <i data-lucide="cpu" class="w-5 h-5"></i>
          </div>
          <i data-lucide="check-circle-2" class="w-5 h-5 text-primary-600 ${on?'':'opacity-0'}"></i>
        </div>
        <div class="font-semibold text-[15px]">${m.name}</div>
        <div class="text-[13px] text-ink-600">${m.subtitle}</div>
        <div class="text-[12px] text-ink-500 mt-2 font-medium">${m.arch}</div>
        ${disabled?'<div class="text-[11px] text-danger mt-1">Checkpoint tidak tersedia</div>':''}
      </button>`;
    }).join('');
    wrap.querySelectorAll('[data-model]').forEach(b =>
      b.addEventListener('click', () => { if (!b.disabled) selectModel(b.dataset.model); }));
    lucide.createIcons();
  }

  function renderTTA() {
    const wrap = document.getElementById('ttaOptions');
    if (!S.model) { wrap.innerHTML = `<p class="text-[13px] text-ink-500 col-span-2">Pilih model dulu.</p>`; return; }
    const n = DEFAULT_TTA[S.modality];
    const opts = [
      { val: false, label: 'Tanpa TTA (1×)', rec: false },
      { val: true, label: `${n}× TTA`, rec: true },
    ];
    wrap.innerHTML = opts.map(o => {
      const on = S.tta === o.val;
      return `<button data-tta="${o.val}"
        class="tta-opt relative text-left rounded-lg border p-3 transition
               ${on?'border-2 border-primary-600 bg-primary-50':'border-line bg-surface'}">
        <div class="font-medium text-[14px]">${o.label}</div>
        ${o.rec?'<span class="inline-block mt-1 text-[11px] font-semibold text-primary-600 bg-primary-100 px-2 py-0.5 rounded">Direkomendasikan</span>':''}
      </button>`;
    }).join('');
    wrap.querySelectorAll('[data-tta]').forEach(b =>
      b.addEventListener('click', () => { S.tta = b.dataset.tta === 'true'; renderTTA(); refresh(); }));
  }

  const EXAMPLES = {
    cfp: [
      { label: 'Normal', file: 'cfp-normal.png' },
      { label: 'DR ringan', file: 'cfp-mild.png' },
      { label: 'DR sedang', file: 'cfp-moderate.png' },
      { label: 'DR berat', file: 'cfp-severe.png' },
      { label: 'DR proliferatif', file: 'cfp-pdr.png' },
    ],
    oct: [
      { label: 'Normal', file: 'oct-normal.jpeg' },
      { label: 'ARMD', file: 'oct-armd.jpeg' },
      { label: 'CSR', file: 'oct-csr.jpeg' },
      { label: 'Ret. diabetik', file: 'oct-dr.jpeg' },
      { label: 'Macular hole', file: 'oct-mh.jpeg' },
    ],
  };

  function renderExamples(tab) {
    const grid = document.getElementById('exampleGrid');
    const items = EXAMPLES[tab] || [];
    // Placeholder SVG (data-URI) kalau file belum ada -> tidak memicu 404 berulang.
    const ph = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80'%3E%3Crect width='80' height='80' fill='%23F1F5F9'/%3E%3Ctext x='50%25' y='52%25' font-size='9' fill='%2394A3B8' text-anchor='middle' font-family='sans-serif'%3Eno image%3C/text%3E%3C/svg%3E";
    grid.innerHTML = items.map(it => {
      const src = `/static/assets/examples/${tab}/${it.file}`;
      return `<div class="text-center">
        <button data-ex-src="${src}" data-ex-label="${it.label}" class="block w-full aspect-square rounded-lg overflow-hidden bg-surface-2 border border-line hover:border-primary-400 transition">
          <img src="${src}" alt="${it.label}" class="w-full h-full object-cover" onerror="this.onerror=null;this.src='${ph}';this.classList.add('object-contain')">
        </button>
        <div class="text-[11px] text-ink-600 mt-1 truncate">${it.label}</div>
      </div>`;
    }).join('');
    grid.querySelectorAll('[data-ex-src]').forEach(b =>
      b.addEventListener('click', () => openLightbox(b.dataset.exSrc, b.dataset.exLabel)));
    document.querySelectorAll('.ex-tab').forEach(b => {
      const on = b.dataset.ex === tab;
      b.classList.toggle('bg-primary-50', on);
      b.classList.toggle('text-primary-600', on);
      b.classList.toggle('text-ink-600', !on);
    });
    lucide.createIcons();
  }

  function openLightbox(src, label) {
    let lb = document.getElementById('exLightbox');
    if (!lb) {
      lb = document.createElement('div');
      lb.id = 'exLightbox';
      lb.className = 'fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-6 hidden';
      lb.innerHTML = `<div class="bg-surface rounded-panel overflow-hidden max-w-lg w-full">
        <div class="flex items-center justify-between px-4 py-3 border-b border-line">
          <span id="lbLabel" class="font-semibold text-[14px]"></span>
          <button id="lbClose" class="p-1.5 rounded-lg hover:bg-surface-2"><i data-lucide="x" class="w-4 h-4"></i></button>
        </div>
        <img id="lbImg" class="w-full max-h-[70vh] object-contain bg-surface-2">`;
      document.body.appendChild(lb);
      lb.addEventListener('click', e => { if (e.target === lb || e.target.closest('#lbClose')) lb.classList.add('hidden'); });
    }
    lb.querySelector('#lbLabel').textContent = label;
    lb.querySelector('#lbImg').src = src;
    lb.classList.remove('hidden');
    lucide.createIcons();
  }

  // ---------- selection ----------
  function selectModality(m) {
    S.modality = m;
    const models = RS.modelsFor(m);
    S.model = models.find(x => x.available)?.key || models[0]?.key || null;
    S.tta = true; // default direkomendasikan
    renderModalityCards(); renderModelCards(); renderTTA();
    renderExamples(m);
    refresh();
  }
  function selectModel(k) { S.model = k; renderModelCards(); refresh(); }

  // ---------- analyze button state ----------
  function refresh() {
    const ready = S.modality && S.model && S.tta !== null && S.images.length > 0 && S.status !== 'analyzing';
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = !ready;
    const hint = document.getElementById('analyzeHint');
    if (S.images.length === 0) hint.textContent = 'Lengkapi modality, model, TTA, dan minimal 1 citra.';
    else if (ready) hint.textContent = `Siap menganalisis ${S.images.length} citra.`;
  }

  // ---------- run ----------
  async function analyze() {
    if (S.status === 'analyzing') return;
    S.status = 'analyzing';
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    document.getElementById('analyzeLabel').textContent = 'Menganalisis...';
    document.getElementById('analyzeIcon').innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i>';
    lucide.createIcons();

    const fd = new FormData();
    fd.append('model', S.model);
    fd.append('tta', String(S.tta));
    S.images.forEach(im => fd.append('images', im.file, im.name));

    try {
      const r = await fetch('/api/predict_batch', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'gagal');
      S.results = d.results;
      S._processingMs = d.processing_ms;
      S.status = 'done';
      saveHistory(d);
      window.RSResults.show();
    } catch (e) {
      alert('Inferensi gagal: ' + e.message);
      S.status = 'idle';
    } finally {
      document.getElementById('analyzeLabel').textContent = 'Analisis Citra';
      document.getElementById('analyzeIcon').innerHTML = '<i data-lucide="play" class="w-4 h-4"></i>';
      lucide.createIcons();
      refresh();
    }
  }

  function saveHistory(d) {
    const model = RS.models[S.model] || {};
    RS.saveHistoryEntry({
      id: RS.uid(),
      modality: S.modality,
      modalityLabel: S.modality === 'cfp' ? 'Analisis Fundus' : 'Analisis OCT',
      model: S.model,
      modelName: model.name,
      tta: S.tta ? `${model.tta_views}×` : '1×',
      count: S.images.length,
      ts: Date.now(),
      status: 'selesai',
    });
  }

  // ---------- wire events ----------
  document.querySelectorAll('.modality-card').forEach(c =>
    c.addEventListener('click', () => selectModality(c.dataset.modality)));
  document.querySelectorAll('.ex-tab').forEach(b =>
    b.addEventListener('click', () => renderExamples(b.dataset.ex)));
  document.getElementById('analyzeBtn').addEventListener('click', analyze);

  window.RSUpload.init(refresh);
  window.RSResults.init();

  // initial render
  renderModelCards(); renderTTA(); renderExamples('cfp');
  refresh();
  lucide.createIcons();
})();
