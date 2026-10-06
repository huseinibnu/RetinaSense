/* results.js — render panel hasil: navigasi, prediksi, probabilitas,
   XAI nyata (Attention Rollout & LeGrad via /api/explain), dan Batch Summary. */

window.RSResults = (function () {
  const S = () => window.RS.state;
  const METHOD_LABEL = { attention_rollout: 'Attention Rollout', legrad: 'LeGrad' };
  const METHOD_INFO = {
    attention_rollout: '<b>Attention Rollout</b> — menunjukkan <b>di mana model memusatkan perhatian</b> saat melihat citra, dengan menelusuri attention di seluruh lapisan jaringan. Area merah = paling diperhatikan. Tujuannya: memastikan model fokus pada bagian retina yang relevan, bukan pada latar atau artefak.',
    legrad: '<b>LeGrad</b> — menyoroti bagian citra yang paling <b>memengaruhi keputusan</b> untuk kelas yang diprediksi (berbasis gradien). Area merah = paling mendorong model memilih diagnosis tersebut. Tujuannya: menjelaskan <i>mengapa</i> model sampai pada prediksi itu, bukan sekadar di mana ia melihat.',
  };

  // ---------- panel prediksi ----------
  function renderPrediction(res) {
    const top = res.display[res.argmax];
    const conf = Math.round(res.confidence * 100);

    let gradeLine = '';
    if (res.task_type === 'ordinal') {
      gradeLine = `<div class="text-[12px] text-ink-500 mt-0.5">Expected grade ${res.expected_grade.toFixed(2)} · ${res.display[res.grade]}</div>`;
    }

    document.getElementById('predMain').innerHTML = `
      <div class="text-[22px] font-bold leading-tight">${top}</div>
      ${gradeLine}
      <div class="mt-2 text-[12px] font-semibold text-ink-600">Confidence</div>
      <div class="text-[28px] font-bold text-primary-600 leading-none">${conf}%</div>`;

    // flag rujukan hanya untuk ordinal (APTOS)
    const refEl = document.getElementById('predReferral');
    if (res.task_type === 'ordinal') {
      refEl.innerHTML = res.refer
        ? `<span class="inline-flex items-center gap-1.5 mt-3 px-3 py-1.5 rounded-lg bg-dangerbg text-danger text-[13px] font-semibold"><i data-lucide="alert-triangle" class="w-4 h-4"></i>Perlu rujukan dokter mata</span>`
        : `<span class="inline-flex items-center gap-1.5 mt-3 px-3 py-1.5 rounded-lg bg-okbg text-ok text-[13px] font-semibold"><i data-lucide="check-circle-2" class="w-4 h-4"></i>Tidak perlu rujukan</span>`;
    } else { refEl.innerHTML = ''; }

    // probability bars
    const maxIdx = res.argmax;
    document.getElementById('probList').innerHTML = res.display.map((label, i) => {
      const pct = Math.round(res.probs[i] * 100);
      return `<div>
        <div class="flex justify-between text-[12px] mb-1"><span class="${i===maxIdx?'font-semibold text-ink-900':'text-ink-600'}">${label}</span><span class="tabular-nums ${i===maxIdx?'font-semibold':'text-ink-500'}">${pct}%</span></div>
        <div class="prob-track"><div class="prob-fill ${i===maxIdx?'is-top':''}" style="width:${pct}%"></div></div>
      </div>`;
    }).join('');
  }

  // ---------- XAI (Attention Rollout & LeGrad via /api/explain) ----------
  function setXaiImages(xai, method) {
    const m = xai && xai.methods && xai.methods[method];
    document.getElementById('xaiBase').src = xai ? xai.base : '';
    document.getElementById('xaiMap').src = m ? m.map : '';
    document.getElementById('xaiOverlay').src = m ? m.overlay : '';
  }

  function renderXAI() {
    const idx = S().currentIndex;
    const res = S().results[idx];
    const method = S().xai;
    const loadingEl = document.getElementById('xaiLoading');
    if (!res || res.error) { setXaiImages(null, method); loadingEl.classList.add('hidden'); return; }

    if (res._xai) {
      setXaiImages(res._xai, method);
      loadingEl.classList.add('hidden');
    } else if (res._xaiError) {
      setXaiImages(null, method);
      loadingEl.classList.add('hidden');
    } else {
      setXaiImages(null, method);
      loadingEl.classList.remove('hidden');
      fetchExplain(idx);           // lazy, sekali per citra
    }
  }

  async function fetchExplain(idx) {
    const res = S().results[idx], im = S().images[idx];
    if (!res || !im || res._xaiLoading) return;
    res._xaiLoading = true;
    try {
      const fd = new FormData();
      fd.append('model', S().model);
      fd.append('image', im.file, im.name);
      if (typeof res.argmax === 'number') fd.append('class_idx', String(res.argmax));
      const r = await fetch('/api/explain', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'gagal');
      res._xai = d;
    } catch (e) {
      res._xaiError = e.message;
      console.error('XAI gagal:', e);
    } finally {
      res._xaiLoading = false;
      if (S().currentIndex === idx) renderXAI();   // render kalau masih di citra ini
    }
  }

  function openXaiLightbox(which) {
    const res = S().results[S().currentIndex];
    if (!res || !res._xai) return;
    const method = S().xai;
    const m = res._xai.methods[method];
    const src = which === 'base' ? res._xai.base : (which === 'map' ? m.map : m.overlay);
    const titleMap = { base: 'Asli (praproses)', map: 'Heatmap', overlay: 'Overlay' };
    let lb = document.getElementById('xaiLightbox');
    if (!lb) {
      lb = document.createElement('div');
      lb.id = 'xaiLightbox';
      lb.className = 'fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-6 hidden';
      lb.innerHTML = `<div class="bg-surface rounded-panel overflow-hidden max-w-2xl w-full">
        <div class="flex items-center justify-between px-4 py-3 border-b border-line">
          <span id="xlbTitle" class="font-semibold text-[14px]"></span>
          <button id="xlbClose" class="p-1.5 rounded-lg hover:bg-surface-2"><i data-lucide="x" class="w-4 h-4"></i></button>
        </div>
        <img id="xlbImg" class="w-full max-h-[70vh] object-contain bg-[#0F172A]">
        <div class="px-4 py-3 border-t border-line">
          <div class="heatmap-legend h-2 rounded-full"></div>
          <div class="flex justify-between text-[11px] text-ink-500 mt-1"><span>Atensi rendah</span><span>Atensi tinggi</span></div>
          <p id="xlbInfo" class="text-[12px] text-ink-600 mt-2.5 leading-relaxed"></p>
        </div></div>`;
      document.body.appendChild(lb);
      lb.addEventListener('click', e => { if (e.target === lb || e.target.closest('#xlbClose')) lb.classList.add('hidden'); });
    }
    lb.querySelector('#xlbTitle').textContent = `${METHOD_LABEL[method]} — ${titleMap[which]}`;
    lb.querySelector('#xlbImg').src = src;
    lb.querySelector('#xlbInfo').innerHTML = METHOD_INFO[method] || '';
    lb.classList.remove('hidden');
    lucide.createIcons();
  }

  function renderThumbs() {
    const strip = document.getElementById('thumbStrip');
    strip.innerHTML = S().images.map((im, i) => `
      <button data-idx="${i}" class="shrink-0 w-14 h-14 rounded-lg overflow-hidden border-2 ${i===S().currentIndex?'border-primary-600':'border-line'}">
        <img src="${im.url}" class="w-full h-full object-cover">
      </button>`).join('');
    strip.querySelectorAll('[data-idx]').forEach(b =>
      b.addEventListener('click', () => go(+b.dataset.idx)));
    document.getElementById('imgCounter').textContent = `${S().currentIndex + 1} / ${S().images.length}`;
  }

  function renderCurrent() {
    const idx = S().currentIndex;
    const res = S().results[idx], im = S().images[idx];
    if (!res || !im) return;
    document.getElementById('resOriginal').src = im.url;
    document.getElementById('resFilename').textContent = `${im.name} · ${im.sizeStr}`;
    if (res.error) {
      document.getElementById('predMain').innerHTML = `<div class="text-danger text-[14px] font-medium">${res.error}</div>`;
      document.getElementById('predReferral').innerHTML = '';
      document.getElementById('probList').innerHTML = '';
    } else {
      renderPrediction(res);
    }
    renderXAI();
    renderThumbs();
    lucide.createIcons();
  }

  function go(i) {
    const n = S().images.length;
    S().currentIndex = (i + n) % n;
    renderCurrent();
  }

  // ---------- batch summary ----------
  function renderSummary() {
    const meta = S();
    const model = window.RS.models[meta.model] || {};
    const ok = meta.results.filter(r => !r.error).length;
    const totalMs = meta._processingMs || 0;
    const stats = [
      ['Citra dianalisis', meta.images.length],
      ['Model', model.name || meta.model],
      ['TTA', meta.tta ? `${model.tta_views}×` : 'Nonaktif (1×)'],
      ['Selesai', ok],
      ['Waktu proses', (totalMs / 1000).toFixed(1) + ' s'],
    ];
    document.getElementById('summaryStats').innerHTML = stats.map(([k, v]) =>
      `<div><div class="text-[12px] text-ink-500">${k}</div><div class="font-semibold mt-0.5">${v}</div></div>`).join('');

    document.getElementById('summaryRows').innerHTML = meta.results.map((r, i) => {
      if (r.error) return `<tr><td class="px-4 py-2.5">${meta.images[i].name}</td><td class="px-4 py-2.5 text-danger" colspan="3">${r.error}</td></tr>`;
      const refer = r.task_type === 'ordinal'
        ? (r.refer ? `<span class="text-danger font-medium">Ya</span>` : `<span class="text-ok">Tidak</span>`)
        : '<span class="text-ink-500">—</span>';
      return `<tr class="hover:bg-surface-2 cursor-pointer" data-row="${i}">
        <td class="px-4 py-2.5 font-medium">${r.filename}</td>
        <td class="px-4 py-2.5">${r.display[r.argmax]}</td>
        <td class="px-4 py-2.5 tabular-nums">${Math.round(r.confidence*100)}%</td>
        <td class="px-4 py-2.5">${refer}</td></tr>`;
    }).join('');
    document.getElementById('summaryRows').querySelectorAll('[data-row]').forEach(tr =>
      tr.addEventListener('click', () => { setTab('results'); go(+tr.dataset.row); }));
  }

  function setTab(tab) {
    S().tab = tab;
    document.getElementById('tabResults').classList.toggle('hidden', tab !== 'results');
    document.getElementById('tabSummary').classList.toggle('hidden', tab !== 'summary');
    document.querySelectorAll('.result-tab').forEach(b => {
      const on = b.dataset.tab === tab;
      b.classList.toggle('border-primary-600', on);
      b.classList.toggle('text-primary-600', on);
      b.classList.toggle('border-transparent', !on);
      b.classList.toggle('text-ink-500', !on);
    });
    if (tab === 'summary') renderSummary();
  }

  function setXAI(x) {
    S().xai = x;
    document.querySelectorAll('.xai-tab').forEach(b => {
      const on = b.dataset.xai === x;
      b.classList.toggle('bg-primary-50', on);
      b.classList.toggle('text-primary-600', on);
      b.classList.toggle('text-ink-600', !on);
    });
    const info = document.getElementById('xaiMethodInfo');
    if (info) info.innerHTML = METHOD_INFO[x] || '';
    renderXAI();
  }

  function show() {
    document.getElementById('resultsSection').classList.remove('hidden');
    S().currentIndex = 0;
    setTab('results');
    setXAI(S().xai);
    renderCurrent();
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
  }

  function init() {
    document.getElementById('prevImg').addEventListener('click', () => go(S().currentIndex - 1));
    document.getElementById('nextImg').addEventListener('click', () => go(S().currentIndex + 1));
    document.querySelectorAll('.result-tab').forEach(b => b.addEventListener('click', () => setTab(b.dataset.tab)));
    document.querySelectorAll('.xai-tab').forEach(b => b.addEventListener('click', () => setXAI(b.dataset.xai)));
    document.querySelectorAll('[data-xai-zoom]').forEach(b => b.addEventListener('click', () => openXaiLightbox(b.dataset.xaiZoom)));
    document.getElementById('downloadBtn').addEventListener('click', download);
  }

  function download() {
    // buang field XAI (base64 besar) & flag internal dari ekspor
    const strip = r => {
      const { _xai, _xaiLoading, _xaiError, ...rest } = r;
      return rest;
    };
    const data = {
      model: S().model, tta: S().tta,
      results: S().results.map(strip),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `retinasense_results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return { init, show };
})();
