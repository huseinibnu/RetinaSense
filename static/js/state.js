/* state.js — sumber kebenaran tunggal untuk halaman Analysis.
   UI selalu di-render dari state ini, bukan sebaliknya. */

window.RS = {
  state: {
    modality: null,        // 'cfp' | 'oct'
    model: null,           // 'cfp_v1' ...
    tta: null,             // true | false
    images: [],            // { id, file, url, name, sizeStr }
    currentIndex: 0,
    results: [],           // hasil dari backend (urutan sama dgn images)
    status: 'idle',        // idle | analyzing | done
    tab: 'results',        // results | summary
    xai: 'attention_rollout',  // attention_rollout | legrad
  },

  models: {},              // registry dari /api/models
  device: 'cpu',

  // ---------- API ----------
  async loadModels() {
    const r = await fetch('/api/models');
    const d = await r.json();
    this.models = d.models;
    this.device = d.device;
    return d;
  },

  modelsFor(modality) {
    return Object.entries(this.models)
      .filter(([, m]) => m.modality === modality)
      .map(([key, m]) => ({ key, ...m }));
  },

  // ---------- history (localStorage) ----------
  HKEY: 'retinasense.history',
  loadHistory() {
    try { return JSON.parse(localStorage.getItem(this.HKEY) || '[]'); }
    catch { return []; }
  },
  saveHistoryEntry(entry) {
    const list = this.loadHistory();
    list.unshift(entry);
    try { localStorage.setItem(this.HKEY, JSON.stringify(list.slice(0, 100))); } catch {}
  },
  deleteHistory(id) {
    const list = this.loadHistory().filter(e => e.id !== id);
    try { localStorage.setItem(this.HKEY, JSON.stringify(list)); } catch {}
  },

  // ---------- util ----------
  fmtSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  },
  uid() { return Math.random().toString(36).slice(2, 10); },
};
