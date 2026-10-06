/* upload.js — dropzone, pilih berkas, render thumbnail terunggah. */

window.RSUpload = (function () {
  const S = () => window.RS.state;
  let onChange = () => {};

  function addFiles(fileList) {
    const imgs = [...fileList].filter(f => f.type.startsWith('image/'));
    for (const f of imgs) {
      if (S().images.length >= 16) break;
      S().images.push({
        id: window.RS.uid(),
        file: f,
        url: URL.createObjectURL(f),
        name: f.name,
        sizeStr: window.RS.fmtSize(f.size),
      });
    }
    renderGrid();
    onChange();
  }

  function remove(id) {
    const i = S().images.findIndex(x => x.id === id);
    if (i >= 0) { URL.revokeObjectURL(S().images[i].url); S().images.splice(i, 1); }
    renderGrid();
    onChange();
  }

  function clearAll() {
    S().images.forEach(x => URL.revokeObjectURL(x.url));
    S().images = [];
    renderGrid();
    onChange();
  }

  function renderGrid() {
    const wrap = document.getElementById('uploadedWrap');
    const grid = document.getElementById('uploadGrid');
    const imgs = S().images;
    document.getElementById('uploadCount').textContent = imgs.length;
    wrap.classList.toggle('hidden', imgs.length === 0);

    grid.innerHTML = imgs.map(im => `
      <div class="relative rounded-lg overflow-hidden border border-line bg-surface-2 group">
        <div class="aspect-square"><img src="${im.url}" class="w-full h-full object-cover" alt=""></div>
        <button data-rm="${im.id}" class="absolute top-1.5 right-1.5 w-6 h-6 rounded-full bg-ink-900/70 text-white
                flex items-center justify-center opacity-0 group-hover:opacity-100 transition" title="Hapus">
          <i data-lucide="x" class="w-3.5 h-3.5"></i>
        </button>
        <span class="absolute top-1.5 left-1.5 w-5 h-5 rounded-full bg-ok text-white flex items-center justify-center">
          <i data-lucide="check" class="w-3 h-3"></i>
        </span>
        <div class="px-2 py-1.5 bg-surface">
          <p class="text-[11px] font-medium truncate">${im.name}</p>
          <p class="text-[10px] text-ink-500">${im.sizeStr}</p>
        </div>
      </div>`).join('');

    grid.querySelectorAll('[data-rm]').forEach(b =>
      b.addEventListener('click', () => remove(b.dataset.rm)));
    lucide.createIcons();
  }

  function init(cb) {
    onChange = cb || onChange;
    const dz = document.getElementById('dropzone');
    const input = document.getElementById('fileInput');

    document.getElementById('browseBtn').addEventListener('click', e => { e.stopPropagation(); input.click(); });
    dz.addEventListener('click', () => input.click());
    input.addEventListener('change', e => { addFiles(e.target.files); input.value = ''; });

    ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => {
      e.preventDefault(); dz.classList.add('bg-primary-50', 'border-primary-600');
    }));
    ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => {
      e.preventDefault(); dz.classList.remove('bg-primary-50', 'border-primary-600');
    }));
    dz.addEventListener('drop', e => { if (e.dataTransfer?.files) addFiles(e.dataTransfer.files); });

    document.getElementById('clearAll').addEventListener('click', clearAll);
  }

  return { init, addFiles, clearAll };
})();
