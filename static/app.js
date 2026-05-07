/* ── Constants ─────────────────────────────────────────────────────────────── */

const BENCH_NAMES = {
  oll_BBH:          "BIG-Bench Hard",
  oll_GPQA:         "GPQA",
  oll_IFEval:       "Instruction Following",
  oll_MATH_lvl5:    "MATH Level 5",
  oll_MMLU_PRO:     "MMLU-Pro",
  oll_MUSR:         "MUSR",
  oe_aggregate:     "OpenEvals Composite",
  oe_aime2026:      "AIME 2026",
  oe_gpqa:          "GPQA (OpenEvals)",
  oe_gsm8k:         "GSM8K",
  oe_hle:           "Humanity's Last Exam",
  oe_hmmt2026:      "HMMT 2026",
  oe_mmlu_pro:      "MMLU-Pro (OE)",
  oe_olmOcr:        "OLM-OCR",
  oe_swePro:        "SWE-bench Pro",
  oe_sweVerified:   "SWE-bench Verified",
  oe_terminalBench: "TerminalBench",
  oe_evasionBench:  "EvasionBench",
  arena_text_elo:   "Chatbot Arena Elo",
};

// Includes special non-benchmark axes
const AXIS_NAMES = { ...BENCH_NAMES, release_date: "Release Date" };

const ORIGIN_COLORS = {
  "United States":        "#5b8ed6",
  "China":                "#e25f5f",
  "France":               "#e8c040",
  "United Kingdom":       "#4bc9a5",
  "Canada":               "#f08c30",
  "Multinational":        "#b07af5",
  "South Korea":          "#6ac96a",
  "United Arab Emirates": "#ff80a0",
  "Unknown":              "#7878a0",
};

const COMPANY_PALETTE = [
  "#5b8ed6","#e25f5f","#4bc9a5","#f08c30","#b07af5",
  "#e8c040","#6ac96a","#ff80a0","#ff6b35","#a8dadc",
  "#457b9d","#e63946","#2a9d8f","#f4a261","#8338ec",
  "#fb8500","#a7c957","#d62828","#606c38","#dda15e",
];

const AREA_PILL_CLASS = {
  math:                  "pill-math",
  coding:                "pill-coding",
  reasoning:             "pill-reasoning",
  knowledge:             "pill-knowledge",
  science:               "pill-science",
  instruction_following: "pill-instruction_following",
  safety:                "pill-safety",
  document_vision:       "pill-document_vision",
  chat_human_eval:       "pill-chat_human_eval",
  aggregate:             "pill-aggregate",
};

/* ── Shared Plotly config ──────────────────────────────────────────────────── */

const AXIS_STYLE = {
  gridcolor:     "#1e1e38",
  zerolinecolor: "#2a2a4a",
  tickfont:      { size: 11, color: "#ffffff" },
  title:         { font: { size: 13, color: "#ffffff" } },
};

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(13,13,26,0.85)",
  font:   { color: "#ffffff", family: "Inter, system-ui, sans-serif" },
  legend: { bgcolor: "rgba(20,20,40,0.92)", bordercolor: "#252540", borderwidth: 1,
            font: { size: 11, color: "#ffffff" }, itemsizing: "constant" },
  margin:    { l: 64, r: 24, t: 24, b: 56 },
  hovermode: "closest",
  hoverlabel: { bgcolor: "#141428", bordercolor: "#7c6af7",
                font: { size: 12, color: "#e0e0f0", family: "Inter, system-ui, sans-serif" }, align: "left" },
  modebar: { bgcolor: "rgba(0,0,0,0)", color: "#6868a0", activecolor: "#7c6af7" },
};

const PLOT_CONFIG = {
  responsive:     true,
  displayModeBar: "hover",
  displaylogo:    false,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
};

/* ── State ─────────────────────────────────────────────────────────────────── */

const state = {
  pca:      { dims: 2, loaded: false },
  custom:   { dims: 2, loaded: false },
  classify: { classifier: "svm", loaded: false },
  info:     { loaded: false },
  activePage: "pca",
};

/* ── Page navigation ───────────────────────────────────────────────────────── */

function showPage(name) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.getElementById(`page-${name}`).classList.add("active");
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`navbtn-${name}`).classList.add("active");
  state.activePage = name;

  if (!state[name].loaded) {
    state[name].loaded = true;
    if (name === "pca")    fetchAndRenderPCA();
    if (name === "custom") fetchAndRenderCustom();
    // classify: no auto-load — user must click "Train & Evaluate"
    // info: built during init, nothing to fetch
  }
}

/* ── Bootstrap ─────────────────────────────────────────────────────────────── */

async function init() {
  try {
    const res = await fetch("/api/filters");
    if (!res.ok) throw new Error(`/api/filters returned ${res.status}`);
    const filters = await res.json();
    buildAllControls(filters);
    state.pca.loaded = true;
    fetchAndRenderPCA();
  } catch (err) {
    showPageEmpty("pca", "Failed to load filter data. Is the server running?");
  }
}

/* ── Build controls ────────────────────────────────────────────────────────── */

function buildAllControls(filters) {
  // PCA
  buildBenchChecks("pca", filters.benchmarks);
  buildOriginChecks("pca", filters.origins);
  buildYearSelects("pca", filters.years);
  wireSparse("pca-toggle-sparse", "pca-sparse-benchmarks", "pca-sparse-icon");
  el("pca-color-by").addEventListener("change", () => { if (_pcaData) renderPCA(_pcaData); });

  // Custom Axes
  buildAxisDropdowns(filters.benchmarks);
  buildOriginChecks("cx", filters.origins);
  buildYearSelects("cx", filters.years);
  el("cx-color-by").addEventListener("change", () => { if (_cxData) renderCustom(_cxData); });

  // Benchmarks info page
  buildInfoPage(filters.benchmarks);

  // Classify
  buildBenchChecks("cls", filters.benchmarks);
  buildOriginChecks("cls", filters.origins);
  buildYearSelects("cls", filters.years);
  wireSlider("cls-top-n", "cls-top-n-val", v => v);
  wireSlider("cls-test-size", "cls-test-size-val", v => v + "%");
  wireSparse("cls-toggle-sparse", "cls-sparse-benchmarks", "cls-sparse-icon");
}

function buildBenchChecks(prefix, benchmarks) {
  const primEl   = el(`${prefix}-benchmark-list`);
  const sparseEl = el(`${prefix}-sparse-benchmarks`);
  for (const b of benchmarks) {
    const item = makeBenchCheckbox(`${prefix}-bench__${b.col}`, b, BENCH_NAMES[b.col] || b.col, !b.is_primary ? false : true);
    (b.is_primary ? primEl : sparseEl).appendChild(item);
  }
}

function buildOriginChecks(prefix, origins) {
  const container = el(`${prefix}-origin-list`);
  for (const origin of origins) {
    const lbl = document.createElement("label");
    lbl.className = "checkbox-item";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.id = `${prefix}-origin__${origin}`; cb.dataset.col = origin; cb.checked = true;
    const sp = document.createElement("span"); sp.className = "cb-text";
    const mn = document.createElement("span"); mn.className = "cb-main"; mn.textContent = origin;
    sp.appendChild(mn); lbl.appendChild(cb); lbl.appendChild(sp);
    container.appendChild(lbl);
  }
}

/* ══════════════════════════════════════════════════════════════════════════════
   BENCHMARKS INFO PAGE
══════════════════════════════════════════════════════════════════════════════ */

const AREA_LABELS = {
  math:                  "Math",
  coding:                "Coding",
  reasoning:             "Reasoning",
  knowledge:             "Knowledge",
  science:               "Science",
  instruction_following: "Instruction Following",
  safety:                "Safety",
  document_vision:       "Document & Vision",
  chat_human_eval:       "Human Eval",
  aggregate:             "Aggregate",
};

const SOURCE_SHORT = {
  "Open LLM Leaderboard (Hugging Face)": "Open LLM Leaderboard",
  "Open LLM Leaderboard":                "Open LLM Leaderboard",
  "OpenEvals/leaderboard-data (Hugging Face)": "OpenEvals",
  "OpenEvals/leaderboard-data":          "OpenEvals",
  "LMSYS-style Arena data via public mirror": "Chatbot Arena",
};

let _infoBenchmarks = [];
let _infoActiveAreas = new Set();

function buildInfoPage(benchmarks) {
  _infoBenchmarks = benchmarks;

  // Collect all unique test areas
  const allAreas = new Set();
  for (const b of benchmarks) for (const a of b.test_areas) allAreas.add(a);

  // Build area filter pills
  const areaContainer = el("info-area-filters");
  for (const area of [...allAreas].sort()) {
    const btn = document.createElement("button");
    btn.className = `area-pill ${AREA_PILL_CLASS[area] || ""} info-area-btn`;
    btn.dataset.area = area;
    btn.textContent = AREA_LABELS[area] || area;
    btn.addEventListener("click", () => {
      if (_infoActiveAreas.has(area)) {
        _infoActiveAreas.delete(area);
        btn.classList.remove("active");
      } else {
        _infoActiveAreas.add(area);
        btn.classList.add("active");
      }
      renderInfoCards();
    });
    areaContainer.appendChild(btn);
  }

  renderInfoCards();
}

function filterInfoCards() { renderInfoCards(); }

function renderInfoCards() {
  const query = (el("info-search")?.value || "").toLowerCase();
  const container = el("info-cards");
  container.innerHTML = "";

  const visible = _infoBenchmarks
    .filter(b => {
      if (_infoActiveAreas.size > 0 && !b.test_areas.some(a => _infoActiveAreas.has(a))) return false;
      if (query) {
        const haystack = `${b.col} ${BENCH_NAMES[b.col] || ""} ${b.description} ${b.test_areas.join(" ")} ${b.source}`.toLowerCase();
        if (!haystack.includes(query)) return false;
      }
      return true;
    })
    .sort((a, b) => (b.n - a.n));

  if (!visible.length) {
    container.innerHTML = `<p class="info-empty">No benchmarks match your filters.</p>`;
    return;
  }

  for (const b of visible) {
    const name     = BENCH_NAMES[b.col] || b.col;
    const srcShort = SOURCE_SHORT[b.source] || b.source;

    let statsHtml = "";
    if (b.n > 0 && b.mean !== null) {
      statsHtml = `
        <div class="bench-stats">
          <span class="bench-stat"><span class="stat-l">N</span><span class="stat-v">${b.n}</span></span>
          <span class="bench-stat"><span class="stat-l">Mean</span><span class="stat-v">${b.mean}</span></span>
          <span class="bench-stat"><span class="stat-l">Median</span><span class="stat-v">${b.median}</span></span>
          <span class="bench-stat"><span class="stat-l">Range</span><span class="stat-v">${b.min} – ${b.max}</span></span>
        </div>`;
    }

    const tagsHtml = b.test_areas
      .map(a => `<span class="area-pill ${AREA_PILL_CLASS[a] || ""}">${AREA_LABELS[a] || a}</span>`)
      .join("");

    const hib = b.higher_is_better !== false
      ? `<span class="bench-badge bench-badge-hib">↑ higher is better</span>`
      : `<span class="bench-badge bench-badge-lib">↓ lower is better</span>`;

    const card = document.createElement("div");
    card.className = "bench-card";
    card.innerHTML = `
      <div class="bench-card-header">
        <div class="bench-card-title">${esc(name)}</div>
        <div class="bench-card-badges">${hib}</div>
      </div>
      <div class="bench-tags">${tagsHtml}</div>
      <p class="bench-desc">${esc(b.description)}</p>
      ${statsHtml}
      <div class="bench-card-footer">
        <span class="bench-key">${esc(b.col)}</span>
        <span class="bench-source">${esc(srcShort)}</span>
      </div>`;
    container.appendChild(card);
  }
}

function buildAxisDropdowns(benchmarks) {
  const defaultAxes = ["oll_MMLU_PRO", "oll_BBH", "oll_GPQA"];
  for (const [idx, sid] of ["cx-axis-x", "cx-axis-y", "cx-axis-z"].entries()) {
    const sel = el(sid);

    // Special axes group
    const specialGrp = document.createElement("optgroup");
    specialGrp.label = "Special";
    specialGrp.appendChild(new Option("Release Date", "release_date"));
    sel.appendChild(specialGrp);

    const primGrp   = document.createElement("optgroup"); primGrp.label   = "Primary Benchmarks";
    const sparseGrp = document.createElement("optgroup"); sparseGrp.label = "Sparse Benchmarks";
    for (const b of benchmarks)
      (b.is_primary ? primGrp : sparseGrp).appendChild(new Option(BENCH_NAMES[b.col] || b.col, b.col));
    sel.appendChild(primGrp);
    sel.appendChild(sparseGrp);
    sel.value = defaultAxes[idx] || benchmarks[idx]?.col || "";
  }
}

function buildYearSelects(prefix, years) {
  const minEl = el(`${prefix}-year-min`), maxEl = el(`${prefix}-year-max`);
  for (const y of years) { minEl.add(new Option(y, y)); maxEl.add(new Option(y, y)); }
  if (years.length) { minEl.value = years[0]; maxEl.value = years[years.length - 1]; }
}

function wireSlider(sliderId, labelId, fmt = v => v + "%") {
  el(sliderId)?.addEventListener("input", e => { el(labelId).textContent = fmt(e.target.value); });
}

function wireSparse(btnId, listId, iconId) {
  el(btnId)?.addEventListener("click", () => {
    const hidden = el(listId).classList.toggle("hidden");
    el(iconId).textContent = hidden ? "▶" : "▼";
  });
}

function selectAll(prefix, checked) {
  document.querySelectorAll(`[id^="${prefix}"]`).forEach(cb => { cb.checked = checked; });
}

/* ── Read helpers ──────────────────────────────────────────────────────────── */

function readBenches(prefix) {
  return [...document.querySelectorAll(`[id^="${prefix}-bench__"]:checked`)].map(cb => cb.dataset.col);
}
function readOrigins(prefix) {
  return [...document.querySelectorAll(`[id^="${prefix}-origin__"]:checked`)].map(cb => cb.dataset.col);
}
function readFilters(prefix) {
  return {
    origins:     readOrigins(prefix),
    yearMin:     el(`${prefix}-year-min`)?.value  || "",
    yearMax:     el(`${prefix}-year-max`)?.value  || "",
    minCoverage: 0,
  };
}

/* ── Dim toggles ───────────────────────────────────────────────────────────── */

function setPcaDims(d) {
  state.pca.dims = d;
  el("pca-btn2d").classList.toggle("active", d === 2);
  el("pca-btn3d").classList.toggle("active", d === 3);
  el("pca-stat-pc3-wrap").style.display = d === 3 ? "" : "none";
}

function setCxDims(d) {
  state.custom.dims = d;
  el("cx-btn2d").classList.toggle("active", d === 2);
  el("cx-btn3d").classList.toggle("active", d === 3);
  el("cx-axis-z-group").style.display = d === 3 ? "" : "none";
  el("cx-stat-z-wrap").style.display  = d === 3 ? "" : "none";
}

/* ── Classifier selection ──────────────────────────────────────────────────── */

const CLS_HINTS = {
  svm: "SVM (RBF kernel). Feature importances not available.",
  dt:  "Decision Tree (max depth 5). Shows feature importances.",
  lr:  "Logistic Regression. Shows mean |coefficient| per feature.",
};

function setClassifier(type) {
  state.classify.classifier = type;
  ["svm", "dt", "lr"].forEach(t =>
    el(`cls-btn-${t}`).classList.toggle("active", t === type));
  el("cls-hint").textContent = CLS_HINTS[type];
}

/* ══════════════════════════════════════════════════════════════════════════════
   PCA PAGE
══════════════════════════════════════════════════════════════════════════════ */

let _pcaData = null;

async function fetchAndRenderPCA() {
  const benches = readBenches("pca");
  if (!benches.length) { showPageEmpty("pca", "Select at least one benchmark."); return; }
  const f = readFilters("pca");
  const params = new URLSearchParams({
    dims: state.pca.dims, benchmarks: benches.join(","),
    origins: f.origins.join(","), min_coverage: f.minCoverage,
  });
  if (f.yearMin) params.set("year_min", f.yearMin);
  if (f.yearMax) params.set("year_max", f.yearMax);

  setLoading("pca", true); hidePageEmpty("pca");
  try {
    const data = await fetchJSON("/api/pca?" + params);
    if (data.error || !data.points.length) { showPageEmpty("pca", data.error || "No data."); return; }
    _pcaData = data;
    updatePcaStats(data);
    renderPCA(data);
    renderLoadingsPanel(data.loadings, data.variance_explained);
  } catch (e) { showPageEmpty("pca", "Error: " + e.message); }
  finally { setLoading("pca", false); }
}

function renderPCA(data) {
  const is3d    = state.pca.dims === 3;
  const colorBy = el("pca-color-by").value;
  const labels  = data.variance_explained.map((v, i) => `PC${i+1} (${pct(v)})`);
  const traces  = buildTraces(data.points, is3d, colorBy,
                    p => p.pc1, p => p.pc2, p => p.pc3 ?? 0);
  Plotly.react("chart-pca", traces, buildLayout(labels, is3d), PLOT_CONFIG);
  if (!is3d) attachAxisTooltips("chart-pca", data.loadings, data.variance_explained);
}

function updatePcaStats(d) {
  const ve = d.variance_explained;
  el("pca-stat-count").textContent   = d.n_points.toLocaleString();
  el("pca-stat-pc1").textContent     = ve[0] != null ? pct(ve[0]) : "—";
  el("pca-stat-pc2").textContent     = ve[1] != null ? pct(ve[1]) : "—";
  el("pca-stat-pc3").textContent     = ve[2] != null ? pct(ve[2]) : "—";
  el("pca-stat-total").textContent   = pct(ve.reduce((a,b) => a+b, 0));
  el("pca-stat-benches").textContent = d.selected_benchmarks.length;
  el("pca-stat-pc3-wrap").style.display = state.pca.dims === 3 ? "" : "none";
}

function renderLoadingsPanel(loadings, ve) {
  if (!loadings || !Object.keys(loadings).length) return;
  const content = el("pca-loadings-content");
  content.innerHTML = "";
  for (let i = 1; i <= Object.keys(loadings).length; i++) {
    const pcLoad = loadings[`pc${i}`];
    const vePct  = ve[i-1] != null ? (ve[i-1] * 100).toFixed(1) : "?";
    const title  = document.createElement("div");
    title.className = "loading-pc-title";
    title.textContent = `PC${i} (${vePct}% variance)`;
    content.appendChild(title);
    const top5   = pcLoad.slice(0, 5);
    const maxAbs = Math.max(...top5.map(l => Math.abs(l.loading)));
    for (const { benchmark, loading } of top5) {
      const row = document.createElement("div"); row.className = "loading-row";
      const nm  = document.createElement("span"); nm.className = "loading-name";
      nm.textContent = BENCH_NAMES[benchmark] || benchmark;
      const bw  = document.createElement("div"); bw.className = "loading-bar-wrap";
      const bar = document.createElement("div");
      bar.className = `loading-bar ${loading >= 0 ? "pos" : "neg"}`;
      bar.style.width = `${(Math.abs(loading) / maxAbs * 100).toFixed(0)}%`;
      bw.appendChild(bar);
      const val = document.createElement("span"); val.className = "loading-val";
      val.textContent = (loading >= 0 ? "+" : "") + loading.toFixed(3);
      row.appendChild(nm); row.appendChild(bw); row.appendChild(val);
      content.appendChild(row);
    }
    if (i < Object.keys(loadings).length) {
      const sep = document.createElement("div"); sep.className = "loading-sep";
      content.appendChild(sep);
    }
  }
  el("pca-loadings-section").style.display = "";
}

function attachAxisTooltips(chartId, loadings, ve) {
  if (!loadings) return;
  setTimeout(() => {
    const svg = document.querySelector(`#${chartId} svg.main-svg`);
    if (!svg) return;
    const attach = (sel, pcKey, label) => {
      const svgEl = svg.querySelector(sel);
      if (!svgEl || !loadings[pcKey]) return;
      const tip = loadings[pcKey].slice(0, 5)
        .map(l => `${BENCH_NAMES[l.benchmark] || l.benchmark}  ${l.loading >= 0 ? "+" : ""}${l.loading.toFixed(3)}`)
        .join("\n");
      svgEl.style.cursor = "help";
      let t = svgEl.querySelector("title");
      if (!t) { t = document.createElementNS("http://www.w3.org/2000/svg", "title"); svgEl.prepend(t); }
      t.textContent = `${label}\n${"─".repeat(22)}\n${tip}`;
    };
    attach(".xtitle", "pc1", `PC1 (${pct(ve[0])} variance)`);
    attach(".ytitle", "pc2", `PC2 (${pct(ve[1])} variance)`);
  }, 220);
}

/* ══════════════════════════════════════════════════════════════════════════════
   CUSTOM AXES PAGE
══════════════════════════════════════════════════════════════════════════════ */

let _cxData = null;

async function fetchAndRenderCustom() {
  const xAxis = el("cx-axis-x").value;
  const yAxis = el("cx-axis-y").value;
  const zAxis = state.custom.dims === 3 ? el("cx-axis-z").value : null;
  if (!xAxis || !yAxis) { showPageEmpty("custom", "Select X and Y axes."); return; }

  const f = readFilters("cx");
  const params = new URLSearchParams({
    x_axis: xAxis, y_axis: yAxis,
    origins: f.origins.join(","), min_coverage: f.minCoverage,
  });
  if (zAxis) params.set("z_axis", zAxis);
  if (f.yearMin) params.set("year_min", f.yearMin);
  if (f.yearMax) params.set("year_max", f.yearMax);

  setLoading("custom", true); hidePageEmpty("custom");
  try {
    const data = await fetchJSON("/api/scatter?" + params);
    if (data.error || !data.points.length) { showPageEmpty("custom", data.error || "No data."); return; }
    _cxData = { ...data, xAxis, yAxis, zAxis };
    updateCxStats(data, xAxis, yAxis, zAxis);
    renderCustom(_cxData);
  } catch (e) { showPageEmpty("custom", "Error: " + e.message); }
  finally { setLoading("custom", false); }
}

function renderCustom(data) {
  const { xAxis, yAxis, zAxis } = data;
  const is3d    = state.custom.dims === 3 && !!zAxis;
  const colorBy = el("cx-color-by").value;
  const pts     = is3d ? data.points.filter(p => p.x != null && p.y != null && p.z != null)
                       : data.points.filter(p => p.x != null && p.y != null);
  const traces  = buildTraces(pts, is3d, colorBy, p => p.x, p => p.y, p => p.z ?? 0);
  const labels  = [AXIS_NAMES[xAxis] || xAxis, AXIS_NAMES[yAxis] || yAxis,
                   zAxis ? (AXIS_NAMES[zAxis] || zAxis) : "Z"];
  Plotly.react("chart-custom", traces, buildLayout(labels, is3d), PLOT_CONFIG);
}

function updateCxStats(data, xAxis, yAxis, zAxis) {
  el("cx-stat-count").textContent = data.n_points.toLocaleString();
  el("cx-stat-x").textContent     = AXIS_NAMES[xAxis] || xAxis;
  el("cx-stat-y").textContent     = AXIS_NAMES[yAxis] || yAxis;
  if (zAxis) {
    el("cx-stat-z").textContent        = AXIS_NAMES[zAxis] || zAxis;
    el("cx-stat-z-wrap").style.display = "";
  } else {
    el("cx-stat-z-wrap").style.display = "none";
  }
}

/* ══════════════════════════════════════════════════════════════════════════════
   CLASSIFICATION PAGE
══════════════════════════════════════════════════════════════════════════════ */

let _clsData = null;

async function fetchAndRenderClassify() {
  const benches = readBenches("cls");
  if (!benches.length) { showClsEmpty("Select at least one benchmark."); return; }

  const f        = readFilters("cls");
  const topN     = parseInt(el("cls-top-n").value);
  const testSize = parseInt(el("cls-test-size").value) / 100;
  const params   = new URLSearchParams({
    classifier:    state.classify.classifier,
    target:        el("cls-target").value,
    benchmarks:    benches.join(","),
    origins:       f.origins.join(","),
    min_coverage:  f.minCoverage,
    top_n_classes: topN,
    test_size:     testSize,
  });
  if (f.yearMin) params.set("year_min", f.yearMin);
  if (f.yearMax) params.set("year_max", f.yearMax);

  setLoading("classify", true);
  hideClsEmpty();
  el("classify-results").classList.add("hidden");

  try {
    const data = await fetchJSON("/api/classify?" + params);
    if (data.error) { showClsEmpty(data.error); return; }
    _clsData = data;
    updateClsStats(data);
    renderConfusionMatrix(data);
    renderFeatureImportances(data);
    el("classify-results").classList.remove("hidden");
    if (data.warning) console.warn("[classify]", data.warning);
  } catch (e) { showClsEmpty("Error: " + e.message); }
  finally { setLoading("classify", false); }
}

function updateClsStats(d) {
  el("cls-stat-acc").textContent     = (d.accuracy * 100).toFixed(1) + "%";
  el("cls-stat-train").textContent   = d.n_train;
  el("cls-stat-test").textContent    = d.n_test;
  el("cls-stat-classes").textContent = d.labels.length;
  el("cls-stat-clf").textContent     = d.classifier;
}

function renderConfusionMatrix(data) {
  const { confusion_matrix: cm, labels } = data;

  // Row-normalize (shows recall per class)
  const cmNorm = cm.map(row => {
    const total = row.reduce((a,b) => a+b, 0);
    return row.map(v => total > 0 ? v / total : 0);
  });

  const short  = labels.map(l => l.length > 15 ? l.slice(0, 14) + "…" : l);
  const fsize  = Math.max(8, 15 - Math.floor(labels.length / 2));

  // Build annotations (count + normalized percent)
  const annotations = [];
  for (let i = 0; i < labels.length; i++) {
    for (let j = 0; j < labels.length; j++) {
      annotations.push({
        x: short[j], y: short[i],
        text: `${cm[i][j]}<br><span style="opacity:.65;font-size:${fsize-2}px">${(cmNorm[i][j]*100).toFixed(0)}%</span>`,
        font: { color: "#ffffff", size: fsize },
        showarrow: false,
        align: "center",
      });
    }
  }

  const trace = {
    type: "heatmap",
    z:    cmNorm,
    x:    short,
    y:    short,
    colorscale: [[0, "#09091a"], [0.35, "#2d1a6e"], [1, "#7c6af7"]],
    showscale: true,
    colorbar: { title: { text: "Recall", font: { color: "#fff" } },
                tickfont: { color: "#fff" }, bgcolor: "#141428",
                outlinecolor: "#252540", thickness: 12, len: 0.8 },
    customdata: cm,
    hovertemplate: "True: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: %{customdata}<br>Recall: %{z:.2%}<extra></extra>",
  };

  const layout = {
    ...BASE_LAYOUT,
    annotations,
    xaxis: { ...AXIS_STYLE, title: { text: "Predicted", font: { color: "#fff", size: 12 } }, side: "bottom" },
    yaxis: { ...AXIS_STYLE, title: { text: "True", font: { color: "#fff", size: 12 } }, autorange: "reversed" },
    margin: { l: 130, r: 60, t: 16, b: 100 },
  };

  Plotly.react("chart-confmat", [trace], layout, PLOT_CONFIG);
}

function renderFeatureImportances(data) {
  const panel = el("importance-panel");
  if (!data.feature_importances) {
    el("chart-importance").innerHTML =
      `<div class="no-importance-msg"><p>Feature importances are not available for SVM (RBF kernel).</p>
       <p style="margin-top:8px;font-size:11px;color:var(--text-muted)">Switch to Decision Tree or Logistic Regression to see feature importances.</p></div>`;
    panel.classList.remove("hidden");
    return;
  }

  const entries = Object.entries(data.feature_importances).sort((a,b) => b[1] - a[1]);
  const trace   = {
    type:        "bar",
    orientation: "h",
    x: entries.map(e => e[1]),
    y: entries.map(e => BENCH_NAMES[e[0]] || e[0]),
    marker: { color: "#4ecdc4", opacity: 0.85 },
    hovertemplate: "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
  };

  const layout = {
    ...BASE_LAYOUT,
    xaxis: { ...AXIS_STYLE, title: { text: "Importance", font: { color: "#fff", size: 11 } } },
    yaxis: { ...AXIS_STYLE, tickfont: { color: "#fff", size: 11 }, automargin: true },
    margin: { l: 16, r: 24, t: 16, b: 40 },
  };

  const chartHeight = Math.max(220, entries.length * 32);
  el("chart-importance").style.height = chartHeight + "px";

  panel.classList.remove("hidden");
  Plotly.react("chart-importance", [trace], layout, PLOT_CONFIG);
}

/* ══════════════════════════════════════════════════════════════════════════════
   SHARED TRACE & LAYOUT BUILDERS
══════════════════════════════════════════════════════════════════════════════ */

function buildTraces(points, is3d, colorBy, gX, gY, gZ) {
  switch (colorBy) {
    case "Origin":       return catTraces(points, is3d, p => p.origin, ORIGIN_COLORS, "Unknown", gX, gY, gZ);
    case "Release Year": return yearTraces(points, is3d, gX, gY, gZ);
    case "Company":      return companyTraces(points, is3d, gX, gY, gZ);
    default:             return catTraces(points, is3d, p => p.origin, ORIGIN_COLORS, "Unknown", gX, gY, gZ);
  }
}

function catTraces(points, is3d, getKey, colorMap, fallback, gX, gY, gZ) {
  const groups = new Map();
  for (const p of points) {
    const k = getKey(p) || fallback;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(p);
  }
  return [...groups.keys()]
    .sort((a,b) => {
      if (a==="Other"||a==="Unknown") return 1;
      if (b==="Other"||b==="Unknown") return -1;
      return groups.get(b).length - groups.get(a).length;
    })
    .map(key => {
      const pts  = groups.get(key);
      const col  = colorMap[key] || "#707090";
      const hvr  = pts.map(buildHover);
      const base = { name: key, text: hvr, hovertemplate: "%{text}<extra></extra>" };
      if (is3d) return { ...base, type:"scatter3d", mode:"markers",
        x:pts.map(gX), y:pts.map(gY), z:pts.map(gZ), marker:{color:col,opacity:0.80,size:4} };
      return { ...base, type:"scattergl", mode:"markers",
        x:pts.map(gX), y:pts.map(gY), marker:{color:col,opacity:0.78,size:7} };
    });
}

function yearTraces(points, is3d, gX, gY, gZ) {
  const cv = points.map(p => p.release_year ?? 0);
  const hv = points.map(buildHover);
  const cb = { title:{text:"Year",font:{color:"#fff"}}, thickness:12,
               bgcolor:"#141428", tickfont:{color:"#fff"}, outlinecolor:"#252540" };
  if (is3d) return [{ type:"scatter3d", mode:"markers", name:"Release Year",
    x:points.map(gX), y:points.map(gY), z:points.map(gZ), text:hv, hovertemplate:"%{text}<extra></extra>",
    marker:{size:4,color:cv,colorscale:"Plasma",opacity:0.8,showscale:true,colorbar:cb} }];
  return [{ type:"scattergl", mode:"markers", name:"Release Year",
    x:points.map(gX), y:points.map(gY), text:hv, hovertemplate:"%{text}<extra></extra>",
    marker:{size:7,color:cv,colorscale:"Plasma",opacity:0.78,showscale:true,colorbar:cb} }];
}

function companyTraces(points, is3d, gX, gY, gZ) {
  const counts = {};
  for (const p of points) counts[p.company] = (counts[p.company] || 0) + 1;
  const top20  = new Set(Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,20).map(e=>e[0]));
  const cmap   = { Other: "#505068" };
  let i = 0;
  for (const c of top20) cmap[c] = COMPANY_PALETTE[i++ % COMPANY_PALETTE.length];
  return catTraces(points, is3d, p => top20.has(p.company) ? p.company : "Other",
                   cmap, "Other", gX, gY, gZ);
}

/* ── Layout builder ────────────────────────────────────────────────────────── */

function buildLayout(axisLabels, is3d) {
  const [xL, yL, zL] = axisLabels;
  if (is3d) return { ...BASE_LAYOUT, scene: { bgcolor:"rgba(13,13,26,0.85)",
    xaxis:{...AXIS_STYLE,title:{text:xL,font:AXIS_STYLE.title.font}},
    yaxis:{...AXIS_STYLE,title:{text:yL,font:AXIS_STYLE.title.font}},
    zaxis:{...AXIS_STYLE,title:{text:zL||"Z",font:AXIS_STYLE.title.font}} } };
  return { ...BASE_LAYOUT,
    xaxis:{...AXIS_STYLE,title:{text:xL,font:AXIS_STYLE.title.font}},
    yaxis:{...AXIS_STYLE,title:{text:yL,font:AXIS_STYLE.title.font}} };
}

/* ── Hover text ────────────────────────────────────────────────────────────── */

function buildHover(p) {
  const parts = p.model.split("/");
  const repo  = parts.slice(1).join("/") || parts[0];
  const org   = parts.length > 1 ? parts[0] : "";
  let h = `<b>${esc(repo)}</b>`;
  if (org) h += `<br><span style="color:#9090c0">Org:</span> ${esc(org)}`;
  h += `<br><span style="color:#9090c0">Origin:</span> ${esc(p.origin)}`;
  h += `<br><span style="color:#9090c0">Released:</span> ${p.release_date || "—"}`;
  const scores = Object.entries(p.scores || {}).filter(([,v]) => v !== null).slice(0, 6);
  if (scores.length) {
    h += "<br>";
    for (const [k,v] of scores)
      h += `<br><span style="color:#9090c0">${esc(BENCH_NAMES[k]||k)}:</span> ${v.toFixed(1)}`;
  }
  return h;
}

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

/* ── UI helpers ────────────────────────────────────────────────────────────── */

function el(id)       { return document.getElementById(id); }
function pct(v)       { return (v * 100).toFixed(1) + "%"; }
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function setLoading(page, on) { el(`${page}-loading`).classList.toggle("hidden", !on); }

function showPageEmpty(page, msg) {
  el(`${page}-empty-msg`).textContent = msg;
  el(`${page}-empty`).classList.remove("hidden");
  const chart = el(`chart-${page}`);
  if (chart) chart.innerHTML = "";
}
function hidePageEmpty(page) { el(`${page}-empty`).classList.add("hidden"); }

function showClsEmpty(msg) {
  el("classify-empty-msg").textContent = msg;
  el("classify-empty").classList.remove("hidden");
  el("classify-results").classList.add("hidden");
}
function hideClsEmpty() { el("classify-empty").classList.add("hidden"); }

/* Benchmark checkbox factory */
function makeBenchCheckbox(id, bench, displayName, checked) {
  const lbl = document.createElement("label"); lbl.className = "checkbox-item"; lbl.title = bench.description;
  const cb  = document.createElement("input"); cb.type="checkbox"; cb.id=id; cb.dataset.col=bench.col; cb.checked=checked;
  const wrap = document.createElement("span"); wrap.className="cb-text";
  const main = document.createElement("span"); main.className="cb-main"; main.textContent=displayName;
  const sub  = document.createElement("span"); sub.className="cb-sub";
  for (const area of bench.test_areas) {
    const pill = document.createElement("span");
    pill.className = `area-pill ${AREA_PILL_CLASS[area]||"pill-aggregate"}`;
    pill.textContent = area.replace(/_/g," ");
    sub.appendChild(pill);
  }
  wrap.appendChild(main);
  if (bench.test_areas.length) wrap.appendChild(sub);
  lbl.appendChild(cb); lbl.appendChild(wrap);
  if (!bench.is_primary) {
    const badge = document.createElement("span"); badge.className="badge-sparse";
    badge.textContent = `${bench.null_pct}%`; lbl.appendChild(badge);
  }
  return lbl;
}

/* ── Start ─────────────────────────────────────────────────────────────────── */

init();
