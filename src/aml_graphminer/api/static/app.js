/* AML-GraphMiner — web investigation interface.
 * Vanilla JavaScript, no dependency: works offline and inside the Docker image.
 * Every value coming from the data is inserted with textContent (never innerHTML).
 * The API answers in English with codes and figures; fr.js words them in French. */
"use strict";

(() => {
  // ------------------------------------------------------------------ configuration
  const ROOT = window.location.pathname.replace(/\/ui(\/.*)?$/, "");
  const API = `${ROOT}/api`;
  const PAGE = 25;
  const FR = window.AML_FR;
  const fmt = FR.fmt;
  const state = {
    model: null,
    datasetId: null,
    overview: null,
    alerts: { offset: 0 },
    tx: { offset: 0 },
  };
  let graphCounter = 0;

  // ------------------------------------------------------------------ DOM helpers
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function h(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat(2)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  function s(tag, attrs = {}, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined) continue;
      if (key === "text") node.textContent = String(value);
      else node.setAttribute(key, String(value));
    }
    for (const child of children.flat()) if (child) node.append(child);
    return node;
  }

  function on(node, event, handler) {
    node.addEventListener(event, handler);
    return node;
  }

  // ------------------------------------------------------------------ formatting (fmt: fr.js)
  function threshold() {
    return state.model ? state.model.threshold : 0.5;
  }

  function scoreCell(value, limit = threshold()) {
    const v = Math.max(0, Math.min(1, Number(value) || 0));
    const level = v >= limit ? "high" : v >= limit / 2 ? "mid" : "";
    return h(
      "span",
      { class: "score" },
      h("span", { class: "track" }, h("span", { class: `fill ${level}`, style: `width:${(100 * v).toFixed(1)}%` })),
      fmt.score(v),
    );
  }

  function kv(key, value) {
    return [h("dt", {}, key), h("dd", {}, value)];
  }

  function kpi(label, value, hint = "", cls = "") {
    return h(
      "div",
      { class: `kpi ${cls}` },
      h("div", { class: "label" }, label),
      h("div", { class: "value" }, value),
      hint ? h("div", { class: "hint" }, hint) : null,
    );
  }

  // ------------------------------------------------------------------ API & feedback
  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(`${API}${path}`, options);
    } catch {
      throw new Error(FR.NETWORK_ERROR); // no answer at all (offline, server asleep or stopped)
    }
    const type = response.headers.get("content-type") || "";
    const body = type.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) throw new Error(FR.errorMessage(response.status, body));
    return body;
  }

  function setStatus(text, kind = "") {
    const node = $("#status");
    node.textContent = text;
    node.className = `status ${kind}`;
  }

  async function withOverlay(message, task) {
    $("#overlay-text").textContent = message;
    $("#overlay").hidden = false;
    try {
      return await task();
    } finally {
      $("#overlay").hidden = true;
    }
  }

  function fail(error) {
    console.error(error);
    setStatus(`Erreur : ${error.message}`, "error");
  }

  function showTab(name) {
    $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  }

  function placeholder(text) {
    return h("p", { class: "muted" }, text);
  }

  function errorBox(error) {
    return h("p", { class: "status error" }, `Erreur : ${error.message}`);
  }

  // ------------------------------------------------------------------ model & datasets
  async function loadModel() {
    state.model = await api("/model");
    const m = state.model.model;
    const kind = m.kind ? ` · ${FR.kindName(m.kind)}` : "";
    $("#model-badge").textContent =
      `${FR.modelName(m.model_type)}${kind} · seuil ${fmt.score(state.model.threshold)} · v${state.model.library_version}`;
    renderAbout();
  }

  async function refreshDatasets(preferred = null) {
    const items = await api("/datasets");
    const select = $("#dataset-select");
    select.replaceChildren(
      ...items.map((d) => h("option", { value: d.id }, `${FR.datasetName(d)} — ${fmt.int(d.transactions)} transactions`)),
    );
    const hasData = items.length > 0;
    document.body.classList.toggle("no-data", !hasData);
    ["#btn-delete", "#btn-export-alerts", "#btn-export-scores"].forEach((id) => {
      // while a close is in progress the close button stays disabled (closeDataset re-enables it
      // once the next dataset is loaded): an enabled-looking button must never ignore a click
      $(id).disabled = !hasData || (id === "#btn-delete" && Boolean(state.closing));
    });
    if (!hasData) {
      state.datasetId = null;
      return;
    }
    const ids = items.map((d) => d.id);
    const target = ids.includes(preferred) ? preferred : ids.includes(state.datasetId) ? state.datasetId : ids[0];
    select.value = target;
    await selectDataset(target);
  }

  // An answer that arrives after the user switched to (or closed) another dataset is ignored:
  // otherwise a slow request could fill the tables with the previous dataset's rows.
  const stale = (id) => id !== state.datasetId;

  async function fetchFor(id, path) {
    try {
      const data = await api(path);
      return stale(id) ? null : data;
    } catch (error) {
      if (stale(id)) return null; // e.g. 404: the dataset was closed meanwhile
      throw error;
    }
  }

  async function selectDataset(id) {
    state.datasetId = id;
    state.alerts = { offset: 0 };
    state.tx = { offset: 0 };
    const overview = await fetchFor(id, `/datasets/${id}`);
    if (!overview) return;
    state.overview = overview;
    renderOverview(state.overview);
    loadAlertNetwork(id).catch((error) => $("#overview-graph").replaceChildren(errorBox(error)));
    fillPatternFilter();
    $("#alert-detail").replaceChildren(placeholder("Sélectionnez une alerte pour voir ses preuves et son sous-graphe."));
    $("#tx-detail").replaceChildren(placeholder("Sélectionnez une transaction pour l'expliquer."));
    $("#account-detail").replaceChildren();
    await Promise.all([loadAlerts(), loadTransactions()]);
    if (stale(id)) return;
    const d = state.overview.dataset;
    setStatus(
      `« ${FR.datasetName(d)} » analysé en ${fmt.num(d.analysis_seconds, 2)} s : ${fmt.int(d.n_alerts)} alerte(s) de motifs, ` +
        `${fmt.int(d.n_flagged)} transaction(s) signalée(s) par le modèle.`,
      "ok",
    );
  }

  async function createDemo() {
    try {
      const body = { accounts: 1500, background: 15000, illicit_ratio: 0.02, seed: Math.floor(Math.random() * 1000) };
      const d = await withOverlay("Génération et analyse d'un jeu de démonstration…", () =>
        api("/datasets/demo", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
      );
      await withOverlay("Chargement…", () => refreshDatasets(d.id));
    } catch (error) {
      fail(error);
    }
  }

  async function uploadFile(file) {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("schema", $("#schema-select").value);
    try {
      const d = await withOverlay(`Chargement et analyse de « ${file.name} »…`, () =>
        api("/datasets/upload", { method: "POST", body: form }),
      );
      await withOverlay("Chargement…", () => refreshDatasets(d.id));
    } catch (error) {
      fail(error);
    } finally {
      $("#file-input").value = "";
    }
  }

  async function closeDataset() {
    if (!state.datasetId || state.closing) return;
    // disabled until the next dataset is loaded: a double click must not close two datasets
    state.closing = true;
    $("#btn-delete").disabled = true;
    try {
      await api(`/datasets/${state.datasetId}`, { method: "DELETE" });
      state.datasetId = null;
      await refreshDatasets();
      setStatus("Jeu de données fermé.");
    } catch (error) {
      fail(error);
    } finally {
      state.closing = false;
      $("#btn-delete").disabled = !state.datasetId;
    }
  }

  function exportCsv(kind) {
    if (state.datasetId) window.location.href = `${API}/datasets/${state.datasetId}/export/${kind}.csv`;
  }

  // ------------------------------------------------------------------ charts (inline SVG)
  function tickLabel(v, percent) {
    if (percent) return `${(100 * v).toFixed(0)} %`;
    if (v >= 1e6) return `${fmt.num(v / 1e6, 1)} M`;
    if (v >= 1e3) return `${fmt.num(v / 1e3, 1)} k`;
    return v >= 10 ? String(Math.round(v)) : fmt.num(v, 1);
  }

  function barChart(container, { labels, series, line = null, log = false, marker = null, overlap = false }) {
    const W = 560;
    const H = 230;
    const m = { l: 46, r: line ? 46 : 12, t: 12, b: 34 };
    const iw = W - m.l - m.r;
    const ih = H - m.t - m.b;
    const prepared = series.map((sr) => {
      const total = sr.values.reduce((a, b) => a + b, 0) || 1;
      return { ...sr, data: sr.normalise ? sr.values.map((v) => v / total) : sr.values };
    });
    const percent = prepared.some((p) => p.normalise);
    const maxV = Math.max(1e-9, ...prepared.flatMap((sr) => sr.data));
    const tf = (v) => (log ? Math.log10(1 + v) : v);
    const yMax = tf(maxV) || 1;
    const n = Math.max(labels.length, 1);
    const band = iw / n;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    svg.append(s("line", { x1: m.l, y1: m.t + ih, x2: m.l + iw, y2: m.t + ih, class: "axis" }));
    for (let i = 0; i <= 4; i += 1) {
      const frac = i / 4;
      const y = m.t + ih - frac * ih;
      const value = log ? 10 ** (frac * yMax) - 1 : frac * maxV;
      svg.append(s("text", { x: m.l - 6, y: y + 3, "text-anchor": "end", text: tickLabel(value, percent) }));
      if (i) svg.append(s("line", { x1: m.l, y1: y, x2: m.l + iw, y2: y, class: "axis", "stroke-dasharray": "2 4" }));
    }
    const groups = overlap ? 1 : prepared.length;
    prepared.forEach((sr, k) => {
      const width = (band * 0.86) / groups;
      sr.data.forEach((v, i) => {
        const bh = (tf(v) / yMax) * ih;
        const x = m.l + i * band + band * 0.07 + (overlap ? 0 : k * width);
        const bar = s("rect", {
          x,
          y: m.t + ih - bh,
          width: Math.max(width, 0.5),
          height: Math.max(bh, 0),
          class: sr.cls,
          opacity: overlap && k ? 0.7 : null,
        });
        bar.append(s("title", { text: `${labels[i]} — ${sr.name} : ${percent ? fmt.pct(v, 1) : fmt.int(v)}` }));
        svg.append(bar);
      });
    });
    const every = Math.ceil(n / 12);
    labels.forEach((label, i) => {
      if (i % every === 0) {
        svg.append(s("text", { x: m.l + i * band + band / 2, y: m.t + ih + 14, "text-anchor": "middle", text: label }));
      }
    });
    if (line) {
      const lmax = Math.max(1e-9, ...line.values);
      const points = line.values.map((v, i) => `${m.l + i * band + band / 2},${m.t + ih - (v / lmax) * ih}`).join(" ");
      svg.append(s("polyline", { points, class: "line" }));
      svg.append(s("text", { x: W - m.r + 4, y: m.t + 8, text: fmt.pct(lmax, 1) }));
      svg.append(s("text", { x: W - m.r + 4, y: m.t + ih, text: "0 %" }));
    }
    if (marker) {
      const x = m.l + marker.position * iw;
      svg.append(s("line", { x1: x, y1: m.t, x2: x, y2: m.t + ih, class: "threshold" }));
      svg.append(s("text", { x: x + 4, y: m.t + 10, text: `${marker.label} ${fmt.score(marker.position)}` }));
    }
    const legend = prepared.map((sr) =>
      h("span", {}, h("i", { style: `background:${sr.cls.includes("bad") ? "var(--danger)" : "var(--primary)"}` }), sr.name),
    );
    if (line) legend.push(h("span", {}, h("i", { style: "background:var(--danger);height:2px" }), `${line.name} (axe droit)`));
    if (log) legend.push(h("span", {}, "échelle logarithmique"));
    container.replaceChildren(svg, h("div", { class: "legend" }, legend));
  }

  function hbarChart(container, labels, values, notes = []) {
    const W = 560;
    const rowH = 22;
    const m = { l: 150, r: 130, t: 6, b: 6 };
    const H = m.t + m.b + rowH * Math.max(labels.length, 1);
    const iw = W - m.l - m.r;
    const maxV = Math.max(1, ...values);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    labels.forEach((label, i) => {
      const y = m.t + i * rowH;
      const w = (values[i] / maxV) * iw;
      svg.append(s("text", { x: m.l - 8, y: y + rowH / 2 + 3, "text-anchor": "end", text: label }));
      const bar = s("rect", { x: m.l, y: y + 4, width: Math.max(w, 1), height: rowH - 8, rx: 3, class: "bar" });
      bar.append(s("title", { text: `${label} : ${fmt.int(values[i])}` }));
      svg.append(bar);
      const note = notes[i] ? `  (${notes[i]})` : "";
      svg.append(s("text", { x: m.l + w + 6, y: y + rowH / 2 + 3, text: `${fmt.int(values[i])}${note}` }));
    });
    container.replaceChildren(svg);
  }

  // ------------------------------------------------------------------ graph view
  // Separate groups of accounts (several alerts in one graph) would be pushed far apart by the
  // repulsion and shrunk to dots: each connected component is laid out alone, then the
  // components are packed in rows, largest first.
  function layoutGraph(nodes, links, W, H) {
    const parent = nodes.map((_, i) => i);
    const root = (i) => (parent[i] === i ? i : (parent[i] = root(parent[i])));
    for (const [a, b] of links) parent[root(a)] = root(b);
    const groups = new Map();
    nodes.forEach((_, i) => {
      const r = root(i);
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r).push(i);
    });
    const components = [...groups.values()].sort((a, b) => b.length - a.length);
    if (components.length < 2) {
      forceLayout(nodes, links, W, H);
      return;
    }
    const gap = 40;
    let x = 0;
    let y = 0;
    let rowHeight = 0;
    for (const members of components) {
      const local = new Map(members.map((i, k) => [i, k]));
      const sub = members.map((i) => nodes[i]);
      const subLinks = links.filter(([a]) => local.has(a)).map(([a, b]) => [local.get(a), local.get(b)]);
      const side = Math.max(90, 70 * Math.sqrt(members.length));
      forceLayout(sub, subLinks, side, side);
      const minX = Math.min(...sub.map((n) => n.x));
      const minY = Math.min(...sub.map((n) => n.y));
      const width = Math.max(...sub.map((n) => n.x)) - minX;
      const height = Math.max(...sub.map((n) => n.y)) - minY;
      if (x > 0 && x + width > W) {
        x = 0;
        y += rowHeight + gap;
        rowHeight = 0;
      }
      sub.forEach((n) => {
        n.x += x - minX;
        n.y += y - minY;
      });
      x += width + gap;
      rowHeight = Math.max(rowHeight, height);
    }
  }

  function forceLayout(nodes, links, W, H) {
    const n = nodes.length;
    nodes.forEach((d, i) => {
      const r = 14 * Math.sqrt(i + 0.5);
      const a = i * 2.399963; // golden angle: deterministic, well-spread start
      d.x = W / 2 + r * Math.cos(a);
      d.y = H / 2 + r * Math.sin(a);
    });
    if (n < 2) return;
    const k = 0.8 * Math.sqrt((W * H) / n);
    const iterations = Math.min(350, 100 + 4 * n);
    let temperature = W / 8;
    const cooling = temperature / (iterations + 1);
    const dx = new Float64Array(n);
    const dy = new Float64Array(n);
    for (let it = 0; it < iterations; it += 1) {
      dx.fill(0);
      dy.fill(0);
      for (let i = 0; i < n; i += 1) {
        for (let j = i + 1; j < n; j += 1) {
          let ex = nodes[i].x - nodes[j].x;
          let ey = nodes[i].y - nodes[j].y;
          let d2 = ex * ex + ey * ey;
          if (d2 < 0.01) {
            ex = 0.1 * (i - j);
            ey = 0.1;
            d2 = ex * ex + ey * ey;
          }
          const f = (k * k) / d2; // repulsion k²/d along the unit vector
          dx[i] += ex * f;
          dy[i] += ey * f;
          dx[j] -= ex * f;
          dy[j] -= ey * f;
        }
      }
      for (const [a, b] of links) {
        if (a === b) continue;
        const ex = nodes[a].x - nodes[b].x;
        const ey = nodes[a].y - nodes[b].y;
        const f = Math.sqrt(ex * ex + ey * ey) / k; // attraction d²/k along the unit vector
        dx[a] -= ex * f;
        dy[a] -= ey * f;
        dx[b] += ex * f;
        dy[b] += ey * f;
      }
      for (let i = 0; i < n; i += 1) {
        dx[i] += (W / 2 - nodes[i].x) * 0.06; // gravity keeps components together
        dy[i] += (H / 2 - nodes[i].y) * 0.06;
        const len = Math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]) || 1;
        const step = Math.min(len, temperature);
        nodes[i].x += (dx[i] / len) * step;
        nodes[i].y += (dy[i] / len) * step;
      }
      temperature = Math.max(temperature - cooling, 0.5);
    }
  }

  function colourFor(value, limit) {
    if (value >= limit) return "var(--danger)";
    if (value >= limit / 2) return "var(--warn)";
    return "var(--ok)";
  }

  function graphView(graph, { mode = "score" } = {}) {
    const wrap = h("div", { class: "graph" });
    if (!graph || !graph.nodes || !graph.nodes.length) {
      wrap.append(h("p", { class: "muted", style: "padding:10px" }, "Graphe vide."));
      return wrap;
    }
    const uid = (graphCounter += 1);
    const W = 640;
    const H = 380;
    const limit = threshold();
    const nodes = graph.nodes.map((n) => ({ ...n }));
    const position = new Map(nodes.map((n, i) => [n.id, i]));
    const edges = graph.edges.filter((e) => position.has(e.source) && position.has(e.target));
    layoutGraph(nodes, edges.map((e) => [position.get(e.source), position.get(e.target)]), W, H);

    // sizes and colours
    const maxImpNode = Math.max(1e-12, ...nodes.map((n) => Math.abs(n.importance || 0)));
    const maxImpEdge = Math.max(1e-12, ...edges.map((e) => Math.abs(e.importance || 0)));
    const logs = edges.map((e) => Math.log10((e.amount || 0) + 1));
    const minLog = Math.min(...logs);
    const spanLog = Math.max(...logs) - minLog || 1;
    nodes.forEach((n) => {
      const weight = mode === "importance" ? Math.abs(n.importance || 0) / maxImpNode : n.score || 0;
      n.r = 5 + 9 * Math.max(0, Math.min(1, weight));
    });

    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "graphe de transactions" });
    const defs = s("defs");
    const markers = { grey: "var(--edge)", red: "var(--danger)", blue: "var(--primary)" };
    for (const [name, colour] of Object.entries(markers)) {
      defs.append(
        s(
          "marker",
          { id: `m-${name}-${uid}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" },
          s("path", { d: "M0,0 L10,5 L0,10 z", style: `fill:${colour}` }),
        ),
      );
    }
    svg.append(defs);
    const viewport = s("g");
    svg.append(viewport);
    const edgeLayer = s("g");
    const nodeLayer = s("g");
    viewport.append(edgeLayer, nodeLayer);

    // parallel edges between the same two accounts are drawn as separate curves
    const siblings = new Map();
    edges.forEach((e) => {
      const key = [e.source, e.target].sort().join("\u0000");
      if (!siblings.has(key)) siblings.set(key, []);
      siblings.get(key).push(e);
    });
    edges.forEach((e) => {
      const key = [e.source, e.target].sort().join("\u0000");
      const group = siblings.get(key);
      e.bend = (group.indexOf(e) - (group.length - 1) / 2) * 22 * (e.source < e.target ? 1 : -1);
    });

    const edgeShapes = edges.map((e, i) => {
      let cls = "edge";
      let marker = "grey";
      let width;
      if (mode === "importance") {
        const imp = e.importance || 0;
        if (e.role === "target") {
          cls += " target";
          marker = "red";
          width = 4;
        } else {
          if (imp > 0) {
            cls += " pos";
            marker = "blue";
          }
          width = 1 + (5 * Math.abs(imp)) / maxImpEdge;
        }
      } else {
        if (e.alert) {
          cls += " alert";
          marker = "red";
        }
        width = 1 + (2.5 * (logs[i] - minLog)) / spanLog;
      }
      const path = s("path", { class: cls, "stroke-width": width.toFixed(2), "marker-end": `url(#m-${marker}-${uid})` });
      const details = [
        `transaction ${e.id}`,
        `${e.source} → ${e.target}`,
        `montant ${fmt.amount(e.amount, e.currency)}`,
        `date ${fmt.date(e.timestamp)}`,
        e.score === null || e.score === undefined ? null : `score ${fmt.score(e.score)}`,
        mode === "importance" ? `importance ${fmt.num(e.importance || 0, 4)}` : null,
      ].filter(Boolean);
      path.append(s("title", { text: details.join("\n") }));
      edgeLayer.append(path);
      return path;
    });

    function drawEdges() {
      edges.forEach((e, i) => {
        const a = nodes[position.get(e.source)];
        const b = nodes[position.get(e.target)];
        if (a === b) {
          const r = a.r + 10;
          edgeShapes[i].setAttribute("d", `M${a.x},${a.y - a.r} C${a.x - r},${a.y - 2 * r} ${a.x + r},${a.y - 2 * r} ${a.x},${a.y - a.r}`);
          return;
        }
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const cx = mx - (dy / len) * e.bend;
        const cy = my + (dx / len) * e.bend;
        // stop the curve at the border of the target circle so the arrow stays visible
        const tx = b.x - cx;
        const ty = b.y - cy;
        const tl = Math.sqrt(tx * tx + ty * ty) || 1;
        const ex = b.x - (tx / tl) * (b.r + 2);
        const ey = b.y - (ty / tl) * (b.r + 2);
        edgeShapes[i].setAttribute("d", `M${a.x},${a.y} Q${cx},${cy} ${ex},${ey}`);
      });
    }

    const nodeShapes = nodes.map((n) => {
      const colour =
        mode === "importance"
          ? n.role === "sender" || n.role === "receiver"
            ? "var(--danger)"
            : (n.importance || 0) > 0
              ? "var(--primary)"
              : "var(--edge)"
          : colourFor(n.score || 0, limit);
      const group = s("g", { class: `node${n.highlight ? " highlight" : ""}`, tabindex: 0 });
      const circle = s("circle", { r: n.r.toFixed(1), style: `fill:${colour}` });
      const label = s("text", { "text-anchor": "middle", text: String(n.id).slice(-10) });
      const lines = [
        `compte ${n.id}`,
        n.role ? `rôle : ${FR.roleLabel(n.role)}` : null,
        mode === "importance" ? `importance ${fmt.num(n.importance || 0, 4)}` : `score max ${fmt.score(n.score)}`,
        "clic : profil du compte",
      ].filter(Boolean);
      circle.append(s("title", { text: lines.join("\n") }));
      group.append(circle, label);
      nodeLayer.append(group);
      return { group, circle, label };
    });

    function drawNodes() {
      nodes.forEach((n, i) => {
        const { circle, label } = nodeShapes[i];
        circle.setAttribute("cx", n.x.toFixed(1));
        circle.setAttribute("cy", n.y.toFixed(1));
        label.setAttribute("x", n.x.toFixed(1));
        label.setAttribute("y", (n.y + n.r + 11).toFixed(1));
      });
    }

    // fit the drawing in the frame
    const xs = nodes.map((n) => n.x);
    const ys = nodes.map((n) => n.y);
    const pad = 40;
    let view = {
      x: Math.min(...xs) - pad,
      y: Math.min(...ys) - pad,
      w: Math.max(Math.max(...xs) - Math.min(...xs) + 2 * pad, 200),
      h: Math.max(Math.max(...ys) - Math.min(...ys) + 2 * pad, 150),
    };
    const initial = { ...view };
    const applyView = () => svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`);

    function toGraph(event) {
      const rect = svg.getBoundingClientRect();
      // preserveAspectRatio "xMidYMid meet": one scale, centred
      const scale = Math.min(rect.width / view.w, rect.height / view.h);
      const offsetX = (rect.width - view.w * scale) / 2;
      const offsetY = (rect.height - view.h * scale) / 2;
      return {
        x: view.x + (event.clientX - rect.left - offsetX) / scale,
        y: view.y + (event.clientY - rect.top - offsetY) / scale,
        scale,
      };
    }

    function zoom(factor, centre = null) {
      const cx = centre ? centre.x : view.x + view.w / 2;
      const cy = centre ? centre.y : view.y + view.h / 2;
      view = { x: cx - (cx - view.x) * factor, y: cy - (cy - view.y) * factor, w: view.w * factor, h: view.h * factor };
      applyView();
    }

    // interactions: drag a node, pan the background, wheel to zoom, click to open an account
    let drag = null;
    nodeShapes.forEach(({ group }, i) => {
      on(group, "pointerdown", (event) => {
        event.stopPropagation();
        drag = { kind: "node", index: i, moved: false, start: toGraph(event) };
        svg.setPointerCapture(event.pointerId);
      });
      on(group, "keydown", (event) => {
        if (event.key === "Enter") openAccount(nodes[i].id);
      });
    });
    on(svg, "pointerdown", (event) => {
      drag = { kind: "pan", moved: false, start: toGraph(event), view: { ...view } };
      svg.setPointerCapture(event.pointerId);
    });
    on(svg, "pointermove", (event) => {
      if (!drag) return;
      const p = toGraph(event);
      if (drag.kind === "node") {
        const n = nodes[drag.index];
        if (Math.abs(p.x - drag.start.x) + Math.abs(p.y - drag.start.y) > 2) drag.moved = true;
        n.x = p.x;
        n.y = p.y;
        drawNodes();
        drawEdges();
      } else {
        const scale = p.scale || 1;
        const rect = svg.getBoundingClientRect();
        const dxPix = event.clientX - (drag.lastX ?? event.clientX);
        const dyPix = event.clientY - (drag.lastY ?? event.clientY);
        drag.lastX = event.clientX;
        drag.lastY = event.clientY;
        if (rect.width && (dxPix || dyPix)) {
          drag.moved = true;
          view = { ...view, x: view.x - dxPix / scale, y: view.y - dyPix / scale };
          applyView();
        }
      }
    });
    const release = (event) => {
      if (drag && drag.kind === "node" && !drag.moved) openAccount(nodes[drag.index].id);
      drag = null;
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    };
    on(svg, "pointerup", release);
    on(svg, "pointercancel", release);
    on(svg, "wheel", (event) => {
      event.preventDefault();
      zoom(event.deltaY > 0 ? 1.15 : 1 / 1.15, toGraph(event));
    });

    drawNodes();
    drawEdges();
    applyView();

    const tools = h(
      "div",
      { class: "tools" },
      on(h("button", { type: "button", title: "zoom avant" }, "+"), "click", () => zoom(1 / 1.25)),
      on(h("button", { type: "button", title: "zoom arrière" }, "−"), "click", () => zoom(1.25)),
      on(h("button", { type: "button", title: "recadrer" }, "⟲"), "click", () => {
        view = { ...initial };
        applyView();
      }),
    );
    wrap.append(svg, tools);
    const legend =
      mode === "importance"
        ? [
            h("span", {}, h("i", { style: "background:var(--danger)" }), "transaction expliquée"),
            h("span", {}, h("i", { style: "background:var(--primary)" }), "augmente le score"),
            h("span", {}, h("i", { style: "background:var(--edge)" }), "neutre / diminue"),
            h("span", {}, "épaisseur et taille ∝ importance"),
          ]
        : [
            h("span", {}, h("i", { style: "background:var(--danger)" }), "transaction en alerte"),
            h("span", {}, h("i", { style: "background:var(--edge)" }), "autre transaction"),
            h("span", {}, "taille du nœud ∝ score max · épaisseur ∝ montant"),
          ];
    const container = h("div", {}, wrap, h("div", { class: "graph-legend" }, legend));
    return container;
  }

  // ------------------------------------------------------------------ overview
  function renderOverview(ov) {
    const d = ov.dataset;
    const eda = ov.eda;
    const cards = [
      kpi("Transactions", fmt.int(d.transactions)),
      kpi("Comptes", fmt.int(d.accounts)),
      kpi("Période", `${fmt.day(d.first_timestamp)} → ${fmt.day(d.last_timestamp)}`),
      kpi("Montant total", fmt.money(eda.total_amount)),
      kpi("Alertes de motifs", fmt.int(d.n_alerts), "détecteurs déterministes", "alert"),
      kpi("Signalées par le modèle", fmt.int(d.n_flagged), `score ≥ ${fmt.score(ov.threshold)}`, "alert"),
    ];
    if (d.labelled) cards.push(kpi("Blanchiment (étiquettes)", fmt.int(eda.illicit), fmt.pct(eda.illicit_ratio, 3)));
    $("#kpis").replaceChildren(...cards);

    barChart($("#chart-daily"), {
      labels: eda.daily.map((x) => x.day.slice(5)),
      series: [{ values: eda.daily.map((x) => x.transactions), cls: "bar", name: "transactions" }],
      line: d.labelled
        ? { values: eda.daily.map((x) => (x.transactions ? x.illicit / x.transactions : 0)), name: "part de blanchiment" }
        : null,
    });

    const patterns = Object.entries(ov.alerts_by_pattern || {}).sort((a, b) => b[1] - a[1]);
    if (patterns.length) {
      hbarChart($("#chart-patterns"), patterns.map(([p]) => fmt.pattern(p)), patterns.map(([, v]) => v));
    } else {
      $("#chart-patterns").replaceChildren(placeholder("Aucune alerte de motif."));
    }

    const hist = ov.score_histogram;
    barChart($("#chart-scores"), {
      labels: hist.edges.slice(0, -1).map((e) => fmt.num(Number(e), 2)),
      series: [{ values: hist.counts, cls: "bar", name: "transactions" }],
      log: true,
      marker: { position: Math.max(0, Math.min(1, ov.threshold)), label: "seuil" },
    });

    const am = eda.amounts;
    const amountSeries = [{ values: am.all, cls: "bar", name: "toutes les transactions", normalise: true }];
    if (am.illicit && am.illicit.some((v) => v > 0)) {
      amountSeries.push({ values: am.illicit, cls: "bar bad", name: "blanchiment", normalise: true });
    }
    barChart($("#chart-amounts"), {
      labels: am.log10_bins.slice(0, -1).map((e) => fmt.num(Number(e), 1)),
      series: amountSeries,
      overlap: true,
    });

    const formats = eda.payment_formats;
    hbarChart(
      $("#chart-formats"),
      formats.map((f) => FR.paymentFormat(f.name)),
      formats.map((f) => f.transactions),
      formats.map((f) => (f.illicit_rate === null ? "" : `${fmt.pct(f.illicit_rate, 2)} blanchiment`)),
    );
    renderMetrics(ov);
  }

  function renderMetrics(ov) {
    const box = $("#metrics");
    if (!ov.metrics) {
      box.replaceChildren(
        placeholder(
          "Ce jeu n'est pas entièrement étiqueté : les métriques de détection ne sont pas calculables. " +
            "Les scores, alertes et explications restent disponibles.",
        ),
      );
      return;
    }
    const rows = ["pr_auc", "roc_auc", "precision", "recall", "f1", "recall_at_k", "recall_at_1pct"].map((key) => [
      FR.metricLabel(key),
      key,
    ]);
    const table = h(
      "table",
      {},
      h("thead", {}, h("tr", {}, h("th", {}, "Métrique"), h("th", { class: "num" }, "Modèle"), h("th", { class: "num" }, "Motifs"))),
      h(
        "tbody",
        {},
        rows.map(([label, key]) =>
          h(
            "tr",
            {},
            h("td", {}, label),
            h("td", { class: "num" }, fmt.score(ov.metrics[key])),
            h("td", { class: "num" }, fmt.score(ov.motif_metrics ? ov.motif_metrics[key] : null)),
          ),
        ),
      ),
    );
    box.replaceChildren(
      table,
      h(
        "p",
        { class: "muted small" },
        "Calculées sur tout le jeu chargé, au seuil du modèle (motifs : « signalée par au moins une alerte »). " +
          "Si ce jeu a servi à l'entraînement, ces chiffres sont optimistes : utilisez « amlgm benchmark » pour une " +
          "évaluation sur un découpage temporel.",
      ),
    );
  }

  // ------------------------------------------------------------------ tables (shared)
  function pager(container, slot, total, reload) {
    const from = total ? slot.offset + 1 : 0;
    const to = Math.min(slot.offset + PAGE, total);
    const prev = on(h("button", { type: "button", disabled: slot.offset === 0 }, "‹ Précédent"), "click", () => {
      slot.offset = Math.max(0, slot.offset - PAGE);
      reload().catch(fail);
    });
    const next = on(h("button", { type: "button", disabled: slot.offset + PAGE >= total }, "Suivant ›"), "click", () => {
      slot.offset += PAGE;
      reload().catch(fail);
    });
    container.replaceChildren(h("span", {}, `${fmt.int(from)}–${fmt.int(to)} sur ${fmt.int(total)}`), h("span", {}, prev, " ", next));
  }

  function markSelected(table, row) {
    $$("tr.selected", table).forEach((r) => r.classList.remove("selected"));
    if (row) row.classList.add("selected");
  }

  function clickableRow(cells, open) {
    const row = h("tr", { tabindex: 0 }, cells);
    on(row, "click", (event) => {
      if (!event.target.classList.contains("link")) open(row);
    });
    on(row, "keydown", (event) => {
      if (event.key === "Enter") open(row);
    });
    return row;
  }

  function accountLink(account) {
    const link = h("span", { class: "link acct", title: account }, fmt.short(account));
    return on(link, "click", (event) => {
      event.stopPropagation();
      openAccount(account);
    });
  }

  function txMiniTable(rows) {
    return h(
      "table",
      { class: "table" },
      h(
        "thead",
        {},
        h("tr", {}, h("th", {}, "Tx"), h("th", {}, "Date"), h("th", {}, "Émetteur → destinataire"), h("th", { class: "num" }, "Montant"), h("th", {}, "Score")),
      ),
      h(
        "tbody",
        {},
        rows.map((t) =>
          clickableRow(
            [
              h("td", {}, t.tx_id),
              h("td", {}, fmt.date(t.timestamp)),
              h("td", {}, accountLink(t.sender), " → ", accountLink(t.receiver)),
              h("td", { class: "num" }, fmt.amount(t.amount, t.currency)),
              h("td", {}, scoreCell(t.score)),
            ],
            () => openTransaction(t.tx_id),
          ),
        ),
      ),
    );
  }

  function openTransaction(txId) {
    showTab("transactions");
    showTransaction(txId, null);
  }

  function openAccount(account) {
    showTab("accounts");
    $("#account-input").value = account;
    showAccount(account);
  }

  function openAlert(alertId) {
    showTab("alerts");
    showAlert(alertId, null);
  }

  // ------------------------------------------------------------------ alerts
  function fillPatternFilter() {
    const counts = (state.overview && state.overview.alerts_by_pattern) || {};
    $("#alert-pattern").replaceChildren(
      h("option", { value: "" }, "tous"),
      ...Object.keys(counts).map((p) => h("option", { value: p }, `${fmt.pattern(p)} (${counts[p]})`)),
    );
  }

  async function loadAlerts() {
    if (!state.datasetId) return;
    const minScore = Math.min(1, Math.max(0, Number($("#alert-min").value) || 0));
    const params = new URLSearchParams({ limit: PAGE, offset: state.alerts.offset, min_score: minScore });
    const pattern = $("#alert-pattern").value;
    if (pattern) params.set("pattern", pattern);
    const page = await fetchFor(state.datasetId, `/datasets/${state.datasetId}/alerts?${params}`);
    if (!page) return;
    $("#alerts-table tbody").replaceChildren(
      ...page.items.map((a) => {
        const row = clickableRow(
          [
            h("td", {}, a.alert_id),
            h("td", {}, h("span", { class: "badge motif" }, fmt.pattern(a.pattern))),
            h("td", {}, scoreCell(a.score, 0.5)),
            h("td", {}, a.n_tx),
            h("td", {}, a.n_accounts),
            h("td", {}, fmt.date(a.start)),
            h("td", { class: "num" }, fmt.money(a.total_amount)),
          ],
          (r) => showAlert(a.alert_id, r),
        );
        row.title = FR.alertSentence(a);
        return row;
      }),
    );
    $("#alert-count").textContent = `${fmt.int(page.total)} alerte(s)`;
    pager($("#alerts-pager"), state.alerts, page.total, loadAlerts);
    // nothing selected yet: open the strongest alert, so its sub-graph is on screen at once
    if (page.items.length && !$("#alert-detail h2")) showAlert(page.items[0].alert_id, $("#alerts-table tbody tr"));
  }

  // ------------------------------------------------------------------ network of the strongest alerts
  const NETWORK_ALERTS = 5;

  async function loadAlertNetwork(id) {
    const box = $("#overview-graph");
    box.replaceChildren(placeholder("Chargement du graphe…"));
    const page = await fetchFor(id, `/datasets/${id}/alerts?limit=${NETWORK_ALERTS}`);
    if (!page) return;
    if (!page.items.length) {
      box.replaceChildren(placeholder("Aucune alerte de motif : pas de graphe à montrer."));
      return;
    }
    const details = await Promise.all(page.items.map((a) => fetchFor(id, `/datasets/${id}/alerts/${a.alert_id}`)));
    if (details.some((d) => !d)) return;
    // one graph: accounts and transactions of these alerts (an account shared by two alerts once)
    const nodes = new Map();
    const edges = new Map();
    for (const d of details) {
      for (const n of d.graph.nodes) {
        const seen = nodes.get(n.id);
        nodes.set(n.id, seen ? { ...seen, score: Math.max(seen.score, n.score), highlight: seen.highlight || n.highlight } : { ...n });
      }
      for (const e of d.graph.edges) edges.set(e.id, e);
    }
    const motifs = details.map((d) => `n° ${d.alert_id} ${fmt.pattern(d.pattern)}`).join(", ");
    box.replaceChildren(
      graphView({ nodes: [...nodes.values()], edges: [...edges.values()] }, { mode: "score" }),
      h(
        "p",
        { class: "muted small" },
        `${fmt.int(nodes.size)} comptes et ${fmt.int(edges.size)} transactions des ${details.length} alertes les plus fortes ` +
          `(${motifs}). Cliquez un nœud pour le profil du compte ; l'onglet « Alertes » détaille chaque alerte.`,
      ),
    );
  }

  async function showAlert(alertId, row) {
    markSelected($("#alerts-table"), row);
    const pane = $("#alert-detail");
    pane.replaceChildren(placeholder("Chargement…"));
    try {
      const a = await api(`/datasets/${state.datasetId}/alerts/${alertId}`);
      pane.replaceChildren(
        h("h2", {}, `Alerte n° ${a.alert_id} — ${fmt.pattern(a.pattern)}`),
        h("div", { class: "summary" }, `${FR.upperFirst(FR.patternDescription(a.pattern, a.description))}.`),
        h(
          "dl",
          { class: "kv" },
          kv("Score (calibré)", fmt.score(a.score)),
          kv("Période", `${fmt.date(a.start)} → ${fmt.date(a.end)}`),
          kv("Montant", fmt.money(a.total_amount)),
          kv("Comptes", fmt.int(a.accounts.length)),
          kv("Transactions", fmt.int(a.tx_ids.length)),
          FR.evidenceItems(a.evidence).map(([label, value]) => kv(label, value)),
        ),
        h("h4", {}, "Sous-graphe de l'alerte"),
        graphView(a.graph, { mode: "score" }),
        h("h4", {}, "Transactions de l'alerte"),
        txMiniTable(a.transactions),
      );
    } catch (error) {
      pane.replaceChildren(errorBox(error));
    }
  }

  // ------------------------------------------------------------------ transactions
  async function loadTransactions() {
    if (!state.datasetId) return;
    const params = new URLSearchParams({
      limit: PAGE,
      offset: state.tx.offset,
      sort: $("#tx-sort").value,
      descending: $("#tx-order").value === "desc",
      only_alerts: $("#tx-only-alerts").checked,
    });
    const account = $("#tx-account").value.trim();
    if (account) params.set("account", account);
    const page = await fetchFor(state.datasetId, `/datasets/${state.datasetId}/transactions?${params}`);
    if (!page) return;
    $("#tx-table tbody").replaceChildren(
      ...page.items.map((t) =>
        clickableRow(
          [
            h("td", {}, t.tx_id),
            h("td", {}, fmt.date(t.timestamp)),
            h("td", {}, accountLink(t.sender)),
            h("td", {}, accountLink(t.receiver)),
            h("td", { class: "num" }, fmt.amount(t.amount, t.currency)),
            h("td", {}, FR.paymentFormat(t.transaction_type)),
            h("td", {}, scoreCell(t.score), " ", t.alert ? h("span", { class: "badge alert" }, "alerte") : null),
            h(
              "td",
              {},
              t.motif ? h("span", { class: "badge motif" }, fmt.pattern(t.motif)) : "",
              t.label === 1 ? h("span", { class: "badge truth", title: "étiquette : blanchiment" }, "réel") : null,
            ),
          ],
          (row) => showTransaction(t.tx_id, row),
        ),
      ),
    );
    $("#tx-count").textContent = `${fmt.int(page.total)} transaction(s)`;
    pager($("#tx-pager"), state.tx, page.total, loadTransactions);
  }

  async function showTransaction(txId, row) {
    markSelected($("#tx-table"), row);
    const pane = $("#tx-detail");
    pane.replaceChildren(placeholder("Chargement…"));
    try {
      const t = await api(`/datasets/${state.datasetId}/transactions/${txId}`);
      const method = h(
        "select",
        { "aria-label": "Méthode d'explication" },
        h("option", { value: "auto" }, FR.methodLabel("auto")),
        state.model.explain_methods.map((m) => h("option", { value: m }, FR.methodLabel(m))),
      );
      const hops = h("select", { "aria-label": "Rayon du voisinage" }, [1, 2, 3].map((k) => h("option", { value: k, selected: k === 2 }, `${k} saut(s)`)));
      const output = h("div", {});
      const run = on(h("button", { type: "button", class: "primary" }, "Expliquer"), "click", () =>
        explain(t.tx_id, method.value, Number(hops.value), output),
      );
      const truth =
        t.label === null || t.label === undefined
          ? null
          : kv("Étiquette", t.label === 1 ? `blanchiment${t.true_pattern ? ` (${fmt.pattern(t.true_pattern)})` : ""}` : "légitime");
      const parts = [
        h("h2", {}, `Transaction ${t.tx_id}`),
        h(
          "dl",
          { class: "kv" },
          kv("Date", fmt.date(t.timestamp)),
          kv("Émetteur", accountLink(t.sender)),
          kv("Destinataire", accountLink(t.receiver)),
          kv("Montant payé", fmt.amount(t.amount, t.currency)),
          kv("Montant reçu", fmt.amount(t.amount_received, t.currency_received)),
          kv("Format", FR.paymentFormat(t.transaction_type)),
          kv("Score du modèle", scoreCell(t.score)),
          kv("Décision", t.alert ? h("span", { class: "badge alert" }, "alerte") : "non signalée"),
          kv("Motif le plus fort", t.motif ? `${fmt.pattern(t.motif)} (${fmt.score(t.motif_score)})` : "aucun"),
          truth,
        ),
      ];
      if (t.alerts.length) {
        parts.push(
          h("h4", {}, "Alertes de motifs contenant cette transaction"),
          h(
            "ul",
            { class: "links" },
            t.alerts.map((a) =>
              h(
                "li",
                {},
                on(h("span", { class: "link" }, `n° ${a.alert_id} ${fmt.pattern(a.pattern)} (${fmt.score(a.score)})`), "click", () =>
                  openAlert(a.alert_id),
                ),
                " — ",
                FR.alertSentence(a, { label: false }),
              ),
            ),
          ),
        );
      }
      parts.push(
        h("h4", {}, "Pourquoi ce score ?"),
        h(
          "div",
          { class: "explain-bar" },
          h("label", { class: "field compact" }, h("span", {}, "Méthode"), method),
          h("label", { class: "field compact" }, h("span", {}, "Voisinage"), hops),
          run,
        ),
        output,
      );
      pane.replaceChildren(...parts);
      explain(t.tx_id, "auto", 2, output);
    } catch (error) {
      pane.replaceChildren(errorBox(error));
    }
  }

  // items: [name shown, value, importance, tooltip of the name (default: the name)];
  // words: names are French phrases (wrapped) rather than identifiers (one line)
  function importanceList(items, { words = false } = {}) {
    const maxAbs = Math.max(1e-12, ...items.map(([, , v]) => Math.abs(v)));
    return h(
      "div",
      { class: words ? "imp words" : "imp" },
      items.map(([name, value, imp, tooltip]) => [
        h("span", { class: "name", title: tooltip || name }, name),
        h("span", { class: "num" }, value),
        h(
          "span",
          {},
          h("div", {
            class: `bar ${imp < 0 ? "neg" : ""}`,
            style: `width:${Math.max(2, (100 * Math.abs(imp)) / maxAbs).toFixed(1)}%`,
            title: `importance ${fmt.signed(imp, 4)}`,
          }),
        ),
      ]),
    );
  }

  async function explain(txId, method, hops, output) {
    output.replaceChildren(
      placeholder(method === "pgexplainer" ? "Entraînement de PGExplainer sur les alertes les plus fortes…" : "Calcul de l'explication…"),
    );
    try {
      const params = new URLSearchParams({ method, hops, max_edges: 60 });
      const e = await api(`/datasets/${state.datasetId}/transactions/${txId}/explain?${params}`);
      const neighbours = e.edges
        .filter((x) => x.role !== "target")
        .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
        .slice(0, 8);
      const parts = [
        h("div", { class: "summary" }, `${FR.methodLabel(e.method)} — ${FR.explanationSummary(e)}`),
        graphView(e.graph, { mode: "importance" }),
      ];
      if (e.patterns && e.patterns.length) {
        parts.push(h("h4", {}, "Motifs impliqués"), h("ul", { class: "links" }, e.patterns.map((p) => h("li", {}, FR.alertSentence(p)))));
      }
      if (neighbours.length) {
        parts.push(
          h("h4", {}, "Transactions voisines les plus influentes"),
          importanceList(neighbours.map((x) => [`${x.tx_id} : ${fmt.short(x.sender)} → ${fmt.short(x.receiver)}`, fmt.money(x.amount), x.importance])),
        );
      }
      if (e.features && e.features.length) {
        parts.push(
          h("h4", {}, "Attributs les plus influents"),
          importanceList(
            e.features.map((f) => [FR.upperFirst(FR.featureLabel(f.name)), FR.featureValue(f), f.importance, `nom technique : ${f.name}`]),
            { words: true },
          ),
        );
      }
      output.replaceChildren(...parts);
    } catch (error) {
      output.replaceChildren(errorBox(error));
    }
  }

  // ------------------------------------------------------------------ accounts
  async function showAccount(account) {
    const box = $("#account-detail");
    if (!state.datasetId || !account) return;
    box.replaceChildren(placeholder("Chargement…"));
    try {
      const a = await api(`/datasets/${state.datasetId}/accounts/${encodeURIComponent(account)}`);
      const risky = a.max_score >= threshold() ? "alert" : "";
      box.replaceChildren(
        h(
          "div",
          { class: "kpis" },
          kpi("Compte", fmt.short(a.account), a.bank ? `banque ${a.bank}` : ""),
          kpi("Envoyées", fmt.int(a.n_sent), fmt.money(a.amount_sent)),
          kpi("Reçues", fmt.int(a.n_received), fmt.money(a.amount_received)),
          kpi("Contreparties", `${fmt.int(a.counterparties_out)} / ${fmt.int(a.counterparties_in)}`, "sortantes / entrantes"),
          kpi("Score maximal", fmt.score(a.max_score), `${fmt.int(a.n_flagged)} transaction(s) signalée(s)`, risky),
          kpi("Activité", fmt.day(a.first_seen), `→ ${fmt.day(a.last_seen)}`),
        ),
        h(
          "div",
          { class: "grid two" },
          h("article", { class: "card" }, h("h3", {}, "Voisinage (transactions du compte)"), graphView(a.graph, { mode: "score" })),
          h(
            "article",
            { class: "card" },
            h("h3", {}, `Alertes de motifs (${a.alerts.length})`),
            a.alerts.length
              ? h(
                  "ul",
                  { class: "links" },
                  a.alerts.map((x) =>
                    h("li", {}, on(h("span", { class: "link" }, `n° ${x.alert_id} ${fmt.pattern(x.pattern)}`), "click", () => openAlert(x.alert_id)), ` — score ${fmt.score(x.score)}`),
                  ),
                )
              : placeholder("Aucune alerte de motif."),
            h("h3", {}, "Transactions les plus suspectes"),
            txMiniTable(a.transactions.slice(0, 25)),
          ),
        ),
      );
    } catch (error) {
      box.replaceChildren(errorBox(error));
    }
  }

  // ------------------------------------------------------------------ about
  function renderAbout() {
    const m = state.model;
    const hidden = ["model_type", "metrics", "component_metrics"];
    const entries = Object.entries(m.model).filter(([key]) => !hidden.includes(key));
    const box = $("#about-model");
    box.replaceChildren(
      h(
        "dl",
        { class: "kv" },
        kv("Type", FR.modelName(m.model.model_type)),
        entries.map(([key, value]) => kv(...FR.modelField(key, value))),
        kv("Explications", m.explain_methods.map(FR.methodLabel).join(", ")),
        kv("Version", m.library_version),
      ),
    );
    const metrics = m.model.metrics || {};
    const keys = ["pr_auc", "roc_auc", "f1", "precision", "recall", "recall_at_k"].filter((k) => k in metrics);
    if (keys.length) {
      box.append(
        h("h4", {}, "Métriques de test enregistrées avec le modèle"),
        h("dl", { class: "kv" }, keys.map((k) => kv(FR.metricLabel(k), fmt.score(metrics[k])))),
      );
    }
    $("#about-patterns").replaceChildren(
      h(
        "dl",
        { class: "kv" },
        Object.entries(m.patterns).map(([key, text]) => kv(fmt.pattern(key), `${FR.upperFirst(FR.patternDescription(key, text))}.`)),
      ),
    );
  }

  // ------------------------------------------------------------------ start
  async function init() {
    $$(".tabs button").forEach((b) => on(b, "click", () => showTab(b.dataset.tab)));
    on($("#dataset-select"), "change", (event) => withOverlay("Chargement…", () => selectDataset(event.target.value)).catch(fail));
    on($("#btn-demo"), "click", createDemo);
    on($("#file-input"), "change", (event) => uploadFile(event.target.files[0]));
    on($("#btn-delete"), "click", closeDataset);
    on($("#btn-export-alerts"), "click", () => exportCsv("alerts"));
    on($("#btn-export-scores"), "click", () => exportCsv("scores"));
    const reloadAlerts = () => {
      state.alerts.offset = 0;
      loadAlerts().catch(fail);
    };
    on($("#alert-pattern"), "change", reloadAlerts);
    on($("#alert-min"), "change", reloadAlerts);
    const reloadTransactions = () => {
      state.tx.offset = 0;
      loadTransactions().catch(fail);
    };
    ["#tx-sort", "#tx-order", "#tx-only-alerts"].forEach((id) => on($(id), "change", reloadTransactions));
    let timer = null;
    on($("#tx-account"), "input", () => {
      clearTimeout(timer);
      timer = setTimeout(reloadTransactions, 350);
    });
    on($("#account-form"), "submit", (event) => {
      event.preventDefault();
      showAccount($("#account-input").value.trim());
    });
    document.body.classList.add("no-data");
    try {
      await loadModel();
      await withOverlay("Chargement des jeux de données…", () => refreshDatasets());
      if (!state.datasetId) setStatus("Chargez un fichier ou générez un jeu de démonstration pour commencer.");
    } catch (error) {
      fail(error);
    }
  }

  init();
})();
