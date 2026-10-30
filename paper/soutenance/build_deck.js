// Builds AML-GraphMiner_soutenance.pptx (French defence slides) from deck_data.json.
//
//   python make_data.py            # measured numbers -> deck_data.json
//   npm install && node build_deck.js
//   python animate.py AML-GraphMiner_soutenance.pptx   # transitions + entrance animations
//
// Slides carry figures, charts, icons and motif diagrams; the talk itself is in the speaker notes.
// Objects named "a<k>[w]_..." appear in order k when the slide opens (w = wipe, otherwise fade).

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const FIG = path.join(__dirname, "..", "figures");

// ----------------------------------------------------------------------------- palette
const NAVY = "0B1D3A"; // dominant: night navy (title, divider, closing)
const NAVY2 = "16305C";
const TEAL = "00A99D"; // learned models, "done"
const CORAL = "F2545B"; // laundering, alerts, rules
const GOLD = "F5B700"; // highlights
const SKY = "4C9BE8";
const VIOLET = "7B61FF";
const INK = "1B2333";
const MUTED = "5B6577";
const TINT = "EEF3FA"; // card background on white slides
const WHITE = "FFFFFF";
const HEAD = "Cambria";
const BODY = "Calibri";

const W = 13.333;
const H = 7.5;
const fr = (x, d = 3) => x.toFixed(d).replace(".", ",");
const bench = (ds, m) => D.benchmark.find((r) => r.dataset === ds && r.model === m);

// ----------------------------------------------------------------------------- icons
async function iconPng(Icon, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function iconDisc(slide, Icon, x, y, d, bg, fg, name) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg }, objectName: name + "_disc" });
  const pad = d * 0.24;
  slide.addImage({ data: await iconPng(Icon, fg), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad, objectName: name + "_icon" });
}

function title(slide, text, color = INK) {
  slide.addText(text, {
    x: 0.6, y: 0.35, w: W - 1.2, h: 0.85, fontFace: HEAD, fontSize: 34, bold: true,
    color, margin: 0, valign: "middle", isTextBox: true, objectName: "title",
  });
}

function text(slide, value, opts) {
  slide.addText(value, { fontFace: BODY, margin: 0, isTextBox: true, ...opts });
}

function card(slide, x, y, w, h, name, fill = TINT) {
  slide.addShape("roundRect", {
    x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill },
    shadow: { type: "outer", color: "9AA7BD", opacity: 0.35, blur: 6, offset: 2, angle: 90 }, objectName: name,
  });
}

// ----------------------------------------------------------------------------- motif diagrams
function arrow(slide, x1, y1, x2, y2, r, color, name) {
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
  const ux = dx / len, uy = dy / len;
  const sx = x1 + ux * r, sy = y1 + uy * r, ex = x2 - ux * (r + 0.02), ey = y2 - uy * (r + 0.02);
  slide.addShape("line", {
    x: Math.min(sx, ex), y: Math.min(sy, ey), w: Math.max(Math.abs(ex - sx), 0.001), h: Math.max(Math.abs(ey - sy), 0.001),
    flipH: ex < sx, flipV: ey < sy, line: { color, width: 1.75, endArrowType: "triangle" }, objectName: name,
  });
}

// nodes: [[x, y, colour]] relative to a 1 x 1 box; edges: [[i, j]]
function motif(slide, box, nodes, edges, name) {
  const r = 0.11;
  const P = nodes.map(([u, v]) => [box.x + u * box.w, box.y + v * box.h]);
  edges.forEach(([i, j], k) => arrow(slide, P[i][0], P[i][1], P[j][0], P[j][1], r, "8A95A8", `${name}_e${k}`));
  P.forEach(([x, y], k) => slide.addShape("ellipse", {
    x: x - r, y: y - r, w: 2 * r, h: 2 * r, fill: { color: nodes[k][2] || SKY }, line: { color: WHITE, width: 1.5 },
    objectName: `${name}_n${k}`,
  }));
}

// ----------------------------------------------------------------------------- deck
async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.author = "AML-GraphMiner";
  pres.title = "AML-GraphMiner — soutenance";
  pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

  const dark = (s) => { s.background = { color: NAVY }; };
  const light = (s) => { s.background = { color: WHITE }; };
  let n = 1; // the title slide is slide 1 (not numbered)
  const number = (s, color = MUTED) => {
    n += 1;
    text(s, String(n), { x: W - 0.9, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 11, color, align: "right", objectName: "page" });
  };

  // 1 ------------------------------------------------------------------- title
  let s = pres.addSlide(); dark(s);
  motif(s, { x: 8.2, y: 1.0, w: 4.4, h: 5.2 }, [
    [0.15, 0.1, SKY], [0.55, 0.05, SKY], [0.9, 0.25, SKY], [0.3, 0.45, CORAL], [0.75, 0.55, CORAL],
    [0.45, 0.85, CORAL], [0.05, 0.75, SKY], [0.95, 0.85, SKY], [0.6, 0.35, GOLD],
  ], [[0, 1], [1, 2], [3, 4], [4, 5], [5, 3], [6, 3], [2, 8], [8, 4], [5, 7], [1, 8]], "a1_net");
  text(s, "AML-GraphMiner", { x: 0.7, y: 1.7, w: 7.4, h: 1.1, fontFace: HEAD, fontSize: 54, bold: true, color: WHITE, objectName: "a2_title" });
  text(s, "Détection explicable du blanchiment d'argent sur graphes de transactions", {
    x: 0.7, y: 2.85, w: 7.2, h: 1.0, fontSize: 24, color: "7FE3D8", objectName: "a3_sub" });
  text(s, [
    { text: "Projet de fin d'études — Master", options: { breakLine: true } },
    { text: "Unité de recherche Al-Khawarizmi · FST · Université de Nouakchott", options: { breakLine: true } },
    { text: "Ahmed Ahmed Mahmoud Abeideck · Saade Jaavar Boubacar", options: { breakLine: true } },
    { text: "Encadrant académique : Ahmed El Emine Sejad", options: { breakLine: true } },
    { text: "Encadrant professionnel : Mohamedou El Ghotob Cheikh Tourad" },
  ], { x: 0.7, y: 4.5, w: 7.4, h: 1.6, fontSize: 15, color: "C9D3E6", paraSpaceAfter: 4, objectName: "a4_who" });
  s.addNotes("Bonjour. Je présente AML-GraphMiner, une boîte à outils open source qui détecte le blanchiment d'argent dans des graphes de transactions et explique chaque alerte. Le graphe à droite illustre l'idée : le blanchiment se cache dans des structures — ici un cycle en rouge — au milieu de transactions ordinaires.");

  // 2 ------------------------------------------------------------------- problem
  s = pres.addSlide(); light(s); title(s, "Un problème massif… et très rare"); number(s);
  const hi = D.eda["HI-Small"];
  const stats = [
    [fa.FaGlobeAfrica, CORAL, D.unodc, "blanchis dans le monde en 2009 (ONUDC)"],
    [fa.FaSearchDollar, GOLD, fr(hi.illicit_ratio * 100, 2) + " %", "des transactions sont illicites (IBM HI-Small)"],
    [fa.FaExchangeAlt, TEAL, (hi.transactions / 1e6).toFixed(1).replace(".", ",") + " M", "transactions à analyser pour un seul jeu de données"],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.7 + i * 4.1;
    card(s, x, 1.6, 3.7, 4.2, `a${i + 1}_card`);
    await iconDisc(s, stats[i][0], x + 1.35, 1.9, 1.0, stats[i][1], WHITE, `a${i + 1}_ic`);
    text(s, stats[i][2], { x: x + 0.2, y: 3.1, w: 3.3, h: 1.1, fontFace: HEAD, fontSize: 40, bold: true, color: stats[i][1], align: "center", objectName: `a${i + 1}_num` });
    text(s, stats[i][3], { x: x + 0.3, y: 4.3, w: 3.1, h: 1.1, fontSize: 16, color: MUTED, align: "center", objectName: `a${i + 1}_lab` });
  }
  text(s, "Trouver l'aiguille… et expliquer pourquoi c'est une aiguille.", { x: 0.7, y: 6.25, w: 12, h: 0.5, fontSize: 18, italic: true, color: NAVY2, objectName: "a4_tag" });
  s.addNotes("Selon l'ONUDC, environ 1 600 milliards de dollars ont été blanchis en 2009, 2,7 % du PIB mondial. Mais à l'échelle d'une banque, c'est une aiguille dans une botte de foin : 0,10 % des transactions du jeu IBM HI-Small, sur plus de 5 millions. Les systèmes à règles noient les analystes sous les fausses alertes ; les réseaux de neurones sur graphes sont prometteurs mais peu reproductibles et peu explicables. Or le GAFI et la BCEAO exigent des décisions justifiables.");

  // 3 ------------------------------------------------------------------- objectives
  s = pres.addSlide(); light(s); title(s, "Six objectifs du cahier des charges"); number(s);
  const objs = [
    [fa.FaDatabase, SKY, "O1", "Ingestion & graphes"],
    [fa.FaProjectDiagram, CORAL, "O2", "Motifs de blanchiment"],
    [fa.FaBrain, VIOLET, "O3", "GNN : GCN · SAGE · GAT"],
    [fa.FaEye, GOLD, "O4", "Explicabilité"],
    [fa.FaChartLine, TEAL, "O5", "Évaluation rigoureuse"],
    [fa.FaDesktop, NAVY2, "O6", "API · CLI · Web"],
  ];
  for (let i = 0; i < 6; i++) {
    const x = 0.7 + (i % 3) * 4.1, y = 1.55 + Math.floor(i / 3) * 2.75;
    card(s, x, y, 3.7, 2.4, `a${i + 1}_c`);
    await iconDisc(s, objs[i][0], x + 0.3, y + 0.35, 1.0, objs[i][1], WHITE, `a${i + 1}_i`);
    text(s, objs[i][2], { x: x + 1.55, y: y + 0.35, w: 1.9, h: 0.6, fontFace: HEAD, fontSize: 30, bold: true, color: objs[i][1], objectName: `a${i + 1}_o` });
    text(s, objs[i][3], { x: x + 0.3, y: y + 1.55, w: 3.2, h: 0.6, fontSize: 18, bold: true, color: INK, objectName: `a${i + 1}_t` });
  }
  s.addNotes("Le cahier des charges fixe six objectifs : ingérer les données et construire les graphes, détecter les motifs classiques, entraîner trois GNN, expliquer les alertes, évaluer rigoureusement face aux modèles de référence, et fournir des interfaces : API REST, ligne de commande et, en plus, une interface web d'investigation.");

  // 4 ------------------------------------------------------------------- architecture
  s = pres.addSlide(); light(s); title(s, "Une architecture en six couches"); number(s);
  const layers = [
    [fa.FaDatabase, SKY, "Ingestion", "IBM AML · Elliptic · CSV/JSON/Parquet · Pydantic"],
    [fa.FaStream, "3FB6D1", "Caractéristiques", "fenêtres causales 1 h · 24 h · 7 j · PageRank"],
    [fa.FaProjectDiagram, TEAL, "Graphes", "NetworkX · PyTorch Geometric · Neo4j"],
    [fa.FaShieldAlt, CORAL, "Détecteurs", "7 motifs · tabulaires · GNN · hybride"],
    [fa.FaEye, GOLD, "Explications", "motifs · occlusion · GNNExplainer · Captum"],
    [fa.FaDesktop, VIOLET, "Interfaces", "web · REST · CLI · notebooks · MLflow · Hub"],
  ];
  for (let i = 0; i < 6; i++) {
    const y = 6.35 - i * 0.92; // bottom-up
    s.addShape("roundRect", { x: 1.6, y, w: 10.2, h: 0.78, rectRadius: 0.1, fill: { color: layers[i][1] }, line: { color: layers[i][1] }, objectName: `a${i + 1}w_bar` });
    s.addImage({ data: await iconPng(layers[i][0], WHITE), x: 1.8, y: y + 0.14, w: 0.5, h: 0.5, objectName: `a${i + 1}w_ic` });
    text(s, layers[i][2], { x: 2.5, y: y + 0.1, w: 2.8, h: 0.58, fontFace: HEAD, fontSize: 20, bold: true, color: WHITE, valign: "middle", objectName: `a${i + 1}w_n` });
    text(s, layers[i][3], { x: 5.3, y: y + 0.1, w: 6.3, h: 0.58, fontSize: 16, color: WHITE, valign: "middle", objectName: `a${i + 1}w_d` });
  }
  text(s, "▲", { x: 0.7, y: 1.5, w: 0.6, h: 0.5, fontSize: 24, color: MUTED, align: "center", objectName: "a7_up" });
  text(s, "chaque couche n'utilise que celles du dessous", { x: 0.45, y: 2.0, w: 1.1, h: 3.5, fontSize: 12, color: MUTED, align: "center", objectName: "a7_txt" });
  s.addNotes("Le code est organisé en six couches ; chacune ne dépend que de celles du dessous. Les dépendances lourdes, PyTorch ou FastAPI, sont optionnelles : le cœur tourne sur un simple portable. Tous les détecteurs partagent la même interface — fit, score, save, load — ce qui permet de les servir, de les comparer et de les combiner.");

  // 5 ------------------------------------------------------------------- data
  s = pres.addSlide(); light(s); title(s, "Trois jeux de données de référence"); number(s);
  const li = D.eda["LI-Small"];
  const sets = [
    [fa.FaUniversity, CORAL, "IBM HI-Small", (hi.transactions / 1e6).toFixed(2).replace(".", ",") + " M", "transactions", fr(hi.illicit_ratio * 100, 2) + " % illicites"],
    [fa.FaUniversity, SKY, "IBM LI-Small", (li.transactions / 1e6).toFixed(2).replace(".", ",") + " M", "transactions", fr(li.illicit_ratio * 100, 2) + " % illicites"],
    [fa.FaBitcoin, GOLD, "Elliptic", "49", "pas de temps (Bitcoin)", "6,5 % illicites (test)"],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.7 + i * 4.1;
    card(s, x, 1.55, 3.7, 3.9, `a${i + 1}_c`);
    await iconDisc(s, sets[i][0], x + 0.3, 1.8, 0.8, sets[i][1], WHITE, `a${i + 1}_i`);
    text(s, sets[i][2], { x: x + 1.25, y: 1.85, w: 2.3, h: 0.7, fontFace: HEAD, fontSize: 20, bold: true, color: INK, valign: "middle", objectName: `a${i + 1}_n` });
    text(s, sets[i][3], { x: x + 0.2, y: 2.85, w: 3.3, h: 1.0, fontFace: HEAD, fontSize: 48, bold: true, color: sets[i][1], align: "center", objectName: `a${i + 1}_v` });
    text(s, sets[i][4], { x: x + 0.2, y: 3.8, w: 3.3, h: 0.4, fontSize: 16, color: MUTED, align: "center", objectName: `a${i + 1}_u` });
    text(s, sets[i][5], { x: x + 0.2, y: 4.45, w: 3.3, h: 0.5, fontSize: 18, bold: true, color: INK, align: "center", objectName: `a${i + 1}_r` });
  }
  await iconDisc(s, fa.FaLightbulb, 0.7, 5.85, 0.75, GOLD, WHITE, "a4_tip");
  text(s, [
    { text: "Indice clé : ", options: { bold: true, color: INK } },
    { text: "les virements ACH portent 0,75 % de blanchiment, les autres formats ≤ 0,04 %", options: { color: INK } },
  ], { x: 1.65, y: 5.85, w: 11, h: 0.75, fontSize: 18, valign: "middle", objectName: "a4_tiptxt" });
  s.addNotes("Trois jeux publics : IBM HI-Small et LI-Small, des transactions bancaires simulées avec leurs motifs de blanchiment annotés, et Elliptic, un graphe Bitcoin. L'analyse exploratoire a révélé des indices décisifs, par exemple que le blanchiment passe presque toujours par des virements ACH.");

  // 6 ------------------------------------------------------------------- protocol
  s = pres.addSlide(); light(s); title(s, "Un protocole sans fuite de données"); number(s);
  const segs = [[TEAL, 60, "Entraînement", "apprendre"], [GOLD, 20, "Validation", "régler"], [CORAL, 20, "Test", "mesurer"]];
  let xs = 0.7;
  for (let i = 0; i < 3; i++) {
    const w = (11.9 * segs[i][1]) / 100;
    s.addShape("rect", { x: xs, y: 1.9, w: w - 0.05, h: 1.0, fill: { color: segs[i][0] }, line: { color: segs[i][0] }, objectName: `a${i + 1}w_seg` });
    text(s, `${segs[i][2]} · ${segs[i][1]} %`, { x: xs + 0.15, y: 1.9, w: w - 0.3, h: 1.0, fontSize: 20, bold: true, color: WHITE, valign: "middle", objectName: `a${i + 1}w_lab` });
    text(s, segs[i][3], { x: xs, y: 3.0, w: w, h: 0.45, fontSize: 16, italic: true, color: MUTED, align: "center", objectName: `a${i + 1}w_verb` });
    xs += w;
  }
  text(s, "temps →", { x: 11.6, y: 3.45, w: 1.0, h: 0.4, fontSize: 14, color: MUTED, align: "right", objectName: "a3w_time" });
  const rules = [
    [fa.FaHistory, TEAL, "Caractéristiques causales", "jamais d'information future"],
    [fa.FaSlidersH, GOLD, "Seuils et réglages", "choisis sur la validation"],
    [fa.FaDice, VIOLET, "3 graines", "moyenne ± écart-type"],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.7 + i * 4.1;
    await iconDisc(s, rules[i][0], x, 4.35, 0.9, rules[i][1], WHITE, `a${i + 4}_r`);
    text(s, rules[i][2], { x: x + 1.1, y: 4.35, w: 2.9, h: 0.5, fontSize: 18, bold: true, color: INK, objectName: `a${i + 4}_t` });
    text(s, rules[i][3], { x: x + 1.1, y: 4.85, w: 2.9, h: 0.5, fontSize: 16, color: MUTED, objectName: `a${i + 4}_d` });
  }
  s.addNotes("Pour des résultats comparables, le découpage est temporel : 60 % pour apprendre, 20 % pour régler, 20 % pour mesurer, une seule fois. Les caractéristiques n'utilisent jamais le futur — des tests unitaires le vérifient en modifiant les données futures — et chaque modèle est entraîné sur trois graines.");

  // 7 ------------------------------------------------------------------- motifs
  s = pres.addSlide(); light(s); title(s, "Sept motifs détectés par des règles"); number(s);
  const pp = D.per_pattern["HI-Small"];
  const motifs = [
    ["Fan-in", [[0.1, 0.1], [0.1, 0.5], [0.1, 0.9], [0.85, 0.5, CORAL]], [[0, 3], [1, 3], [2, 3]], pp.fan_in],
    ["Fan-out", [[0.15, 0.5, CORAL], [0.9, 0.1], [0.9, 0.5], [0.9, 0.9]], [[0, 1], [0, 2], [0, 3]], pp.fan_out],
    ["Cycle", [[0.5, 0.08, CORAL], [0.92, 0.75, CORAL], [0.08, 0.75, CORAL]], [[0, 1], [1, 2], [2, 0]], pp.cycle],
    ["Scatter-gather", [[0.08, 0.5], [0.5, 0.1], [0.5, 0.5], [0.5, 0.9], [0.92, 0.5, CORAL]], [[0, 1], [0, 2], [0, 3], [1, 4], [2, 4], [3, 4]], pp.scatter_gather],
    ["Gather-scatter", [[0.08, 0.15], [0.08, 0.85], [0.5, 0.5, CORAL], [0.92, 0.15], [0.92, 0.85]], [[0, 2], [1, 2], [2, 3], [2, 4]], pp.gather_scatter],
    ["Relais (stack)", [[0.08, 0.5], [0.5, 0.5, CORAL], [0.92, 0.5]], [[0, 1], [1, 2]], pp.stack],
    ["Communauté", [[0.2, 0.2, CORAL], [0.8, 0.2, CORAL], [0.85, 0.8, CORAL], [0.15, 0.8, CORAL], [0.5, 0.5, GOLD]], [[0, 4], [4, 1], [2, 4], [4, 3], [0, 1], [2, 3]], null],
  ];
  for (let i = 0; i < 7; i++) {
    const col = i % 4, row = Math.floor(i / 4);
    const x = 0.7 + col * 3.05 + (row === 1 ? 1.5 : 0), y = 1.45 + row * 2.85;
    card(s, x, y, 2.75, 2.6, `a${i + 1}_c`);
    motif(s, { x: x + 0.45, y: y + 0.25, w: 1.85, h: 1.25 }, motifs[i][1], motifs[i][2], `a${i + 1}_m`);
    text(s, motifs[i][0], { x: x + 0.1, y: y + 1.65, w: 2.55, h: 0.4, fontSize: 17, bold: true, color: INK, align: "center", objectName: `a${i + 1}_t` });
    const r = motifs[i][3];
    text(s, r === null ? "groupes de comptes relais" : `trouvé : ${Math.round(r * 100)} %`, {
      x: x + 0.1, y: y + 2.05, w: 2.55, h: 0.35, fontSize: 14, color: r === null ? MUTED : TEAL, bold: r !== null, align: "center", objectName: `a${i + 1}_r` });
  }
  s.addNotes("Le scanner détecte sept structures sans aucun apprentissage : collecte (fan-in), dispersion (fan-out), cycles, scatter-gather, gather-scatter, relais — le motif « stack » d'IBM — et communautés de comptes relais. Chaque alerte garde ses preuves : comptes, transactions, montants convertis en dollars. Le pourcentage indique la part des transactions de chaque motif annoté retrouvée par son propre détecteur sur le test HI-Small.");

  // 8 ------------------------------------------------------------------- J2 journey
  s = pres.addSlide(); light(s); title(s, "Objectif J2 : des règles au-delà de 0,5"); number(s);
  const steps = D.motif_history;
  const colors = ["9AA7BD", SKY, VIOLET, TEAL];
  for (let i = 0; i < steps.length; i++) {
    const d = 1.35 + steps[i].value * 3.1; // disc size grows with the PR-AUC
    const cx = 1.6 + i * 3.15, cy = 3.35;
    s.addShape("ellipse", { x: cx - d / 2, y: cy - d / 2, w: d, h: d, fill: { color: colors[i] }, line: { color: WHITE, width: 2 }, objectName: `a${i + 1}_d` });
    text(s, fr(steps[i].value), { x: cx - 1.2, y: cy - 0.4, w: 2.4, h: 0.8, fontFace: HEAD, fontSize: i === 3 ? 40 : 26, bold: true, color: WHITE, align: "center", valign: "middle", objectName: `a${i + 1}_v` });
    text(s, steps[i].step, { x: cx - 1.4, y: 5.25, w: 2.8, h: 0.8, fontSize: 16, color: INK, bold: i === 3, align: "center", objectName: `a${i + 1}_s` });
    if (i < steps.length - 1) {
      text(s, "➜", { x: cx + 1.0, y: cy - 0.3, w: 1.2, h: 0.6, fontSize: 28, color: MUTED, align: "center", objectName: `a${i + 1}_arr` });
    }
  }
  text(s, `PR-AUC test HI-Small · intervalle à 95 % : ${fr(D.motif_ci[0])} – ${fr(D.motif_ci[1])}`, {
    x: 0.7, y: 6.25, w: 12, h: 0.5, fontSize: 16, italic: true, color: NAVY2, objectName: "a5_ci" });
  s.addNotes("Le jalon J2 demandait une PR-AUC supérieure à 0,5 avec les seuls motifs. Au départ : 0,005. Le diagnostic a montré que les motifs IBM changent de devise à chaque saut et durent plusieurs jours : conversion des montants et contexte historique, puis restriction aux virements ACH : 0,340. Ensuite, fans plus longs et exclusions : 0,399. Enfin un détecteur de relais et une précision calibrée par tranche de score : 0,503, avec un intervalle de confiance de 0,479 à 0,526 — l'objectif est atteint, de justesse. Tous les réglages ont été choisis sur la validation ; 36 % du blanchiment n'appartient à aucun motif, ce qui plafonne toute approche à base de règles.");

  // 9 ------------------------------------------------------------------- results IBM
  s = pres.addSlide(); light(s); title(s, "Résultats IBM AML : PR-AUC sur le test"); number(s);
  const names = { sage: "GraphSAGE", rf: "Forêt aléatoire", hgb: "HistGB", xgb: "XGBoost", gat: "GAT", gcn: "GCN", patterns: "Motifs (règles)", logreg: "Rég. logistique" };
  const order = ["sage", "patterns", "rf", "hgb", "xgb", "gat", "gcn", "logreg"];
  for (const [k, ds] of ["HI-Small", "LI-Small"].entries()) {
    const rows = order.map((m) => [names[m], bench(ds, m)]).sort((a, b) => a[1].mean - b[1].mean);
    const learned = rows.map(([, r]) => (r.model === "patterns" ? 0 : r.mean));
    const rules = rows.map(([, r]) => (r.model === "patterns" ? r.mean : 0));
    s.addChart(pres.charts.BAR, [
      { name: "Modèles appris", labels: rows.map((r) => r[0]), values: learned },
      { name: "Règles", labels: rows.map((r) => r[0]), values: rules },
    ], {
      x: 0.5 + k * 6.3, y: 1.35, w: 6.2, h: 5.0, barDir: "bar", barGrouping: "stacked",
      chartColors: [TEAL, CORAL], showValue: true, dataLabelPosition: "inEnd", dataLabelColor: WHITE,
      dataLabelFormatCode: "0.000;-0.000;;", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
      showTitle: true, title: `${ds}${ds === "HI-Small" ? " (0,18 % illicites)" : " (0,07 % illicites)"}`,
      titleFontSize: 16, titleColor: INK, titleFontFace: "+mj-lt",
      catAxisLabelColor: INK, catAxisLabelFontSize: 13, catAxisLabelFontFace: "+mn-lt",
      valAxisMaxVal: 0.6, valAxisMinVal: 0, valAxisLabelColor: MUTED, valAxisLabelFontSize: 11, valAxisLabelFormatCode: "0.0",
      valGridLine: { color: "E1E6EF", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false,
      objectName: `a${k + 1}w_chart`,
    });
  }
  text(s, "Objectif du cahier des charges : 0,5 — atteint par GraphSAGE et par les règles seules sur HI-Small", {
    x: 0.7, y: 6.5, w: 12, h: 0.45, fontSize: 15, italic: true, color: NAVY2, objectName: "a3_obj" });
  s.addNotes("Sur HI-Small, GraphSAGE obtient 0,531 en moyenne sur trois graines, juste devant les motifs (0,503) et la forêt aléatoire (0,499). Sur LI-Small, beaucoup plus difficile avec 0,07 % de transactions illicites, la forêt aléatoire mène (0,191), les motifs suivent de près (0,189), devant HistGB, GCN et GAT.");

  // 10 ------------------------------------------------------------------- hybrid
  s = pres.addSlide(); light(s); title(s, "Hybride : un modèle + les motifs"); number(s);
  const hy = D.hybrid.filter((r) => r.dataset === "HI-Small");
  const bn = { rf: "Forêt aléatoire", hgb: "HistGB", sage: "GraphSAGE" };
  s.addChart(pres.charts.BAR, [
    { name: "Modèle seul", labels: hy.map((r) => bn[r.base]), values: hy.map((r) => r.alone) },
    { name: "Motifs seuls", labels: hy.map((r) => bn[r.base]), values: hy.map((r) => r.motifs) },
    { name: "Hybride", labels: hy.map((r) => bn[r.base]), values: hy.map((r) => r.hybrid) },
  ], {
    x: 0.5, y: 1.35, w: 8.0, h: 5.4, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: ["9AA7BD", CORAL, TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
    showTitle: true, title: "PR-AUC test · IBM HI-Small", titleFontSize: 16, titleColor: INK, titleFontFace: "+mj-lt",
    catAxisLabelColor: INK, catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt",
    valAxisMaxVal: 0.75, valAxisMinVal: 0, valAxisLabelColor: MUTED, valAxisLabelFontSize: 11, valAxisLabelFormatCode: "0.0",
    valGridLine: { color: "E1E6EF", size: 0.75 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 13, legendColor: INK, legendFontFace: "+mn-lt", objectName: "a1w_chart",
  });
  const best = Math.max(...hy.map((r) => r.hybrid));
  const liRf = D.hybrid.find((r) => r.dataset === "LI-Small" && r.base === "rf");
  card(s, 8.9, 1.6, 3.8, 2.3, "a2_c1", NAVY);
  text(s, fr(best), { x: 9.0, y: 1.75, w: 3.6, h: 1.2, fontFace: HEAD, fontSize: 60, bold: true, color: "7FE3D8", align: "center", objectName: "a2_v1" });
  text(s, "meilleure PR-AUC sur HI-Small", { x: 9.0, y: 2.95, w: 3.6, h: 0.7, fontSize: 16, color: WHITE, align: "center", objectName: "a2_l1" });
  card(s, 8.9, 4.2, 3.8, 2.3, "a3_c2", TINT);
  text(s, `${fr(liRf.alone)} → ${fr(liRf.hybrid)}`, { x: 9.0, y: 4.4, w: 3.6, h: 1.0, fontFace: HEAD, fontSize: 36, bold: true, color: TEAL, align: "center", objectName: "a3_v2" });
  text(s, "LI-Small : forêt aléatoire + motifs", { x: 9.0, y: 5.4, w: 3.6, h: 0.7, fontSize: 16, color: INK, align: "center", objectName: "a3_l2" });
  s.addNotes("Les deux approches sont complémentaires. Un modèle déjà entraîné est combiné aux motifs par une régression logistique à trois coefficients, apprise sur la validation. Sur HI-Small, chaque combinaison dépasse ses deux parties : jusqu'à 0,641 avec HistGB ou GraphSAGE. Sur LI-Small, la forêt aléatoire passe de 0,191 à 0,236. Ces modèles viennent de Kaggle, sans réentraînement, sur une seule graine.");

  // 11 ------------------------------------------------------------------- elliptic
  s = pres.addSlide(); light(s); title(s, "Elliptic Bitcoin : les arbres dominent"); number(s);
  const em = ["hgb", "xgb", "rf", "sage", "gat", "gcn", "logreg"].map((m) => [names[m], bench("elliptic", m)]);
  s.addChart(pres.charts.BAR, [{ name: "PR-AUC", labels: em.map((r) => r[0]), values: em.map((r) => r[1].mean) }], {
    x: 0.5, y: 1.35, w: 8.2, h: 5.4, barDir: "col", chartColors: [GOLD], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.000", dataLabelFontSize: 12, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
    showTitle: true, title: "PR-AUC test · pas de temps 35–49", titleFontSize: 16, titleColor: INK, titleFontFace: "+mj-lt",
    catAxisLabelColor: INK, catAxisLabelFontSize: 13, catAxisLabelFontFace: "+mn-lt",
    valAxisMaxVal: 1, valAxisMinVal: 0, valAxisLabelColor: MUTED, valAxisLabelFontSize: 11, valAxisLabelFormatCode: "0.0",
    valGridLine: { color: "E1E6EF", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false, objectName: "a1w_chart",
  });
  await iconDisc(s, fa.FaTree, 9.3, 1.9, 1.0, TEAL, WHITE, "a2_i1");
  text(s, "≈ 0,79", { x: 10.5, y: 1.85, w: 2.4, h: 0.7, fontFace: HEAD, fontSize: 36, bold: true, color: TEAL, objectName: "a2_v1" });
  text(s, "HistGB · XGBoost · forêt", { x: 10.5, y: 2.55, w: 2.5, h: 0.5, fontSize: 15, color: MUTED, objectName: "a2_l1" });
  await iconDisc(s, fa.FaBrain, 9.3, 3.6, 1.0, VIOLET, WHITE, "a3_i2");
  text(s, fr(bench("elliptic", "sage").mean), { x: 10.5, y: 3.55, w: 2.4, h: 0.7, fontFace: HEAD, fontSize: 36, bold: true, color: VIOLET, objectName: "a3_v2" });
  text(s, "meilleur GNN (GraphSAGE)", { x: 10.5, y: 4.25, w: 2.5, h: 0.5, fontSize: 15, color: MUTED, objectName: "a3_l2" });
  text(s, "Pas de modèle universel : l'outil compare toujours les deux familles.", { x: 9.2, y: 5.3, w: 3.8, h: 1.1, fontSize: 16, italic: true, color: NAVY2, objectName: "a4_t" });
  s.addNotes("Sur Elliptic, les modèles à base d'arbres atteignent environ 0,79 alors que le meilleur GNN reste à 0,497. C'est une raison de plus pour toujours comparer les familles de modèles sous le même protocole, ce que fait l'outil.");

  // 12 ------------------------------------------------------------------- explanations
  s = pres.addSlide(); light(s); title(s, "Expliquer chaque alerte"); number(s);
  card(s, 0.6, 1.4, 6.3, 4.9, "a1_c", WHITE);
  s.addImage({ path: path.join(FIG, "example_explanation.png"), x: 0.85, y: 1.6, w: 5.8, h: 3.87, objectName: "a1_img" });
  text(s, "Preuve exacte d'un cycle : la transaction en rouge boucle l'argent", { x: 0.85, y: 5.55, w: 5.8, h: 0.6, fontSize: 13, italic: true, color: MUTED, objectName: "a1_cap" });
  const ex = [
    [fa.FaLayerGroup, VIOLET, "6 méthodes", "motifs · occlusion · GNNExplainer · PGExplainer · Captum · attention"],
    [fa.FaBalanceScale, GOLD, "Effet du voisinage mesuré", `score ${D.neighbourhood.with} avec, ${D.neighbourhood.without} sans`],
    [fa.FaFolderOpen, TEAL, "Dossier d'alertes", "les K alertes les plus fortes expliquées en une commande"],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.55 + i * 1.6;
    await iconDisc(s, ex[i][0], 7.3, y, 0.95, ex[i][1], WHITE, `a${i + 2}_i`);
    text(s, ex[i][2], { x: 8.45, y: y, w: 4.4, h: 0.5, fontSize: 19, bold: true, color: INK, objectName: `a${i + 2}_t` });
    text(s, ex[i][3], { x: 8.45, y: y + 0.5, w: 4.4, h: 0.8, fontSize: 15, color: MUTED, objectName: `a${i + 2}_d` });
  }
  s.addNotes("Chaque alerte est expliquée. Pour les motifs, la preuve est exacte : ici un cycle de trois comptes. Pour les modèles, six méthodes sont disponibles. Point important : l'outil mesure l'effet réel du voisinage. Pour GraphSAGE, retirer les transactions voisines ne change presque pas le score, ce qui explique pourquoi PGExplainer rend des masques vides : c'est la bonne réponse, et Captum devient alors la vue la plus utile.");

  // 13 ------------------------------------------------------------------- interfaces & quality
  s = pres.addSlide(); light(s); title(s, "Des interfaces prêtes à l'emploi, un code testé"); number(s);
  const ifs = [
    [fa.FaDesktop, VIOLET, "Interface web", "tableau de bord · alertes · comptes"],
    [fa.FaTerminal, NAVY2, "CLI amlgm", "scan · train · explain · serve"],
    [fa.FaPlug, SKY, "API REST", "FastAPI · OpenAPI 3.1"],
    [fa.FaDocker, "2496ED", "Docker", "démo en une commande"],
  ];
  for (let i = 0; i < 4; i++) {
    const x = 0.7 + i * 3.1;
    card(s, x, 1.45, 2.8, 2.5, `a${i + 1}_c`);
    await iconDisc(s, ifs[i][0], x + 0.95, 1.65, 0.9, ifs[i][1], WHITE, `a${i + 1}_i`);
    text(s, ifs[i][2], { x: x + 0.1, y: 2.7, w: 2.6, h: 0.45, fontSize: 18, bold: true, color: INK, align: "center", objectName: `a${i + 1}_t` });
    text(s, ifs[i][3], { x: x + 0.1, y: 3.15, w: 2.6, h: 0.6, fontSize: 14, color: MUTED, align: "center", objectName: `a${i + 1}_d` });
  }
  const q = [[String(D.tests), "tests automatiques", TEAL], [`${D.coverage} %`, "du code couvert", GOLD], ["CI", "Linux · Windows · macOS", CORAL]];
  for (let i = 0; i < 3; i++) {
    const x = 0.7 + i * 4.1;
    text(s, q[i][0], { x, y: 4.5, w: 3.7, h: 1.1, fontFace: HEAD, fontSize: 54, bold: true, color: q[i][2], align: "center", objectName: `a5_q${i}` });
    text(s, q[i][1], { x, y: 5.6, w: 3.7, h: 0.5, fontSize: 16, color: MUTED, align: "center", objectName: `a5_l${i}` });
  }
  s.addNotes("Les mêmes fonctions sont accessibles par une interface web d'investigation, la ligne de commande amlgm, une API REST documentée en OpenAPI 3.1 et une image Docker. La qualité est contrôlée par 475 tests automatiques, 98 % du code couvert, et une intégration continue sur Linux, Windows et macOS.");

  // 14 ------------------------------------------------------------------- deliverables
  s = pres.addSlide(); light(s); title(s, "Bilan par rapport au cahier des charges"); number(s);
  const items = [
    ["J1", "Pipeline + EDA", "ok"], ["J2", "Motifs > 0,5", "ok"], ["J3", "3 GNN + benchmark", "ok"], ["J4", "Article soumis", "todo"],
    ["L1", "Dépôt GitHub", "ok"], ["L2", "Package PyPI", "todo"], ["L3", "API déployée", "ok"], ["L4", "CLI", "ok"],
    ["L5", "Docs + notebooks", "part"], ["L6", "Modèles sur le Hub", "ok"], ["L7", "Article", "part"], ["L8", "Mémoire", "part"],
  ];
  const st = { ok: [TEAL, fa.FaCheck], part: [GOLD, fa.FaHourglassHalf], todo: ["9AA7BD", fa.FaRegClock] };
  for (let i = 0; i < items.length; i++) {
    const x = 0.7 + (i % 4) * 3.1, y = 1.5 + Math.floor(i / 4) * 1.6;
    card(s, x, y, 2.85, 1.3, `a${Math.floor(i / 4) + 1}_c${i}`);
    await iconDisc(s, st[items[i][2]][1], x + 0.2, y + 0.3, 0.7, st[items[i][2]][0], WHITE, `a${Math.floor(i / 4) + 1}_s${i}`);
    text(s, items[i][0], { x: x + 1.05, y: y + 0.18, w: 1.7, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: INK, objectName: `a${Math.floor(i / 4) + 1}_k${i}` });
    text(s, items[i][1], { x: x + 1.05, y: y + 0.63, w: 1.75, h: 0.5, fontSize: 14, color: MUTED, objectName: `a${Math.floor(i / 4) + 1}_l${i}` });
  }
  const legend = [[TEAL, "fait"], [GOLD, "prêt, à publier"], ["9AA7BD", "à venir"]];
  for (let i = 0; i < 3; i++) {
    s.addShape("ellipse", { x: 0.7 + i * 2.6, y: 6.45, w: 0.28, h: 0.28, fill: { color: legend[i][0] }, line: { color: legend[i][0] }, objectName: `a4_lg${i}` });
    text(s, legend[i][1], { x: 1.08 + i * 2.6, y: 6.38, w: 2.1, h: 0.42, fontSize: 14, color: MUTED, objectName: `a4_lt${i}` });
  }
  s.addNotes("Bilan : les jalons J1 à J3 sont atteints, J2 de justesse. Les livrables techniques sont prêts ; restent les étapes de publication qui demandent les comptes personnels : dépôt public et DOI, PyPI, documentation en ligne, modèles sur Hugging Face, déploiement de l'API, soumission de l'article et rédaction du mémoire.");

  // 15 ------------------------------------------------------------------- limits & next
  s = pres.addSlide(); light(s); title(s, "Limites et perspectives"); number(s);
  const lim = [
    [fa.FaExclamationTriangle, CORAL, "Données simulées", "les scores mesurent la capacité à retrouver les motifs du générateur"],
    [fa.FaFilter, CORAL, "Règles propres à IBM", "ACH seul, pas de multi-devises : à revoir sur données réelles"],
    [fa.FaPuzzlePiece, CORAL, `${D.no_motif_share} % hors motif`, "aucune règle ne peut voir ce blanchiment"],
  ];
  const nxt = [
    [fa.FaClock, TEAL, "GNN temporels et hétérogènes", "suivre l'évolution des comptes"],
    [fa.FaUsers, TEAL, "Apprentissage fédéré", "plusieurs banques sans partager leurs données"],
    [fa.FaHandshake, TEAL, "Données réelles", "validation avec une institution partenaire"],
  ];
  for (const [k, list] of [lim, nxt].entries()) {
    text(s, k === 0 ? "Limites" : "Perspectives", { x: 0.7 + k * 6.2, y: 1.4, w: 5.6, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: k === 0 ? CORAL : TEAL, objectName: `a${k * 4 + 1}_h` });
    for (let i = 0; i < 3; i++) {
      const x = 0.7 + k * 6.2, y = 2.1 + i * 1.5;
      await iconDisc(s, list[i][0], x, y, 0.85, list[i][1], WHITE, `a${k * 4 + i + 2}_i`);
      text(s, list[i][2], { x: x + 1.05, y, w: 4.7, h: 0.45, fontSize: 18, bold: true, color: INK, objectName: `a${k * 4 + i + 2}_t` });
      text(s, list[i][3], { x: x + 1.05, y: y + 0.45, w: 4.7, h: 0.6, fontSize: 15, color: MUTED, objectName: `a${k * 4 + i + 2}_d` });
    }
  }
  s.addNotes("Limites : les données IBM sont simulées ; certains réglages, comme la restriction aux virements ACH, sont propres au simulateur ; et 36 % du blanchiment n'appartient à aucun motif. Perspectives : des GNN temporels et hétérogènes, l'apprentissage fédéré entre banques, et surtout une validation sur des données réelles.");

  // 16 ------------------------------------------------------------------- closing
  s = pres.addSlide(); dark(s); number(s, "C9D3E6");
  motif(s, { x: 9.3, y: 0.9, w: 3.3, h: 2.2 }, [[0.08, 0.5, CORAL], [0.5, 0.08, CORAL], [0.92, 0.5, CORAL], [0.5, 0.92, GOLD]], [[0, 1], [1, 2], [2, 3], [3, 0]], "a1_loop");
  text(s, "Merci", { x: 0.7, y: 2.0, w: 8, h: 1.3, fontFace: HEAD, fontSize: 66, bold: true, color: WHITE, objectName: "a2_t" });
  text(s, "Questions ?", { x: 0.7, y: 3.3, w: 8, h: 0.9, fontSize: 32, color: "7FE3D8", objectName: "a3_q" });
  card(s, 0.7, 4.9, 8.2, 1.0, "a4_c", NAVY2);
  text(s, "amlgm serve --demo --open-browser", { x: 0.95, y: 4.9, w: 7.8, h: 1.0, fontFace: "Courier New", fontSize: 22, color: GOLD, valign: "middle", objectName: "a4_cmd" });
  s.addNotes("Merci. Je peux faire une démonstration en direct avec la commande affichée : elle ouvre l'interface d'investigation sur des données synthétiques, sans aucune préparation.");

  const out = path.join(__dirname, "AML-GraphMiner_soutenance.pptx");
  await pres.writeFile({ fileName: out });
  console.log("slides written to", out);
}

main().catch((e) => { console.error(e); process.exit(1); });
