/* AML-GraphMiner — French texts of the web investigation interface.
 * The API answers in English with codes (motif, method, evidence key, attribute name, payment
 * format ...) and figures; this file turns them into French labels and sentences. Pure functions,
 * no DOM: app.js uses them in the browser, tests/test_web_texts.py runs them with Node on real
 * answers of the API. */
"use strict";

((root, factory) => {
  const texts = factory();
  if (typeof module === "object" && module.exports) module.exports = texts;
  else root.AML_FR = texts;
})(globalThis, () => {
  // ------------------------------------------------------------------ numbers, dates, durations
  const nf = new Intl.NumberFormat("fr-FR");
  const fixedFormats = new Map();
  const fixed = (digits) => {
    if (!fixedFormats.has(digits)) {
      fixedFormats.set(digits, new Intl.NumberFormat("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits }));
    }
    return fixedFormats.get(digits);
  };
  const NBSP = String.fromCharCode(0xa0); // no-break space: "24 h" or "12,5 %" stays on one line
  const missing = (v) => v === null || v === undefined || Number.isNaN(v);
  const upperFirst = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : text);
  const lowerFirst = (text) => (text ? text.charAt(0).toLowerCase() + text.slice(1) : text);
  const plural = (n, one, many) => `${nf.format(n)} ${Math.abs(n) < 2 ? one : many}`;

  function duration(seconds) {
    if (missing(seconds)) return "–";
    const s = Math.round(Number(seconds));
    if (s < 60) return `${s}${NBSP}s`;
    const minutes = Math.round(s / 60);
    if (minutes < 60) return `${minutes}${NBSP}min`;
    const h = Math.floor(minutes / 60);
    const m = minutes % 60;
    if (h < 24) return m ? `${h}${NBSP}h${NBSP}${String(m).padStart(2, "0")}${NBSP}min` : `${h}${NBSP}h`;
    const d = Math.floor(h / 24);
    return h % 24 ? `${d}${NBSP}j${NBSP}${h % 24}${NBSP}h` : `${d}${NBSP}j`;
  }

  const fmt = {
    int: (v) => (missing(v) ? "–" : nf.format(v)),
    num: (v, digits = 2) => (missing(v) ? "–" : fixed(digits).format(v)),
    money: (v) => (missing(v) ? "–" : fixed(2).format(v)),
    amount: (v, currency) => (missing(v) ? "–" : `${fixed(2).format(v)}${currency ? `${NBSP}${currencyCode(currency)}` : ""}`),
    pct: (v, digits = 2) => (missing(v) ? "–" : `${fixed(digits).format(100 * v)}${NBSP}%`),
    // tiny thresholds (gradient boosting outputs ~1e-4 for legitimate payments) stay readable
    score: (v) => {
      if (missing(v)) return "–";
      const x = Number(v);
      return x !== 0 && Math.abs(x) < 0.001 ? x.toExponential(2).replace(".", ",") : fixed(3).format(x);
    },
    signed: (v, digits = 3) => (missing(v) ? "–" : `${Number(v) < 0 ? "−" : "+"}${fixed(digits).format(Math.abs(v))}`),
    date: (v) => (v ? new Date(v).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" }) : "–"),
    day: (v) => (v ? new Date(v).toLocaleDateString("fr-FR") : "–"),
    duration,
    pattern: (p) => patternLabel(p),
    short: (account) => {
      const text = String(account);
      return text.length > 16 ? `…${text.slice(-14)}` : text;
    },
  };

  // ------------------------------------------------------------------ motifs
  // label and meaning of each motif detector (PATTERN_DESCRIPTIONS on the server)
  const PATTERNS = {
    fan_in: ["Fan-in", "de nombreux comptes envoient de l'argent à un même compte en peu de temps (collecte)"],
    fan_out: ["Fan-out", "un compte envoie de l'argent à de nombreux comptes en peu de temps (dispersion)"],
    smurfing: [
      "Smurfing",
      "de nombreux paiements juste sous un seuil de déclaration (structuration), ou un montant reçu " +
        "redistribué en petites parts (fractionnement)",
    ],
    cycle: ["Cycle", "l'argent revient à son point de départ par une chaîne de comptes (opération circulaire)"],
    scatter_gather: ["Scatter-gather", "l'argent est réparti entre des intermédiaires puis regroupé sur un compte final"],
    gather_scatter: ["Gather-scatter", "l'argent est collecté par un compte pivot puis redistribué vers d'autres comptes"],
    community: ["Communauté à risque", "un petit groupe de comptes de transit fait circuler l'argent surtout entre ses membres"],
    relay: ["Relais", "un compte transfère presque aussitôt la quasi-totalité de ce qu'il vient de recevoir (compte de transit)"],
  };
  // motif types of the ground truth (IBM AML patterns files) that no detector reports
  const TRUE_PATTERNS = { stack: "Stack (empilement)", bipartite: "Biparti", random: "Aléatoire" };
  const patternLabel = (p) => (PATTERNS[p] ? PATTERNS[p][0] : TRUE_PATTERNS[p] || p || "–");
  const patternDescription = (p, fallback = p) => (PATTERNS[p] ? PATTERNS[p][1] : fallback);

  // ------------------------------------------------------------------ detector and methods
  const METHODS = {
    auto: "automatique",
    patterns: "Preuves des motifs (exact)",
    occlusion: "Occlusion (tout modèle)",
    gnnexplainer: "GNNExplainer",
    pgexplainer: "PGExplainer",
    captum: "Captum (gradients intégrés)",
    attention: "Attention (GAT)",
  };
  const MODELS = {
    patterns: "Motifs structurels",
    tabular: "Modèle tabulaire",
    gnn: "GNN supervisé",
    gnn_unsupervised: "GNN non supervisé (auto-encodeur)",
    hybrid: "Hybride (modèle + motifs)",
  };
  const KINDS = {
    logreg: "régression logistique",
    rf: "forêt aléatoire",
    hgb: "gradient boosting (HGB)",
    xgb: "XGBoost",
    gcn: "GCN",
    sage: "GraphSAGE",
    gat: "GAT",
  };
  const METRICS = {
    pr_auc: "PR-AUC",
    roc_auc: "ROC-AUC",
    precision: "Précision",
    recall: "Rappel",
    f1: "F1",
    recall_at_k: "Rappel@k (k = nb de positifs)",
    recall_at_1pct: "Rappel sur le top 1 %",
  };
  // fields of the detector description (/api/model → model)
  const MODEL_FIELDS = {
    kind: "Algorithme",
    threshold: "Seuil de décision",
    n_features: "Attributs en entrée",
    detectors: "Détecteurs de motifs",
    motif_detectors: "Détecteurs de motifs",
    calibrated: "Scores calibrés",
    parameters: "Paramètres appris",
    best_epoch: "Meilleure époque",
    inference: "Inférence",
    alpha: "Poids de l'erreur sur les attributs (alpha)",
    base: "Modèle de base",
    stacker: "Poids de la combinaison (stacking)",
  };
  const INFERENCE = { sampled: "voisinage échantillonné", full: "graphe complet" };
  const STACKER_INPUTS = { model_logit: "logit du modèle", motif_score: "score des motifs", motif_fired: "motif déclenché" };

  const methodLabel = (m) => METHODS[m] || m;
  const modelName = (t) => MODELS[t] || t;
  const kindName = (k) => KINDS[k] || k;
  const metricLabel = (k) => METRICS[k] || k;

  function plainValue(value) {
    if (typeof value === "boolean") return value ? "oui" : "non";
    if (typeof value === "number") return Number.isInteger(value) ? fmt.int(value) : fmt.num(value, 4);
    if (Array.isArray(value)) return value.map(plainValue).join(", ");
    if (value && typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  /** [French label, French value] of one field of the detector description. */
  function modelField(key, value) {
    const label = MODEL_FIELDS[key] || key;
    if (key === "kind") return [label, kindName(value)];
    if (key === "threshold") return [label, fmt.score(value)];
    if (key === "detectors" || key === "motif_detectors") return [label, (value || []).map(patternLabel).join(", ")];
    if (key === "inference") return [label, INFERENCE[value] || String(value)];
    if (key === "base" && value && typeof value === "object") {
      return [label, [modelName(value.model_type), value.kind ? kindName(value.kind) : null].filter(Boolean).join(" · ")];
    }
    if (key === "stacker" && value && typeof value === "object") {
      return [label, Object.entries(value).map(([k, v]) => `${STACKER_INPUTS[k] || k} ${fmt.num(v, 2)}`).join(" ; ")];
    }
    return [label, plainValue(value)];
  }

  // ------------------------------------------------------------------ data values
  const PAYMENT_FORMATS = {
    ACH: "ACH",
    Bitcoin: "Bitcoin",
    Cash: "Espèces",
    Cheque: "Chèque",
    "Credit Card": "Carte de crédit",
    Reinvestment: "Réinvestissement",
    Wire: "Virement",
  };
  // currencies of the IBM AML files: ISO code next to amounts, French name elsewhere
  const CURRENCIES = {
    "US Dollar": ["USD", "dollar américain"],
    Euro: ["EUR", "euro"],
    Yuan: ["CNY", "yuan"],
    Bitcoin: ["BTC", "bitcoin"],
    "Swiss Franc": ["CHF", "franc suisse"],
    Ruble: ["RUB", "rouble"],
    Yen: ["JPY", "yen"],
    "UK Pound": ["GBP", "livre sterling"],
    Rupee: ["INR", "roupie indienne"],
    "Canadian Dollar": ["CAD", "dollar canadien"],
    "Australian Dollar": ["AUD", "dollar australien"],
    "Mexican Peso": ["MXN", "peso mexicain"],
    "Brazil Real": ["BRL", "réal brésilien"],
    "Saudi Riyal": ["SAR", "riyal saoudien"],
    Shekel: ["ILS", "shekel"],
  };
  const paymentFormat = (v) => PAYMENT_FORMATS[v] || v;
  const currencyCode = (v) => (CURRENCIES[v] ? CURRENCIES[v][0] : v);
  const currencyName = (v) => (CURRENCIES[v] ? CURRENCIES[v][1] : v);

  // roles of the accounts of an explanation (and of the hub of a smurfing alert)
  const ROLES = { sender: "émetteur", receiver: "destinataire", neighbour: "voisin" };
  const roleLabel = (r) => ROLES[r] || r;

  // ------------------------------------------------------------------ evidence of the alerts
  // label and kind of value of every evidence key of the motif detectors
  const EVIDENCE = {
    hub: ["Compte pivot", "account"],
    n_counterparties: ["Contreparties distinctes", "int"],
    n_transactions: ["Transactions", "int"],
    span_seconds: ["Durée", "duration"],
    window_seconds: ["Fenêtre de détection", "duration"],
    amount_cv: ["Variabilité des montants (CV)", "number"],
    burst_ratio: ["Pic d'activité", "times"],
    mode: ["Variante", "mode"],
    role: ["Rôle du compte", "role"],
    account: ["Compte", "account"],
    threshold: ["Seuil de déclaration", "money"],
    mean_amount: ["Montant moyen", "money"],
    inflow_amount: ["Montant reçu", "money"],
    n_parts: ["Nombre de parts", "int"],
    parts_total: ["Total des parts", "money"],
    largest_part_fraction: ["Plus grosse part (en % du montant reçu)", "pct"],
    length: ["Transactions dans la boucle", "int"],
    first_amount: ["Montant de départ", "money"],
    last_amount: ["Montant revenu à l'origine", "money"],
    amount_retained: ["Part du montant revenue à l'origine", "pct"],
    cycle: ["Parcours", "path"],
    source: ["Compte d'origine", "account"],
    sink: ["Compte final", "account"],
    n_intermediaries: ["Intermédiaires", "int"],
    dispersed_amount: ["Montant dispersé", "money"],
    n_sources: ["Comptes émetteurs", "int"],
    n_targets: ["Comptes destinataires", "int"],
    gathered_amount: ["Montant collecté", "money"],
    scattered_amount: ["Montant redistribué", "money"],
    conservation_ratio: ["Redistribué / collecté", "pct"],
    n_members: ["Membres du groupe", "int"],
    n_relays: ["Comptes de transit", "int"],
    n_internal_tx: ["Transactions internes", "int"],
    internal_share: ["Part du volume restée dans le groupe", "pct"],
    mean_flow_through: ["Taux de transit moyen", "pct"],
    internal_amount: ["Montant échangé dans le groupe", "money"],
    has_cycle: ["Contient un cycle", "bool"],
    relay: ["Compte relais", "account"],
    forwarded_share: ["Part transférée", "pct"],
    delay_seconds: ["Délai avant transfert", "duration"],
    currency_in: ["Devise reçue", "currency"],
    currency_out: ["Devise transférée", "currency"],
  };
  const MODES = { structuring: "structuration (montants juste sous le seuil)", split: "fractionnement d'un montant reçu" };

  const evidenceLabel = (key) => (EVIDENCE[key] ? EVIDENCE[key][0] : key);

  function evidenceValue(key, value) {
    if (value === null || value === undefined) return "–";
    if (Array.isArray(value)) return value.join(" → ");
    if (typeof value === "boolean") return value ? "oui" : "non";
    switch (EVIDENCE[key] && EVIDENCE[key][1]) {
      case "duration":
        return fmt.duration(value);
      case "money":
        return fmt.money(value);
      case "pct":
        return fmt.pct(value, 1);
      case "times":
        return `×${NBSP}${fmt.num(value, 1)}`;
      case "int":
        return fmt.int(value);
      case "mode":
        return MODES[value] || String(value);
      case "role":
        return roleLabel(value);
      case "currency":
        return currencyName(value);
      default:
        break;
    }
    if (typeof value === "number") return Number.isInteger(value) ? fmt.int(value) : fmt.num(value, 2);
    return String(value);
  }

  /** [label, value] of the evidence an analyst reads: scalars and lists (cycle path). */
  function evidenceItems(evidence) {
    return Object.entries(evidence || {})
      .filter(([, v]) => v === null || typeof v !== "object" || Array.isArray(v))
      .map(([key, value]) => [evidenceLabel(key), evidenceValue(key, value)]);
  }

  /** One sentence describing an alert (describe_alert on the server); ``label: false`` leaves
   * out the motif name when the page already shows it next to the sentence. */
  function alertSentence(alert, { label = true, maxItems = 6 } = {}) {
    const facts = Object.entries(alert.evidence || {})
      .filter(([, v]) => v !== null && v !== undefined && typeof v !== "object")
      .slice(0, maxItems)
      .map(([key, value]) => `${lowerFirst(evidenceLabel(key))} : ${evidenceValue(key, value)}`);
    const detail = facts.length ? ` — ${facts.join(", ")}` : "";
    const name = label ? `${patternLabel(alert.pattern)} : ` : "";
    return `${name}${patternDescription(alert.pattern)}${detail}.`;
  }

  // ------------------------------------------------------------------ attributes (features)
  const RAW_ATTRIBUTES = {
    amount: "montant payé",
    amount_received: "montant reçu",
    transaction_type: "format de paiement",
    currency: "devise payée",
    currency_received: "devise reçue",
    sender_bank: "banque de l'émetteur",
    receiver_bank: "banque du destinataire",
  };
  const ACCOUNT_FEATURES = {
    out_rate: "envois par jour",
    in_rate: "réceptions par jour",
    out_amt_mean: "montant moyen envoyé",
    in_amt_mean: "montant moyen reçu",
    out_vol_rate: "volume envoyé par jour",
    in_vol_rate: "volume reçu par jour",
    uniq_out_rate: "destinataires distincts par jour",
    uniq_in_rate: "émetteurs distincts par jour",
    io_ratio: "rapport reçu / envoyé",
    ent_out: "diversité des destinataires (entropie)",
    ent_in: "diversité des émetteurs (entropie)",
    velocity: "transactions par jour d'activité",
    pagerank: "PageRank",
    betweenness: "centralité d'intermédiarité",
  };
  const TX_FEATURES = {
    log_paid: "montant payé",
    log_recv: "montant reçu",
    same_cur: "même devise",
    same_bank: "même banque",
    h_sin: "heure de la journée (sinus)",
    h_cos: "heure de la journée (cosinus)",
  };
  // trailing-window activity of the sender (s_) and of the receiver (r_) of the transaction
  const WINDOW_STATS = {
    s: {
      cnt: "envois de l'émetteur",
      sum: "montant envoyé par l'émetteur",
      mean: "montant moyen envoyé par l'émetteur",
      std: "dispersion des montants envoyés par l'émetteur",
      gap: "délai depuis l'envoi précédent de l'émetteur",
    },
    r: {
      cnt: "réceptions du destinataire",
      sum: "montant reçu par le destinataire",
      mean: "montant moyen reçu par le destinataire",
      std: "dispersion des montants reçus par le destinataire",
      gap: "délai depuis la réception précédente du destinataire",
    },
  };

  function windowName(window) {
    if (window === "all") return "tout l'historique";
    const m = /^(\d+)(s|min|m|h|d|w)$/.exec(window);
    if (!m) return window;
    const n = Number(m[1]);
    const unit = { s: "s", min: "min", m: "min", h: "h", d: n > 1 ? "jours" : "jour", w: n > 1 ? "semaines" : "semaine" }[m[2]];
    return `${n}${NBSP}${unit}`;
  }

  function columnLabel(column) {
    if (RAW_ATTRIBUTES[column]) return RAW_ATTRIBUTES[column];
    if (TX_FEATURES[column]) return TX_FEATURES[column];
    if (column.startsWith("fmt_")) return `format ${paymentFormat(column.slice(4))}`;
    let m = /^([sr])_(cnt|sum|mean|std)_(.+)$/.exec(column);
    if (m) return `${WINDOW_STATS[m[1]][m[2]]} sur ${windowName(m[3])}`;
    m = /^([sr])_gap$/.exec(column);
    return m ? WINDOW_STATS[m[1]].gap : null;
  }

  /** French name of an attribute: "amount", "tx.s_cnt_24h", "sender.out_rate" ... */
  function featureLabel(name) {
    const text = String(name);
    const dot = text.indexOf(".");
    const scope = dot > 0 ? text.slice(0, dot) : "";
    const column = dot > 0 ? text.slice(dot + 1) : text;
    if (scope === "sender" || scope === "receiver") {
      return ACCOUNT_FEATURES[column] ? `${ROLES[scope]} · ${ACCOUNT_FEATURES[column]}` : text;
    }
    return columnLabel(column) || text;
  }

  function featureValue(feature) {
    const name = String(feature.name);
    const column = name.slice(name.indexOf(".") + 1);
    const value = feature.value;
    if (typeof value === "number") {
      return column === "amount" || column === "amount_received" ? fmt.money(value) : fmt.num(value, 2);
    }
    if (column === "transaction_type") return paymentFormat(value);
    if (column === "currency" || column === "currency_received") return currencyName(value);
    return String(value);
  }

  // ------------------------------------------------------------------ explanations
  function topNeighbour(explanation) {
    let best = null;
    for (const edge of explanation.edges || []) {
      if (edge.role === "target") continue;
      if (!best || Math.abs(edge.importance) > Math.abs(best.importance)) best = edge;
    }
    return best;
  }

  // "elle augmente le score de 0,213"; an effect that rounds to 0,000 reads "de moins de 0,001"
  const change = (value, subject, what) => {
    const size = Math.abs(value) < 0.0005 ? "moins de 0,001" : fmt.num(Math.abs(value), 3);
    return `${subject} ${value > 0 ? "augmente" : "diminue"} ${what} de ${size}`;
  };

  /** The summary sentence of an explanation (Explanation.summary on the server), in French. */
  function explanationSummary(explanation) {
    const e = explanation;
    const details = e.details || {};
    if (e.method === "patterns") {
      const alerts = e.patterns || [];
      if (!alerts.length) return "Aucun motif structurel ne contient cette transaction.";
      const n = plural(alerts.length, "alerte structurelle", "alertes structurelles");
      return `Fait partie de ${n}. ${alerts.slice(0, 2).map((a) => alertSentence(a)).join(" ")}`;
    }
    const top = topNeighbour(e);
    if (e.method === "occlusion") {
      const base =
        details.neighbours === undefined
          ? `Score ${fmt.score(e.score)}.`
          : `Score ${fmt.score(e.score)} calculé avec ${plural(details.neighbours, "transaction voisine", "transactions voisines")}.`;
      if (!top || Math.abs(top.importance) < 1e-9) {
        return `${base} Retirer n'importe laquelle des transactions voisines ne change pas le score : il tient à la transaction elle-même.`;
      }
      return (
        `${base} La transaction voisine la plus influente est la n° ${top.tx_id} ` +
        `(${top.sender} → ${top.receiver}, ${fmt.money(top.amount)}) : ` +
        `${change(top.importance, "elle", "le score")}.`
      );
    }
    // GNN methods (gnnexplainer, pgexplainer, captum, attention)
    let text;
    if (details.uniform_mask !== undefined) {
      text =
        `Score ${fmt.score(e.score)} ; aucune transaction voisine ne se distingue (tous les masques ≈ ` +
        `${fmt.num(details.uniform_mask, 2)}) : la prédiction tient aux attributs de la transaction et des ` +
        "comptes plutôt qu'au voisinage — voir la méthode Captum.";
    } else {
      text =
        `Score ${fmt.score(e.score)} ; ` +
        (top
          ? `transaction voisine la plus influente : n° ${top.tx_id} (${top.sender} → ${top.receiver}, ` +
            `importance ${fmt.num(top.importance, 3)}).`
          : "aucune transaction voisine.");
    }
    const features = e.features || [];
    if (e.method === "gnnexplainer" && features.length) {
      text += ` Attributs de compte retenus par le masque : ${features.slice(0, 3).map((f) => featureLabel(f.name)).join(", ")}.`;
    }
    if (e.method === "captum" && features.length) {
      const f = features[0];
      text +=
        ` Attribut le plus influent : ${featureLabel(f.name)} (valeur standardisée ${fmt.num(Number(f.value), 2)}) ; ` +
        `${change(f.importance, "il", "la probabilité")}.`;
    }
    if (details.trained_on !== undefined) {
      text += ` Explicateur entraîné sur ${plural(details.trained_on, "voisinage d'alerte", "voisinages d'alertes")}.`;
    }
    if (details.neighbourhood_effect !== undefined) {
      text +=
        ` Sans ses ${plural(details.neighbours, "transaction voisine", "transactions voisines")}, le score serait de ` +
        `${fmt.score(details.score_without_neighbours)} (effet mesuré du voisinage : ${fmt.signed(details.neighbourhood_effect)}).`;
    }
    return text;
  }

  // ------------------------------------------------------------------ datasets and errors
  function datasetName(dataset) {
    const origin = dataset.origin || {};
    return origin.type === "demo" ? `Démo synthétique (graine ${origin.seed})` : dataset.name;
  }

  const count = (text) => nf.format(Number(String(text).replace(/,/g, "")));
  // messages of the API (English) an analyst can meet → French; anything else is shown as sent
  const ERRORS = [
    [/^unknown account '(.*)'$/, (m) => `Compte « ${m[1]} » absent de ce jeu de données.`],
    [/^unknown transaction (.+)$/, (m) => `Transaction n° ${m[1]} absente de ce jeu de données.`],
    [/^unknown alert (.+)$/, (m) => `Alerte n° ${m[1]} introuvable.`],
    [/^unknown dataset /, () => "Ce jeu de données n'est plus chargé sur le serveur (fermé ou serveur redémarré) : rechargez la page."],
    [/^the table contains no transaction$/, () => "Le fichier ne contient aucune transaction valide."],
    [
      /^([\d,]+) transactions exceed the limit of the interface \(([\d,]+)\)/,
      (m) => `${count(m[1])} transactions dépassent la limite de l'interface (${count(m[2])}) : analysez ce fichier avec « amlgm score ».`,
    ],
    [/^unsupported file type '(.*)'/, (m) => `Type de fichier non pris en charge (« ${m[1]} ») : CSV, JSON, JSON-lines ou Parquet.`],
    [/^file larger than (\d+) MB$/, (m) => `Fichier trop volumineux (limite : ${m[1]} Mo).`],
    [
      /^could not detect the input format from columns (\[.*\])/,
      (m) =>
        `Format de fichier non reconnu (colonnes ${m[1]}) : choisissez le format « IBM AML » ou « canonique », ` +
        "ou nommez les colonnes timestamp, sender, receiver, amount…",
    ],
    [
      /^missing required column\(s\): (\[[^\]]*\])\. Found columns: (\[.*\])$/,
      (m) => `Colonne(s) obligatoire(s) absente(s) du fichier : ${m[1]} (colonnes lues : ${m[2]}).`,
    ],
    [/^ibm: expected 10 or 11 columns \(IBM AML format\), got (\d+)$/, (m) => `Le format IBM AML a 10 ou 11 colonnes ; ce fichier en a ${m[1]}.`],
    [/^label must be 0, 1 or empty$/, () => "La colonne label ne peut contenir que 0, 1 ou une case vide."],
  ];
  const NETWORK_ERROR =
    "Serveur injoignable : vérifiez la connexion (un service gratuit en veille met environ une minute à redémarrer).";

  /** French message for an HTTP error of the API (``body``: parsed JSON or text). */
  function errorMessage(status, body) {
    const detail = body && typeof body === "object" ? (body.detail ?? body.error) : body;
    if (Array.isArray(detail)) {
      // request validation (FastAPI): the parameters at fault
      const fields = detail.map((d) => (d.loc || []).slice(1).join(".")).filter(Boolean);
      return `Paramètre invalide${fields.length ? ` : ${fields.join(", ")}` : ""}.`;
    }
    const text = typeof detail === "string" ? detail.trim() : "";
    for (const [pattern, render] of ERRORS) {
      const m = pattern.exec(text);
      if (m) return render(m);
    }
    if (status === 413) return "Fichier trop volumineux pour le serveur.";
    if (status >= 500) {
      return text && !text.startsWith("<") && text !== "Internal Server Error"
        ? `Erreur interne du serveur : ${text}`
        : `Le serveur n'a pas pu répondre (HTTP ${status}) : réessayez dans un instant.`;
    }
    return text || `Erreur HTTP ${status}.`;
  }

  return {
    fmt,
    upperFirst,
    METHODS,
    NETWORK_ERROR,
    patternLabel,
    patternDescription,
    methodLabel,
    modelName,
    kindName,
    metricLabel,
    modelField,
    paymentFormat,
    currencyCode,
    currencyName,
    roleLabel,
    evidenceLabel,
    evidenceValue,
    evidenceItems,
    alertSentence,
    featureLabel,
    featureValue,
    explanationSummary,
    datasetName,
    errorMessage,
    // every code translated here (tests check that the server sends no other)
    codes: {
      patterns: Object.keys(PATTERNS),
      methods: Object.keys(METHODS),
      models: Object.keys(MODELS),
      kinds: Object.keys(KINDS),
      metrics: Object.keys(METRICS),
      evidence: Object.keys(EVIDENCE),
      modelFields: Object.keys(MODEL_FIELDS),
      paymentFormats: Object.keys(PAYMENT_FORMATS),
      currencies: Object.keys(CURRENCIES),
    },
  };
});
