// End-to-end check of the web investigation interface in a real browser (headless Edge/Chrome).
//
//   cd tests/e2e && npm install
//   node ui_check.js <base url> <screenshot dir> <transactions.csv to upload>
//   BROWSER=/path/to/chrome node ui_check.js ...        (default: Edge on Windows, else Chrome)
//
// Drives every tab and action of /ui/ (demo dataset, dashboard, alerts and their sub-graph, each
// explanation method, sorting, account profile, CSV downloads, upload, close) and fails on any
// console error, page error, failed request or English text left in the page (the API answers
// in English, fr.js words it in French). Screenshots of every step go to <screenshot dir>.
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer-core");

const [BASE, OUT, UPLOAD] = process.argv.slice(2);
const EDGE =
  process.env.BROWSER ||
  (process.platform === "win32"
    ? "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    : "/usr/bin/google-chrome");
fs.mkdirSync(OUT, { recursive: true });
const results = [];
const problems = [];
const exportsSeen = new Set();
let initialIds = []; // datasets that existed before the check: never closed by it
const closedByCheck = new Set();

async function check(name, fn) {
  const t0 = Date.now();
  try {
    const detail = (await fn()) || "";
    results.push({ name, ok: true, seconds: (Date.now() - t0) / 1000, detail });
    console.log(`[PASS] ${name} (${((Date.now() - t0) / 1000).toFixed(1)}s) ${detail}`);
  } catch (e) {
    results.push({ name, ok: false, seconds: (Date.now() - t0) / 1000, detail: String(e.message || e).slice(0, 300) });
    console.log(`[FAIL] ${name} ${String(e.message || e).slice(0, 300)}`);
  }
}
const ensure = (cond, msg) => { if (!cond) throw new Error(msg); };
// English words of the API (summaries, codes, data values) that must not reach the interface
const ENGLISH = new RegExp(
  String.raw`\b(the|and|with|without|of|to|from|is|would|this|which|money|accounts?|amounts?|sends?|receives?|` +
    String.raw`neighbou?r\w*|raises|lowers|structural|unknown|missing|expected|seed|synthetic|threshold|transfers?|` +
    String.raw`split|hub|most|influential|features?|sender|receiver|relay|got|file|columns?|found|larger|exceed|` +
    String.raw`Cheque|Credit Card|Wire|Reinvestment|Cash|US Dollar|Swiss Franc|Canadian Dollar)\b` +
    String.raw`|\b[a-z]+_[a-z_]+\b`, // a raw snake_case code
  "gi",
);
const englishSeen = new Map(); // English word -> where it was seen

(async () => {
  const browser = await puppeteer.launch({ executablePath: EDGE, headless: true, args: ["--no-first-run"] });
  const downloads = path.resolve(OUT, "downloads");
  fs.rmSync(downloads, { recursive: true, force: true });
  fs.mkdirSync(downloads, { recursive: true });
  const cdp = await browser.target().createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: downloads });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  page.setDefaultTimeout(240000);
  page.on("console", (m) => { if (m.type() === "error") problems.push(`console: ${m.text()}`); });
  page.on("pageerror", (e) => problems.push(`page error: ${e.message}`));
  page.on("requestfailed", (r) => {
    const err = r.failure() && r.failure().errorText;
    // a download ends the request with ERR_ABORTED: checked through the saved file instead;
    // closing a dataset cancels its pending requests on purpose
    const closed = [...closedByCheck].some((id) => r.url().includes(`/datasets/${id}/`));
    if (!((r.url().includes("/export/") || closed) && err === "net::ERR_ABORTED")) problems.push(`request failed: ${r.url()} ${err}`);
  });
  page.on("response", (r) => {
    if (r.url().includes("/export/")) exportsSeen.add(`${r.url().split("/export/")[1]} ${r.status()}`);
    if (r.status() >= 400 && !r.url().endsWith("favicon.ico")) problems.push(`HTTP ${r.status()} ${r.url()}`);
  });
  const shot = (name) => page.screenshot({ path: path.join(OUT, name), fullPage: false });
  const idle = () => page.waitForFunction(() => document.querySelector("#overlay").hidden, { polling: 200 });
  const tab = async (name) => { await page.click(`button[data-tab="${name}"]`); await page.waitForSelector(`#tab-${name}.active`); };
  // visible text, tooltips (except the technical names of attributes) and options of the page
  const scanEnglish = async (where) => {
    const texts = await page.evaluate(() => [
      document.body.innerText,
      ...[...document.querySelectorAll("[title]")].map((e) => e.title).filter((t) => !t.startsWith("nom technique")),
      ...[...document.querySelectorAll("svg title, option")].map((e) => e.textContent),
    ]);
    for (const raw of texts) {
      const text = raw.replace(/[\w.-]+\.(?:csv|parquet|json|txt)\b/gi, " "); // file names are the user's
      for (const m of text.matchAll(ENGLISH)) {
        if (!englishSeen.has(m[0])) englishSeen.set(m[0], `${where}: …${text.slice(Math.max(0, m.index - 50), m.index + 50)}…`);
      }
    }
  };

  await check("page loads, detector badge", async () => {
    await page.goto(`${BASE}/ui/`, { waitUntil: "networkidle0" });
    await page.waitForFunction(() => !/chargement/i.test(document.querySelector("#model-badge").textContent));
    initialIds = await page.$$eval("#dataset-select option", (os) => os.map((o) => o.value));
    await shot("01-accueil.png");
    return (await page.$eval("#model-badge", (e) => e.textContent.trim())).slice(0, 80);
  });

  await check("demo dataset: dashboard (KPIs, alert graph, 5 charts, metrics)", async () => {
    await page.click("#btn-demo");
    await page.waitForFunction(() => document.querySelectorAll("#kpis > *").length > 0);
    await idle();
    await page.waitForFunction(() => ["#chart-daily", "#chart-patterns", "#chart-scores", "#chart-amounts", "#chart-formats"]
      .every((s) => document.querySelector(s).children.length > 0));
    await page.waitForSelector("#overview-graph svg circle");
    await shot("02-tableau-de-bord.png");
    await scanEnglish("dashboard");
    const kpis = await page.$$eval("#kpis > *", (els) => els.length);
    const metrics = await page.$eval("#metrics", (e) => e.textContent.trim().length);
    ensure(metrics > 0, "no metrics");
    const nodes = await page.$$eval("#overview-graph svg circle", (c) => c.length);
    return `${kpis} indicators, graph of the strongest alerts (${nodes} accounts), 5 charts, metrics shown`;
  });

  await check("alerts tab: table, filter, evidence + sub-graph", async () => {
    await tab("alerts");
    await page.waitForSelector("#alerts-table tbody tr");
    const before = await page.$eval("#alert-count", (e) => e.textContent);
    await page.click("#alerts-table tbody tr");
    await page.waitForSelector("#alert-detail svg");
    await shot("03-alertes.png");
    await scanEnglish("alerts");
    const options = await page.$$eval("#alert-pattern option", (os) => os.map((o) => o.value).filter(Boolean));
    await page.select("#alert-pattern", options[0]);
    await page.waitForFunction((b) => document.querySelector("#alert-count").textContent !== b, {}, before);
    const after = await page.$eval("#alert-count", (e) => e.textContent);
    return `${before} -> filter "${options[0]}": ${after}; sub-graph drawn`;
  });

  const methodsDone = [];
  await check("transactions tab: auto explanation with graph", async () => {
    await tab("transactions");
    await page.waitForSelector("#tx-table tbody tr");
    await page.click("#tx-table tbody tr");
    await page.waitForSelector("#tx-detail .summary");
    await page.waitForSelector("#tx-detail svg");
    await shot("04-transaction.png");
    await scanEnglish("transaction");
    return (await page.$eval("#tx-detail .summary", (e) => e.textContent)).slice(0, 110);
  });

  const methods = await page.$$eval('#tx-detail select[aria-label="Méthode d\'explication"] option', (os) => os.map((o) => o.value).filter((v) => v !== "auto"));
  for (const m of methods) {
    await check(`explain in the UI with ${m}`, async () => {
      await page.select('#tx-detail select[aria-label="Méthode d\'explication"]', m);
      await page.$eval("#tx-detail .summary", (e) => e.setAttribute("data-old", "1"));
      await page.evaluate(() => [...document.querySelectorAll("#tx-detail button")].find((b) => b.textContent.includes("Expliquer")).click());
      await page.waitForFunction(() => {
        const s = document.querySelector("#tx-detail .summary");
        return document.querySelector("#tx-detail .error") || (s && !s.hasAttribute("data-old"));
      });
      ensure(!(await page.$("#tx-detail .error")), await page.$eval("#tx-detail .error", (e) => e.textContent).catch(() => "error"));
      await page.waitForSelector("#tx-detail svg");
      const heads = await page.$$eval("#tx-detail h4", (hs) => hs.map((x) => x.textContent));
      await shot(`05-explication-${m}.png`);
      await scanEnglish(`explanation ${m}`);
      methodsDone.push(m);
      return heads.join(" | ");
    });
  }

  await check("transactions: sort by amount + alerts only", async () => {
    await page.select("#tx-sort", "amount");
    await page.select("#tx-order", "asc");
    await page.click("#tx-only-alerts");
    await page.waitForFunction(() => document.querySelectorAll("#tx-table tbody tr").length > 0);
    return await page.$eval("#tx-count", (e) => e.textContent);
  });

  await check("accounts tab: profile with graph", async () => {
    const account = await page.$eval("#tx-table tbody tr td:nth-child(3)", (e) => e.textContent.trim()).catch(() => "");
    await tab("accounts");
    const acc = account || (await page.evaluate(() => "")); // fallback below
    const target = acc || (await page.evaluate(async () => {
      const ds = document.querySelector("#dataset-select").value;
      const r = await fetch(`/api/datasets/${ds}/transactions?limit=1`).then((x) => x.json());
      return r.items[0].sender;
    }));
    await page.click("#account-input", { clickCount: 3 });
    await page.type("#account-input", target);
    await page.keyboard.press("Enter");
    await page.waitForSelector("#account-detail svg");
    await shot("06-compte.png");
    await scanEnglish("account");
    return `account ${target}: ${(await page.$eval("#account-detail", (e) => e.textContent)).slice(0, 80)}`;
  });

  await check("about tab: detector and motif descriptions", async () => {
    await tab("about");
    await page.waitForFunction(() => document.querySelector("#about-model").textContent.trim().length > 0);
    await shot("07-a-propos.png");
    await scanEnglish("about");
    return `${await page.$$eval("#about-patterns li, #about-patterns tr, #about-patterns dt", (x) => x.length)} motif entries`;
  });

  await check("CSV exports (alerts, scores)", async () => {
    await tab("overview");
    await page.click("#btn-export-alerts");
    await page.click("#btn-export-scores");
    await new Promise((r) => setTimeout(r, 3000));
    ensure([...exportsSeen].some((x) => x.startsWith("alerts.csv 200")), `exports seen: ${[...exportsSeen]}`);
    ensure([...exportsSeen].some((x) => x.startsWith("scores.csv 200")), `exports seen: ${[...exportsSeen]}`);
    const files = fs.readdirSync(downloads).filter((f) => f.endsWith(".csv"));
    ensure(files.length >= 2, `downloaded files: ${files}`);
    const lines = files.map((f) => `${f}: ${fs.readFileSync(path.join(downloads, f), "utf8").trim().split(/\r?\n/).length - 1} rows`);
    return lines.join(", ");
  });

  await check("upload a CSV file and analyse it", async () => {
    const before = await page.$$eval("#dataset-select option", (os) => os.length);
    const input = await page.$("#file-input");
    await input.uploadFile(UPLOAD);
    await page.waitForFunction((b) => document.querySelectorAll("#dataset-select option").length > b, {}, before);
    await idle();
    await page.waitForFunction(() => document.querySelectorAll("#kpis > *").length > 0);
    await shot("08-fichier-charge.png");
    await scanEnglish("uploaded file");
    return `${before} -> ${await page.$$eval("#dataset-select option", (os) => os.length)} datasets`;
  });

  await check("close the check's datasets (a double click closes one)", async () => {
    const ids = () => page.$$eval("#dataset-select option", (os) => os.map((o) => o.value));
    const snapshot = () => page.evaluate(() => ({
      datasets: document.querySelectorAll("#dataset-select option").length,
      disabled: document.querySelector("#btn-delete").disabled,
      status: document.querySelector("#status").textContent.slice(0, 80),
    }));
    const mine = (await ids()).filter((id) => !initialIds.includes(id));
    ensure(mine.length === 2, `datasets created by the check: ${mine}`);
    for (const [k, id] of mine.entries()) {
      if ((await page.$eval("#dataset-select", (e) => e.value)) !== id) {
        await page.select("#dataset-select", id);
        // the loading overlay blocks clicks on purpose: wait until the dataset is shown
        await page.waitForFunction((x) => document.querySelector("#dataset-select").value === x
          && document.querySelector("#overlay").hidden, { timeout: 120000, polling: 250 }, id);
      }
      const before = (await ids()).length;
      closedByCheck.add(id);
      if (k === 0) {
        // two clicks in the same instant: the second one must be ignored
        const disabledAtOnce = await page.evaluate(() => {
          const b = document.querySelector("#btn-delete");
          b.click();
          const d = b.disabled;
          b.click();
          return d;
        });
        ensure(disabledAtOnce, "close button still enabled while closing");
      } else {
        await page.click("#btn-delete");
      }
      try {
        await page.waitForFunction((n, x) => {
          const left = [...document.querySelectorAll("#dataset-select option")].map((o) => o.value);
          return left.length === n - 1 && !left.includes(x)
            && (left.length === 0 || !document.querySelector("#btn-delete").disabled);
        }, { timeout: 120000, polling: 250 }, before, id);
      } catch (e) {
        throw new Error(`after closing ${id}: ${JSON.stringify(await snapshot())}`);
      }
    }
    const left = await ids();
    ensure(initialIds.every((id) => left.includes(id)), "a pre-existing dataset was closed");
    const status = await page.$eval("#status", (e) => e.textContent);
    ensure(!/erreur/i.test(status), `status shows an error: ${status}`);
    await shot("09-ferme.png");
    return `2 datasets of the check closed (double click closed one), ${left.length} pre-existing kept`;
  });

  await check("the interface is in French (no English text from the API)", async () => {
    const left = [...englishSeen.values()];
    ensure(left.length === 0, left.slice(0, 5).join(" || "));
    return "dashboard, alerts, transactions, every explanation, account, about: no English word";
  });

  await check("no browser error during the session", async () => {
    const relevant = problems.filter((p) => !p.includes("favicon"));
    ensure(relevant.length === 0, relevant.slice(0, 5).join(" || "));
    return "no console error, page error or failed request";
  });

  await browser.close();
  const passed = results.filter((r) => r.ok).length;
  console.log(`\n${passed}/${results.length} UI checks passed against ${BASE}`);
  fs.writeFileSync(path.join(OUT, "ui_results.json"), JSON.stringify({ base: BASE, passed, total: results.length, results, problems, methods: methodsDone }, null, 1));
  process.exit(passed === results.length ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
