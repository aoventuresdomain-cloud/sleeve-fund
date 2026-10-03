// Small helpers for the console: sortable tables and the equity/drawdown chart pair.
window.Console = (() => {
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const money = new Intl.NumberFormat("en-GB", {maximumFractionDigits: 0});
  const day = (t) => new Date(t).toLocaleDateString("en-GB", {day: "2-digit", month: "short", year: "2-digit", timeZone: "UTC"});
  const minute = (t) => new Date(t).toLocaleString("en-GB", {day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC"});

  function dialogs() {
    document.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => {
      const d = document.getElementById(b.dataset.open);
      if (d && d.showModal) { d.showModal(); d.querySelector("input")?.focus(); }
    }));
    document.querySelectorAll("dialog [data-close]").forEach((b) => b.addEventListener("click", () => b.closest("dialog").close()));
  }

  // "Why" buttons open the detail row that follows: the journaled reason and signal values.
  function whys() {
    document.querySelectorAll("[data-why]").forEach((b) => b.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const row = document.getElementById(b.dataset.why);
      if (!row) return;
      row.hidden = !row.hidden;
      b.setAttribute("aria-expanded", String(!row.hidden));
      b.closest("tr")?.classList.toggle("open", !row.hidden);
    }));
  }

  function sortable() {
    document.querySelectorAll("table.sortable").forEach((table) => {
      const heads = [...table.querySelectorAll("thead th")];
      heads.forEach((th, i) => {
        if (!th.dataset.sort) return;
        th.tabIndex = 0;
        const go = () => {
          const dir = th.getAttribute("aria-sort") === "descending" ? "ascending" : "descending";
          heads.forEach((h) => h.removeAttribute("aria-sort"));
          th.setAttribute("aria-sort", dir);
          const body = table.tBodies[0];
          const rows = [...body.rows];
          const val = (r) => {
            const v = r.cells[i].dataset.v ?? r.cells[i].textContent.trim();
            return th.dataset.sort === "num" ? parseFloat(v) || 0 : v.toLowerCase();
          };
          rows.sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * (dir === "ascending" ? 1 : -1));
          rows.forEach((r) => body.appendChild(r));
        };
        th.addEventListener("click", go);
        th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
      });
    });
  }

  // Two charts on one time axis: the curve on top, its drawdown underneath.
  function pair(url, eqId, ddId, labels) {
    const eqEl = document.getElementById(eqId), ddEl = document.getElementById(ddId);
    if (!eqEl || !window.Chart) return;
    // url may be the data itself (the backtest page embeds its result) or an endpoint to fetch.
    (typeof url === "string" ? fetch(url).then((r) => r.json()) : Promise.resolve(url)).then((d) => {
      if (!d.t.length) { eqEl.parentElement.innerHTML = '<p class="empty">No marks yet. The first arrives within a minute of a sleeve starting.</p>'; ddEl.parentElement.remove(); return; }
      const grid = css("--line"), muted = css("--muted");
      const yWidth = (s) => { s.width = 64; };
      const base = {
        maintainAspectRatio: false, animation: false, interaction: {mode: "index", intersect: false},
        plugins: {legend: {display: false}, tooltip: {backgroundColor: css("--raised"), titleColor: css("--text"), bodyColor: css("--text"), borderColor: css("--line-strong"), borderWidth: 1}},
      };
      const eq = new Chart(eqEl, {
        type: "line",
        data: {labels: [], datasets: [
          {label: labels[0], data: [], borderColor: css("--accent"), borderWidth: 1.8, pointRadius: 0, tension: 0},
          {label: labels[1], data: [], borderColor: muted, borderWidth: 1.2, borderDash: [4, 3], pointRadius: 0, tension: 0},
          {label: "Buys", data: [], showLine: false, pointStyle: "triangle", pointRadius: 5, pointBackgroundColor: css("--gain"), pointBorderColor: css("--gain")},
          {label: "Sells", data: [], showLine: false, pointStyle: "triangle", rotation: 180, pointRadius: 5, pointBackgroundColor: css("--loss"), pointBorderColor: css("--loss")},
        ]},
        options: {...base, plugins: {...base.plugins, legend: {display: true, align: "start", labels: {color: css("--text"), boxWidth: 14, boxHeight: 2, filter: (i) => i.datasetIndex < 2}},
          tooltip: {...base.plugins.tooltip, filter: (c) => c.parsed.y !== null, callbacks: {label: (c) => `${c.dataset.label}: ${money.format(c.parsed.y)}`}}},
          scales: {x: {display: false}, y: {afterFit: yWidth, ticks: {color: muted, callback: (v) => money.format(v)}, grid: {color: grid}, border: {display: false}}}},
      });
      const dd = new Chart(ddEl, {
        type: "line",
        data: {labels: [], datasets: [{label: "Drawdown", data: [], borderColor: css("--loss"), backgroundColor: css("--loss-bg"), fill: "origin", borderWidth: 1, pointRadius: 0, tension: 0}]},
        options: {...base, plugins: {...base.plugins, tooltip: {...base.plugins.tooltip, callbacks: {label: (c) => `Drawdown: ${c.parsed.y.toFixed(1)}%`}}},
          scales: {x: {ticks: {color: muted, maxTicksLimit: window.innerWidth < 420 ? 3 : window.innerWidth < 760 ? 4 : 7, maxRotation: 0}, grid: {display: false}},
                   y: {afterFit: yWidth, max: 0, ticks: {color: muted, maxTicksLimit: 3, callback: (v) => `${v}%`}, grid: {color: grid}, border: {display: false}}}},
      });
      const fmt = d.res === "intraday" ? minute : day;
      // Place each fill on the curve at the last point at or before it.
      const marks = (side) => {
        const out = new Array(d.t.length).fill(null);
        const times = d.t.map((x) => Date.parse(x));
        (d.fills || []).filter((f) => f.side === side).forEach((f) => {
          const ft = Date.parse(f.t);
          let i = times.findIndex((x) => x >= ft);
          if (i < 0) i = times.length - 1;
          out[i] = d.equity[i];
        });
        return out;
      };
      const buys = marks("BUY"), sells = marks("SELL");
      const show = (n) => {
        const from = n ? Math.max(0, d.t.length - n) : 0;
        const lab = d.t.slice(from).map(fmt);
        eq.data.labels = lab; eq.data.datasets[0].data = d.equity.slice(from); eq.data.datasets[1].data = d.benchmark.slice(from);
        if (eq.data.datasets[2]) { eq.data.datasets[2].data = buys.slice(from); eq.data.datasets[3].data = sells.slice(from); }
        dd.data.labels = lab; dd.data.datasets[0].data = d.drawdown.slice(from).map((x) => -x * 100);
        eq.update(); dd.update();
      };
      document.querySelectorAll("[data-range]").forEach((b) => b.addEventListener("click", () => {
        document.querySelectorAll("[data-range]").forEach((o) => o.setAttribute("aria-pressed", o === b ? "true" : "false"));
        show(parseInt(b.dataset.range, 10));
      }));
      show(0);
    });
  }

  // TradingView-style price chart: candles, volume, a marker on every fill, and entry/stop/target
  // lines. Clicking a marker shows the reason journaled when the order was sent.
  // src is the data itself, or an endpoint that takes ?interval=.
  function priceChart(boxId, src) {
    const box = document.getElementById(boxId);
    if (!box || !window.LightweightCharts) return;
    const $ = (sel) => box.querySelector(sel);
    const chart = LightweightCharts.createChart($(".pc-canvas"), {
      autoSize: true,
      layout: {background: {type: "solid", color: css("--panel")}, textColor: css("--muted"), fontFamily: getComputedStyle(document.body).fontFamily, attributionLogo: true},
      grid: {vertLines: {color: css("--line")}, horzLines: {color: css("--line")}},
      rightPriceScale: {borderColor: css("--line")}, timeScale: {borderColor: css("--line"), rightOffset: 4},
      crosshair: {mode: 0},
    });
    const candles = chart.addCandlestickSeries({upColor: css("--gain"), downColor: css("--loss"), borderVisible: false, wickUpColor: css("--gain"), wickDownColor: css("--loss")});
    const vol = chart.addHistogramSeries({priceScaleId: "vol", priceFormat: {type: "volume"}, color: css("--line-strong"), lastValueVisible: false, priceLineVisible: false});
    chart.priceScale("vol").applyOptions({scaleMargins: {top: 0.82, bottom: 0}});
    let lines = [], data = null;
    const fmt = (v) => (v >= 100 ? v.toLocaleString("en-GB", {minimumFractionDigits: 2, maximumFractionDigits: 2}) : v.toPrecision(5));
    const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
    const showNote = (ids) => {
      const note = $(".pc-note");
      note.replaceChildren();
      ids.forEach((id) => {
        const n = data.notes[id];
        if (!n) return;
        const block = el("div", "why-block");
        block.append(el("div", "k", `${n.side} · ${n.intent} · ${n.ts} at ${fmt(n.price)}`));
        block.append(el("p", null, n.reason || "Not recorded: this fill predates the order journal."));
        if (n.signal.length) {
          const dl = el("dl", "sig");
          n.signal.forEach(([k, v]) => { const d = el("div"); d.append(el("dt", null, k), el("dd", null, v)); dl.append(d); });
          block.append(dl);
        }
        note.append(block);
      });
      note.hidden = !note.childElementCount;
    };
    const render = (d) => {
      data = d;
      chart.applyOptions({timeScale: {timeVisible: d.interval < 1440, secondsVisible: false}});
      candles.setData(d.candles);
      vol.setData(d.volume.map((v, i) => ({...v, color: d.candles[i] && d.candles[i].close >= d.candles[i].open ? css("--gain-bg") : css("--loss-bg")})));
      candles.setMarkers(d.markers.map((m) => ({...m, color: m.position === "belowBar" ? css("--gain") : css("--loss")})));
      lines.forEach((l) => candles.removePriceLine(l));
      lines = d.lines.map((l) => candles.createPriceLine({price: l.price, title: l.title, lineWidth: 1, lineStyle: 2, axisLabelVisible: true,
        color: l.kind === "stop" ? css("--loss") : l.kind === "target" ? css("--gain") : css("--accent")}));
      chart.timeScale().fitContent();
      $(".pc-source").hidden = d.source !== "marks";
      $(".pc-empty").hidden = d.candles.length > 0;
      showNote([]);
      const tabs = $(".pc-intervals");
      if (tabs && d.intervals) {
        tabs.replaceChildren(...d.intervals.map((k) => {
          const b = el("button", null, k); b.type = "button"; b.setAttribute("aria-pressed", String(k === d.chosen));
          b.addEventListener("click", () => load(k)); return b;
        }));
      }
    };
    const load = (interval) => {
      if (typeof src !== "string") { render(src); return; }
      fetch(src + (interval ? `?interval=${interval}` : "")).then((r) => r.json()).then(render);
    };
    chart.subscribeCrosshairMove((p) => {
      const bar = p.seriesData && p.seriesData.get(candles);
      $(".pc-legend").textContent = bar ? `O ${fmt(bar.open)}  H ${fmt(bar.high)}  L ${fmt(bar.low)}  C ${fmt(bar.close)}` : "";
    });
    chart.subscribeClick((p) => {
      if (!data || !p.time) return;
      const ids = p.hoveredObjectId && data.notes[p.hoveredObjectId] ? [p.hoveredObjectId]
        : data.markers.filter((m) => m.time === p.time).map((m) => m.id);
      if (ids.length) showNote(ids);
    });
    load("");
  }

  // Any form with a strategy picker: show only the chosen strategy's settings, with its sentence filled in.
  function strategyPicker(formId) {
    const form = document.getElementById(formId);
    if (!form) return;
    const sync = () => {
      const strat = form.elements.strategy.value;
      form.querySelectorAll(".params").forEach((p) => { p.hidden = p.dataset.strategy !== strat; });
      const desc = form.querySelector(`.params[data-strategy="${strat}"] .desc`);
      if (desc && desc.dataset.tpl) {
        const vals = JSON.parse(desc.dataset.defaults || "{}");
        form.querySelectorAll(`[name^="p_${strat}__"]`).forEach((i) => { if (i.value !== "") vals[i.name.split("__")[1]] = i.value; });
        desc.textContent = desc.dataset.tpl.replace(/\{(\w+)\}/g, (m, k) => (k in vals ? vals[k] : m));
      }
    };
    form.addEventListener("input", sync); form.addEventListener("change", sync); sync();
    // Only the chosen strategy's inputs go in the URL, so a shared link stays readable.
    form.addEventListener("submit", () => {
      form.querySelectorAll(".params").forEach((p) => p.querySelectorAll("input").forEach((i) => { i.disabled = p.hidden; }));
      form.querySelectorAll("input").forEach((i) => { if (i.value === "") i.disabled = true; });
    });
  }

  // New-sleeve form: show the chosen strategy's settings, keep a plain-English summary, run the look-back.
  function sleeveForm() {
    const form = document.getElementById("sleeve-form");
    if (!form) return;
    const $ = (n) => form.elements[n];
    const pct = (x) => `${(x * 100).toFixed(0)}%`;
    const sync = () => {
      const strat = $("strategy").value;
      form.querySelectorAll(".params").forEach((p) => { p.hidden = p.dataset.strategy !== strat; });
      const desc = form.querySelector(`.params[data-strategy="${strat}"] .desc`);
      if (desc && desc.dataset.tpl) {
        const vals = JSON.parse(desc.dataset.defaults || "{}");
        form.querySelectorAll(`[name^="p_${strat}__"]`).forEach((i) => { if (i.value !== "") vals[i.name.split("__")[1]] = i.value; });
        desc.textContent = desc.dataset.tpl.replace(/\{(\w+)\}/g, (m, k) => (k in vals ? vals[k] : m));
      }
      const opt = $("strategy").selectedOptions[0];
      document.getElementById("g1-note").hidden = opt.dataset.g1 === "PASS";
      const prof = $("risk_profile").selectedOptions[0].dataset;
      const pair = ($("instrument").value || "").toUpperCase();
      const quote = pair.split("/")[1] || "";
      const cap = Number($("starting_balance").value || 0);
      const items = [
        `Trade ${pair || "a pair"} with ${money.format(cap)} ${quote} of simulated money, deciding on ${$("bar_spec").selectedOptions[0].textContent.split(" (")[0]} bars.`,
        `Use ${opt.textContent.split(" (")[0]}: ${desc ? desc.textContent : ""}`,
        `Hold at most ${pct(Number(prof.cap))} of its capital in ${pair.split("/")[0] || "the coin"}.`,
        `Pause for a day after losing ${pct(Number(prof.day))} in a day, and halt for your review at a ${pct(Number(prof.dd))} drawdown.`,
      ];
      const sl = $("stop_loss_pct").value, tp = $("take_profit_pct").value, rpt = $("risk_per_trade_pct").value;
      if (sl || tp) items.push(`Exit any trade ${[sl && `${sl}% below entry`, tp && `${tp}% above entry`].filter(Boolean).join(" or ")}.`);
      if (rpt) items.push(sl ? `Size each trade to lose about ${rpt}% of capital if the stop is hit.` : "Risk per trade needs a stop-loss; add one or clear it.");
      if ($("max_notional").value) items.push(`Never place a single order above ${money.format(Number($("max_notional").value))} ${quote}.`);
      const ul = document.getElementById("summary");
      ul.replaceChildren(...items.map((t) => Object.assign(document.createElement("li"), {textContent: t})));
      if (!$("name").dataset.touched) {
        $("name").value = `${(pair.split("/")[0] || "").toLowerCase()}-${strat.replace(/_/g, "-")}`.replace(/^-/, "").slice(0, 40);
      }
    };
    $("name").addEventListener("input", () => { $("name").dataset.touched = "1"; });
    form.addEventListener("input", sync); form.addEventListener("change", sync); sync();

    let chart;
    document.getElementById("run-preview").addEventListener("click", async (ev) => {
      const box = document.getElementById("preview");
      const btn = ev.currentTarget;
      btn.disabled = true; btn.textContent = "Running…";
      const q = new URLSearchParams(new FormData(form));
      try {
        const r = await fetch(`/api/preview?${q}`);
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || `failed (${r.status})`);
        const f = (x, digits = 1) => (x === null || x === undefined ? "n/a" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`);
        const n = (x) => (x === null || x === undefined ? "n/a" : x.toFixed(2));
        box.innerHTML = `<div class="chart" style="height:150px"><canvas id="pv"></canvas></div>
          <table class="compact"><thead><tr><th></th><th class="num">These settings</th><th class="num">Holding</th></tr></thead><tbody>
          <tr><td>Return</td><td class="num">${f(d.strategy.total_return)}</td><td class="num">${f(d.hold.total_return)}</td></tr>
          <tr><td>Sharpe</td><td class="num">${n(d.strategy.sharpe)}</td><td class="num">${n(d.hold.sharpe)}</td></tr>
          <tr><td>Worst drawdown</td><td class="num">${f(d.strategy.max_drawdown)}</td><td class="num">${f(d.hold.max_drawdown)}</td></tr>
          <tr><td>Closed trades</td><td class="num">${d.trades.trades}${d.trades.trades ? ` · ${f(d.trades.win_rate, 0).replace("+", "")} won` : ""}</td><td class="num">1</td></tr>
          <tr><td>Fees paid</td><td class="num">${money.format(d.fees)}</td><td class="num"></td></tr></tbody></table>
          <p class="muted" style="margin:8px 0 0;font-size:12px">${d.pair}, ${d.from} to ${d.to} (${d.days} days), daily decisions. In-sample: a sense check, not a G1 test.</p>`;
        if (chart) chart.destroy();
        chart = new Chart(document.getElementById("pv"), {
          type: "line",
          data: {labels: d.t.map(day), datasets: [
            {label: "These settings", data: d.equity, borderColor: css("--accent"), borderWidth: 1.6, pointRadius: 0},
            {label: "Holding", data: d.benchmark, borderColor: css("--muted"), borderWidth: 1.1, borderDash: [4, 3], pointRadius: 0}]},
          options: {maintainAspectRatio: false, animation: false, interaction: {mode: "index", intersect: false},
            plugins: {legend: {labels: {color: css("--text"), boxWidth: 12, boxHeight: 2}}},
            scales: {x: {ticks: {color: css("--muted"), maxTicksLimit: 4, maxRotation: 0}, grid: {display: false}},
                     y: {ticks: {color: css("--muted"), maxTicksLimit: 4, callback: (v) => money.format(v)}, grid: {color: css("--line")}}}},
        });
      } catch (e) {
        box.innerHTML = "";
        box.appendChild(Object.assign(document.createElement("p"), {className: "loss", textContent: `Look-back not available: ${e.message}`}));
      } finally {
        btn.disabled = false; btn.textContent = "Run again";
      }
    });
  }

  return {sortable, dialogs, whys, strategyPicker, priceChart, sleeveForm, bookCharts: (url) => pair(url, "eq", "dd", ["Book", "Buy-and-hold"]), pair};
})();
