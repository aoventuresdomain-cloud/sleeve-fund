// Small helpers for the console: sortable tables and the equity/drawdown chart pair.
window.Console = (() => {
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  // Tokens are hex or rgba(); charts need a see-through version of a hex token.
  const rgba = (c, a) => {
    if (!c.startsWith("#")) return c;
    const n = parseInt(c.length === 4 ? c.slice(1).replace(/./g, "$&$&") : c.slice(1, 7), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };
  // A round axis floor for negative percentages: -6.8 becomes -8, -23 becomes -25, -79 becomes -80.
  // A round axis range with about five steps of 1, 2, 2.5 or 5 times a power of ten, padded so lines don't touch the edges.
  const niceRange = (lo, hi) => {
    const pad = Math.max((hi - lo) * 0.06, 0.5);
    lo -= pad; hi += pad;
    const raw = (hi - lo) / 5, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    return {min: Math.floor(lo / step) * step, max: Math.ceil(hi / step) * step, step};
  };
  const niceFloor = (v) => { const step = v > -10 ? 2 : v > -30 ? 5 : 10; return Math.floor(v / step) * step; };
  // A dashed hairline under the cursor, like a trading terminal.
  const crosshair = {id: "crosshair", afterDatasetsDraw(c) {
    const a = c.getActiveElements();
    if (!a.length) return;
    const x = a[0].element.x, {top, bottom} = c.chartArea, g = c.ctx;
    g.save(); g.strokeStyle = css("--line-strong"); g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke(); g.restore();
  }};
  const money = new Intl.NumberFormat("en-GB", {maximumFractionDigits: 0});
  const day = (t) => new Date(t).toLocaleDateString("en-GB", {day: "2-digit", month: "short", year: "2-digit", timeZone: "UTC"});
  const minute = (t) => new Date(t).toLocaleString("en-GB", {day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC"});

  // Listeners sit on the document, so they keep working after live updates replace a panel.
  const once = (key, fn) => { if (!document.documentElement.dataset[key]) { document.documentElement.dataset[key] = "1"; fn(); } };

  function dialogs() {
    once("dialogsBound", () => document.addEventListener("click", (e) => {
      const open = e.target.closest("[data-open]");
      if (open) {
        const d = document.getElementById(open.dataset.open);
        if (d && d.showModal && !d.open) { d.showModal(); d.querySelector("input")?.focus(); }
        return;
      }
      const close = e.target.closest("dialog [data-close]");
      if (close) close.closest("dialog").close();
    }));
  }

  // "Why" buttons open the detail row that follows: the journaled reason and signal values.
  function whys() {
    once("whysBound", () => document.addEventListener("click", (e) => {
      const b = e.target.closest("[data-why]");
      if (!b) return;
      e.stopPropagation();
      const row = document.getElementById(b.dataset.why);
      if (!row) return;
      row.hidden = !row.hidden;
      b.setAttribute("aria-expanded", String(!row.hidden));
      b.closest("tr")?.classList.toggle("open", !row.hidden);
    }));
  }

  function sortBy(table, i, dir) {
    const heads = [...table.querySelectorAll("thead th")];
    const th = heads[i];
    if (!th) return;
    heads.forEach((h) => h.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", dir);
    const body = table.tBodies[0];
    const val = (r) => {
      const c = r.cells[i];
      const v = c ? (c.dataset.v ?? c.textContent.trim()) : "";
      return th.dataset.sort === "num" ? parseFloat(v) || 0 : v.toLowerCase();
    };
    [...body.rows].sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * (dir === "ascending" ? 1 : -1))
      .forEach((r) => body.appendChild(r));
  }

  function sortable() {
    const prep = () => document.querySelectorAll("table.sortable thead th[data-sort]").forEach((th) => { th.tabIndex = 0; });
    prep();
    document.addEventListener("live:swap", prep);
    const go = (th) => {
      const table = th.closest("table");
      sortBy(table, [...th.parentElement.children].indexOf(th), th.getAttribute("aria-sort") === "descending" ? "ascending" : "descending");
    };
    once("sortBound", () => {
      document.addEventListener("click", (e) => { const th = e.target.closest("table.sortable thead th[data-sort]"); if (th) go(th); });
      document.addEventListener("keydown", (e) => {
        const th = e.target.closest?.("table.sortable thead th[data-sort]");
        if (th && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); go(th); }
      });
    });
  }

  // Two charts on one time axis: return against the benchmark on top, drawdown underneath.
  // Both series are rebased to 0% at the start of the chosen range, so they share one honest scale
  // whatever the range, and the legend reads out the values under the cursor.
  function pair(url, eqId, ddId, labels) {
    const eqEl = document.getElementById(eqId), ddEl = document.getElementById(ddId);
    if (!eqEl || !window.Chart) return;
    // url may be the data itself (the backtest page embeds its result) or an endpoint to fetch.
    (typeof url === "string" ? fetch(url).then((r) => r.json()) : Promise.resolve(url)).then((d) => {
      if (!d.t.length) { eqEl.parentElement.innerHTML = '<p class="empty">No marks yet. The first arrives within a minute of a strategy starting.</p>'; ddEl.parentElement.remove(); return; }
      const grid = css("--line"), muted = css("--muted"), faint = css("--faint"), accent = css("--accent");
      Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
      Chart.defaults.font.size = 11;
      const yWidth = (s) => { s.width = 56; };
      const pctTick = (v) => `${v > 0 ? "+" : ""}${Number(v.toFixed(Math.abs(v) < 10 ? 1 : 0))}%`;
      const base = {
        maintainAspectRatio: false, animation: false, interaction: {mode: "index", intersect: false},
        plugins: {legend: {display: false}, tooltip: {enabled: false}},
      };
      const fade = (c) => {
        const {ctx, chartArea} = c.chart;
        if (!chartArea) return null;
        const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
        g.addColorStop(0, rgba(accent, 0.28)); g.addColorStop(1, rgba(accent, 0));
        return g;
      };
      const eq = new Chart(eqEl, {
        type: "line",
        data: {labels: [], datasets: [
          {label: labels[0], data: [], borderColor: accent, backgroundColor: fade, fill: "origin", borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, pointHoverBackgroundColor: accent, pointHoverBorderWidth: 0, tension: 0},
          {label: labels[1], data: [], borderColor: faint, borderWidth: 1.3, borderDash: [3, 3], pointRadius: 0, pointHoverRadius: 3, pointHoverBackgroundColor: muted, tension: 0},
          {label: "Buys", data: [], showLine: false, pointStyle: "triangle", pointRadius: 4.5, pointHoverRadius: 4.5, pointBackgroundColor: css("--gain"), pointBorderWidth: 0},
          {label: "Sells", data: [], showLine: false, pointStyle: "triangle", rotation: 180, pointRadius: 4.5, pointHoverRadius: 4.5, pointBackgroundColor: css("--loss"), pointBorderWidth: 0},
        ]},
        options: {...base,
          scales: {x: {display: false},
                   y: {position: "right", afterFit: yWidth, ticks: {color: muted, padding: 8, callback: pctTick}, grid: {color: grid, drawTicks: false}, border: {display: false}}}},
        plugins: [crosshair],
      });
      const dd = new Chart(ddEl, {
        type: "line",
        data: {labels: [], datasets: [{label: "Drawdown", data: [], borderColor: css("--loss"), backgroundColor: css("--loss-bg"), fill: "origin", borderWidth: 1.2, pointRadius: 0, pointHoverRadius: 3, pointHoverBackgroundColor: css("--loss"), tension: 0}]},
        options: {...base,
          scales: {x: {ticks: {color: muted, autoSkip: true, autoSkipPadding: 40, maxRotation: 0, padding: 6, align: "inner"}, grid: {display: false}, border: {display: false}},
                   y: {position: "right", max: 0, afterFit: yWidth, ticks: {color: muted, maxTicksLimit: 3, padding: 8, callback: pctTick}, grid: {color: grid, drawTicks: false}, border: {display: false}}}},
        plugins: [crosshair],
      });

      // The legend doubles as the readout: range figures at rest, the hovered day's figures under the cursor.
      let legend = eqEl.parentElement.previousElementSibling;
      if (!legend || !legend.classList.contains("chart-legend")) {
        legend = Object.assign(document.createElement("div"), {className: "chart-legend"});
        legend.setAttribute("aria-live", "off");
        eqEl.parentElement.before(legend);
      }
      const signed = (x) => (x === null || x === undefined || Number.isNaN(x) ? "n/a" : `${x >= 0 ? "+" : ""}${x.toFixed(2)}%`);
      let view = null;
      const readout = (i) => {
        const at = i ?? view.ret.length - 1;
        const when = i === null || i === undefined ? `${view.lab[0]} to ${view.lab[view.lab.length - 1]}` : view.lab[at];
        legend.innerHTML = "";
        const item = (cls, name, value, tone) => {
          const s = document.createElement("span");
          if (cls !== null) s.append(Object.assign(document.createElement("i"), {className: cls}));
          s.append(name);
          s.append(Object.assign(document.createElement("b"), {textContent: value, className: tone || ""}));
          legend.append(s);
        };
        item("", labels[0], signed(view.ret[at]), view.ret[at] >= 0 ? "gain" : "loss");
        item("bench", labels[1], signed(view.bench[at]));
        item(null, i === null || i === undefined ? "Worst drawdown" : "Drawdown", signed(i === null || i === undefined ? Math.min(...view.dd) : view.dd[at]));
        legend.append(Object.assign(document.createElement("span"), {textContent: when, className: "faint"}));
      };
      const hover = (_e, els) => readout(els.length ? els[0].index : null);
      eq.options.onHover = hover; dd.options.onHover = hover;
      [eqEl, ddEl].forEach((el) => el.addEventListener("mouseleave", () => { readout(null); sync(null); }));
      // Hovering either chart moves the hairline on both.
      const sync = (i) => [eq, dd].forEach((c) => {
        c.setActiveElements(i === null ? [] : [{datasetIndex: 0, index: i}]);
        c.tooltip.setActiveElements(i === null ? [] : [{datasetIndex: 0, index: i}], {x: 0, y: 0});
        c.update("none");
      });
      eqEl.addEventListener("mousemove", () => { const a = eq.getActiveElements(); if (a.length) { dd.setActiveElements([{datasetIndex: 0, index: a[0].index}]); dd.tooltip.setActiveElements([{datasetIndex: 0, index: a[0].index}], {x: 0, y: 0}); dd.update("none"); } });
      ddEl.addEventListener("mousemove", () => { const a = dd.getActiveElements(); if (a.length) { eq.setActiveElements([{datasetIndex: 0, index: a[0].index}]); eq.tooltip.setActiveElements([{datasetIndex: 0, index: a[0].index}], {x: 0, y: 0}); eq.update("none"); } });

      const intraday = d.res === "intraday";
      const times = d.t.map((x) => Date.parse(x));
      // Place each fill on the curve at the first point at or after it.
      const fillIdx = (side) => {
        const out = new Set();
        (d.fills || []).filter((f) => f.side === side).forEach((f) => {
          const ft = Date.parse(f.t);
          let i = times.findIndex((x) => x >= ft);
          if (i < 0) i = times.length - 1;
          out.add(i);
        });
        return out;
      };
      const buys = fillIdx("BUY"), sells = fillIdx("SELL");
      const rebase = (arr, from) => {
        const b = arr.slice(from).find((v) => v !== null && v !== undefined && v > 0);
        return arr.slice(from).map((v) => (v === null || v === undefined || !b ? null : (v / b - 1) * 100));
      };
      const show = (n) => {
        const from = n ? Math.max(0, d.t.length - n) : 0;
        const span = (times[times.length - 1] - times[from]) / 864e5;
        // Short ranges show the day, long ones the month and year; intraday adds the time.
        const opts = intraday && span < 3 ? {day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"}
          : span > 200 ? {month: "short", year: "2-digit"} : {day: "2-digit", month: "short"};
        const lab = times.slice(from).map((t) => new Date(t).toLocaleString("en-GB", {...opts, timeZone: "UTC"}));
        const ret = rebase(d.equity, from), bench = rebase(d.benchmark, from);
        view = {lab, ret, bench, dd: d.drawdown.slice(from).map((x) => -x * 100)};
        eq.data.labels = lab; eq.data.datasets[0].data = ret; eq.data.datasets[1].data = bench;
        eq.data.datasets[2].data = ret.map((v, i) => (buys.has(i + from) ? v : null));
        eq.data.datasets[3].data = ret.map((v, i) => (sells.has(i + from) ? v : null));
        const vals = [...ret, ...bench].filter((v) => v !== null);
        const r = niceRange(Math.min(0, ...vals), Math.max(0, ...vals));
        Object.assign(eq.options.scales.y, {min: r.min, max: r.max}); eq.options.scales.y.ticks.stepSize = r.step;
        // Drawdown axis fits the worst point in view, so a shallow drawdown isn't a flat line at the top.
        const worst = Math.min(...view.dd);
        dd.options.scales.y.min = worst > -1 ? -1 : niceFloor(worst * 1.1);
        dd.data.labels = lab; dd.data.datasets[0].data = view.dd;
        eq.update(); dd.update();
        readout(null);
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
    const accent = css("--accent");
    const fmt = (v) => (v >= 100 ? v.toLocaleString("en-GB", {minimumFractionDigits: 2, maximumFractionDigits: 2}) : v.toPrecision(5));
    const chart = LightweightCharts.createChart($(".pc-canvas"), {
      localization: {priceFormatter: fmt},
      autoSize: true,
      layout: {background: {type: "solid", color: css("--panel")}, textColor: css("--muted"), fontSize: 11, fontFamily: getComputedStyle(document.body).fontFamily, attributionLogo: true},
      grid: {vertLines: {visible: false}, horzLines: {color: css("--line")}},
      rightPriceScale: {borderVisible: false, scaleMargins: {top: 0.08, bottom: 0.08}},
      timeScale: {borderVisible: false, rightOffset: 6},
      crosshair: {mode: 0, vertLine: {color: css("--line-strong"), labelBackgroundColor: css("--raised")}, horzLine: {color: css("--line-strong"), labelBackgroundColor: css("--raised")}},
    });
    const candles = chart.addCandlestickSeries({upColor: css("--gain"), downColor: css("--loss"), borderVisible: false, wickUpColor: css("--gain"), wickDownColor: css("--loss")});
    // Candles built from the sleeve's own marks have no range inside the bar, so they draw as a line instead.
    const area = chart.addAreaSeries({lineColor: accent, topColor: rgba(accent, 0.22), bottomColor: rgba(accent, 0), lineWidth: 2, visible: false});
    const vol = chart.addHistogramSeries({priceScaleId: "vol", color: rgba(css("--muted"), 0.3), priceFormat: {type: "volume"}, lastValueVisible: false, priceLineVisible: false});
    chart.priceScale("vol").applyOptions({scaleMargins: {top: 0.86, bottom: 0}});
    let main = candles;
    let lines = [], data = null;
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
    let current = "";
    const render = (d, keepView = false) => {
      data = d;
      chart.applyOptions({timeScale: {timeVisible: d.interval < 1440, secondsVisible: false}});
      const flat = d.source === "marks" || d.candles.every((c) => c.high === c.low);
      const was = main;
      main = flat ? area : candles;
      if (was !== main) { lines.forEach((l) => was.removePriceLine(l)); lines = []; was.setMarkers([]); }
      candles.applyOptions({visible: !flat}); area.applyOptions({visible: flat});
      if (flat) { area.setData(d.candles.map((c) => ({time: c.time, value: c.close}))); candles.setData([]); }
      else { candles.setData(d.candles); area.setData([]); }
      // No traded volume (marks, or a venue that doesn't report it): leave the band empty rather than a row of zeros.
      const hasVol = d.volume.some((v) => v.value > 0);
      // Keep the candles clear of the volume band when there is one.
      chart.priceScale("right").applyOptions({scaleMargins: {top: 0.08, bottom: hasVol ? 0.18 : 0.06}});
      vol.setData(hasVol ? d.volume : []);
      main.setMarkers(d.markers.map((m) => ({...m, color: m.position === "belowBar" ? css("--gain") : css("--loss")})));
      lines.forEach((l) => main.removePriceLine(l));
      lines = d.lines.map((l) => main.createPriceLine({price: l.price, title: l.title, lineWidth: 1, lineStyle: 2, axisLabelVisible: true,
        color: l.kind === "stop" ? css("--loss") : l.kind === "target" ? css("--gain") : accent}));
      if (!keepView) chart.timeScale().fitContent();
      $(".pc-source").hidden = d.source !== "marks";
      if (d.note) $(".pc-source").textContent = d.note;
      $(".pc-empty").hidden = d.candles.length > 0;
      if (!keepView) showNote([]);
      const tabs = $(".pc-intervals");
      if (tabs && d.intervals) {
        tabs.replaceChildren(...d.intervals.map((k) => {
          const b = el("button", null, k); b.type = "button"; b.setAttribute("aria-pressed", String(k === d.chosen));
          b.addEventListener("click", () => load(k)); return b;
        }));
      }
    };
    const load = (interval, keepView = false) => {
      if (typeof src !== "string") { render(src); return; }
      current = interval;
      return fetch(src + (interval ? `?interval=${interval}` : ""), {cache: "no-store"}).then((r) => r.json()).then((d) => render(d, keepView));
    };
    // A sleeve's chart follows the market: new candles and fills appear without a reload.
    if (typeof src === "string") setInterval(() => { if (!document.hidden) load(current, true).catch(() => {}); }, 30000);
    chart.subscribeCrosshairMove((p) => {
      const bar = p.seriesData && p.seriesData.get(main);
      $(".pc-legend").textContent = !bar ? "" : main === area ? `Price ${fmt(bar.value)}` : `O ${fmt(bar.open)}  H ${fmt(bar.high)}  L ${fmt(bar.low)}  C ${fmt(bar.close)}`;
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

  // New-sleeve form: show the chosen strategy's settings, keep a plain-English summary.
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
      // A G1 pass holds for the instrument and bar length it was tested on, nothing else.
      const minutes = {MINUTE: 1, HOUR: 60, DAY: 1440};
      const [step, unit] = document.getElementById("bar_spec").value.split("-");
      const here = `${($("instrument").value || "").toUpperCase()}@${Number(step) * (minutes[unit] || 0)}`;
      document.getElementById("g1-note").hidden = (opt.dataset.g1 || "").split("|").includes(here);
      const prof = $("risk_profile").selectedOptions[0].dataset;
      const pair = ($("instrument").value || "").toUpperCase();
      const quote = pair.split("/")[1] || "";
      const cap = Number($("starting_balance").value || 0);
      const items = [
        `Trade ${pair || "the instrument"} with ${money.format(cap)} ${quote} of simulated money, deciding on ${document.getElementById("bar_spec").selectedOptions[0].textContent.split(" (")[0]} bars.`,
        `Use ${opt.textContent.split(" (")[0]}: ${desc ? desc.textContent : ""}`,
        `Hold at most ${pct(Number(prof.cap))} of its capital in ${pair.split("/")[0] || "the instrument"}.`,
        `Pause for a day after losing ${pct(Number(prof.day))} in a day, and halt for your review at a ${pct(Number(prof.dd))} drawdown.`,
      ];
      const sl = $("stop_loss_pct").value, tp = $("take_profit_pct").value, rpt = $("risk_per_trade_pct").value;
      if (sl || tp) items.push(`Exit any trade ${[sl && `${sl}% below entry`, tp && `${tp}% above entry`].filter(Boolean).join(" or ")}.`);
      if (rpt) items.push(sl ? `Size each trade to lose about ${rpt}% of capital if the stop is hit.` : "Risk per trade needs a stop-loss; add one or clear it.");
      if ($("execution").value === "maker") items.push(`Rest each signal order as a post-only limit for the maker fee, and send whatever hasn't filled after ${$("maker_wait_minutes").value || 15} minutes at market. Protective exits go at market.`);
      if ($("max_notional").value) items.push(`Never place a single order above ${money.format(Number($("max_notional").value))} ${quote}.`);
      const ul = document.getElementById("summary");
      ul.replaceChildren(...items.map((t) => Object.assign(document.createElement("li"), {textContent: t})));
      if (!$("name").dataset.touched) {
        $("name").value = `${(pair.split("/")[0] || "").toLowerCase()}-${strat.replace(/_/g, "-")}`.replace(/^-/, "").slice(0, 40);
      }
    };
    $("name").addEventListener("input", () => { $("name").dataset.touched = "1"; });
    form.addEventListener("input", sync); form.addEventListener("change", sync); sync();
    wizard(form);
    orderFields("sleeve-form");

    // A backtest of exactly these settings, in a new tab, so the form keeps what was typed.
    const bt = document.getElementById("bt-these");
    bt.addEventListener("click", () => {
      const q = new URLSearchParams(new FormData(form));
      ["name", "reason", "warmup_bars", "tested_bar_spec", "from", "account"].forEach((k) => q.delete(k));
      q.set("run", "1");
      bt.href = `/backtest?${q}`;
    });
  }

  // Order type: the wait only matters for maker-first orders, so it shows only then.
  function orderFields(formId) {
    const form = document.getElementById(formId);
    if (!form || !form.elements.execution) return;
    const wait = form.querySelector("[data-when-maker]");
    const sync = () => { wait.hidden = form.elements.execution.value !== "maker"; };
    form.elements.execution.addEventListener("change", sync); sync();
  }

  // The new-sleeve form one section at a time, with Next and Back. Without JavaScript every section shows.
  function wizard(form) {
    const steps = [...form.querySelectorAll("fieldset.step")];
    if (steps.length < 2) return;
    const make = (tag, props) => Object.assign(document.createElement(tag), props);
    const nav = make("ol", {className: "wiz-nav"});
    nav.setAttribute("aria-label", "Steps");
    const bar = make("div", {className: "wiz-buttons"});
    const back = make("button", {type: "button", className: "secondary", textContent: "Back"});
    const next = make("button", {type: "button", textContent: "Next"});
    const go = form.querySelector("aside button:not([type=button])").cloneNode(true);
    bar.append(back, next, go);
    let at = 0;
    const show = (i, focus = true) => {
      at = i;
      steps.forEach((s, j) => { s.hidden = j !== i; });
      tabs.forEach((b, j) => { b.setAttribute("aria-current", j === i ? "step" : "false"); b.classList.toggle("done", j < i); });
      back.hidden = i === 0; next.hidden = i === steps.length - 1; go.hidden = !next.hidden;
      steps[i].append(bar);
      nav.scrollLeft = tabs[i].parentElement.offsetLeft - nav.offsetLeft - 8;  // keep the current step in view on a phone
      if (focus) { steps[i].scrollIntoView({block: "nearest"}); steps[i].querySelector("input:not([type=hidden]), select")?.focus({preventScroll: true}); }
    };
    const valid = () => {
      const bad = [...steps[at].querySelectorAll("input, select")].find((el) => !el.checkValidity());
      if (bad) bad.reportValidity();
      return !bad;
    };
    const tabs = steps.map((s, i) => {
      const b = make("button", {type: "button", textContent: s.querySelector("legend").textContent});
      b.addEventListener("click", () => { if (i <= at || valid()) show(i); });
      nav.append(make("li")); nav.lastChild.append(b);
      return b;
    });
    form.before(nav);
    back.addEventListener("click", () => show(at - 1));
    next.addEventListener("click", () => { if (valid()) show(at + 1); });
    // Enter moves on rather than submitting from the middle of the form.
    form.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target.tagName === "INPUT" && at < steps.length - 1) { e.preventDefault(); next.click(); }
    });
    // If the browser blocks a submit, open the step holding the first field it complains about.
    let jumped = false;
    form.addEventListener("invalid", (e) => {
      if (jumped) return;
      jumped = true; setTimeout(() => { jumped = false; });
      const i = steps.findIndex((s) => s.contains(e.target));
      if (i >= 0 && i !== at) show(i, false);
    }, true);
    show(0, false);
  }

  return {sortable, sortBy, dialogs, whys, strategyPicker, priceChart, sleeveForm, orderFields, bookCharts: (url) => pair(url, "eq", "dd", ["Book", "Buy-and-hold"]), pair};
})();
