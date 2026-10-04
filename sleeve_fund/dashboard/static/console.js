// Small helpers for the console: sortable tables and the equity/drawdown chart pair.
window.Console = (() => {
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  // Tokens are hex or rgba(); charts need a see-through version of a hex token.
  const rgba = (c, a) => {
    if (!c.startsWith("#")) return c;
    const n = parseInt(c.length === 4 ? c.slice(1).replace(/./g, "$&$&") : c.slice(1, 7), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };
  // A round axis range with about five steps of 1, 2, 2.5 or 5 times a power of ten, padded so lines don't touch the edges.
  const niceRange = (lo, hi, minPad = 0.5) => {
    const pad = Math.max((hi - lo) * 0.06, minPad);
    lo -= pad; hi += pad;
    const raw = (hi - lo) / 5, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    return {min: Math.floor(lo / step) * step, max: Math.ceil(hi / step) * step, step};
  };
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

  // Two charts on one time axis: growth on top, drawdown underneath.
  // By default the top chart shows equity in money, scaled to its own range, so day-to-day growth is
  // readable. "Compare" rebases equity and buy-and-hold to 0% at the start of the range so the two
  // share one honest scale. The legend reads out the hovered day: value, change on the day, drawdown.
  // opts.compare starts in compare mode (the backtest, whose point is the comparison).
  function pair(url, eqId, ddId, labels, opts = {}) {
    const eqEl = document.getElementById(eqId), ddEl = document.getElementById(ddId);
    if (!eqEl || !window.Chart) return;
    const scope = eqEl.closest("section") || document;
    // url may be the data itself (the backtest page embeds its result) or an endpoint to fetch.
    (typeof url === "string" ? fetch(url).then((r) => r.json()) : Promise.resolve(url)).then((d) => {
      if (!d.t.length) { eqEl.parentElement.innerHTML = '<p class="empty">No marks yet. The first arrives within a minute of a strategy starting.</p>'; ddEl.parentElement.remove(); return; }
      const grid = css("--line"), muted = css("--muted"), faint = css("--faint"), accent = css("--accent");
      Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
      Chart.defaults.font.size = 11;
      const yWidth = (s) => { s.width = 64; };
      const pctTick = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Number(Math.abs(v).toFixed(Math.abs(v) < 10 ? 1 : 0))}%`;
      let compare = !!opts.compare, step = 1;
      const moneyTick = (v) => v.toLocaleString("en-GB", {maximumFractionDigits: step < 1 ? 2 : 0, minimumFractionDigits: step < 1 ? 2 : 0});
      const base = {
        maintainAspectRatio: false, animation: false, interaction: {mode: "index", intersect: false},
        plugins: {legend: {display: false}, tooltip: {enabled: false}},
      };
      const fade = (c) => {
        const {ctx, chartArea} = c.chart;
        if (!chartArea) return null;
        const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
        g.addColorStop(0, rgba(accent, 0.24)); g.addColorStop(1, rgba(accent, 0));
        return g;
      };
      // Dates under the drawdown chart: one label per week, month or quarter boundary, never a
      // squeezed row of every other day. tickAt maps a point's index to its label.
      let tickAt = {};
      const xTicks = {color: muted, autoSkip: false, maxRotation: 0, padding: 6, align: "center",
                      callback: (_v, i) => tickAt[i] ?? null};
      const eq = new Chart(eqEl, {
        type: "line",
        data: {labels: [], datasets: [
          {label: labels[0], data: [], borderColor: accent, backgroundColor: fade, fill: "start", borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, pointHoverBackgroundColor: accent, pointHoverBorderWidth: 0, tension: 0},
          {label: labels[1], data: [], borderColor: muted, borderWidth: 1.3, borderDash: [3, 3], pointRadius: 0, pointHoverRadius: 3, pointHoverBackgroundColor: muted, tension: 0},
          {label: "Buys", data: [], showLine: false, pointStyle: "triangle", pointRadius: 4.5, pointHoverRadius: 4.5, pointBackgroundColor: css("--gain"), pointBorderWidth: 0},
          {label: "Sells", data: [], showLine: false, pointStyle: "triangle", rotation: 180, pointRadius: 4.5, pointHoverRadius: 4.5, pointBackgroundColor: css("--loss"), pointBorderWidth: 0},
        ]},
        options: {...base,
          scales: {x: {display: false},
                   y: {position: "right", afterFit: yWidth, ticks: {color: muted, padding: 8, callback: (v) => (compare ? pctTick(v) : moneyTick(v))}, grid: {color: grid, drawTicks: false}, border: {display: false}}}},
        plugins: [crosshair],
      });
      const dd = new Chart(ddEl, {
        type: "line",
        data: {labels: [], datasets: [{label: "Drawdown", data: [], borderColor: css("--loss"), backgroundColor: css("--loss-bg"), fill: "origin", borderWidth: 1.2, pointRadius: 0, pointHoverRadius: 3, pointHoverBackgroundColor: css("--loss"), tension: 0}]},
        options: {...base,
          scales: {x: {ticks: xTicks, grid: {display: false}, border: {display: false}},
                   y: {position: "right", max: 0, afterFit: yWidth, ticks: {color: muted, padding: 8, callback: pctTick}, grid: {color: grid, drawTicks: false}, border: {display: false}}}},
        plugins: [crosshair],
      });

      // The legend doubles as the readout: the range's figures at rest, the hovered day's under the cursor.
      let legend = eqEl.parentElement.previousElementSibling;
      if (!legend || !legend.classList.contains("chart-legend")) {
        legend = Object.assign(document.createElement("div"), {className: "chart-legend"});
        legend.setAttribute("aria-live", "off");
        eqEl.parentElement.before(legend);
      }
      // Two decimals, or three when a small daily move would otherwise read as 0.00%.
      const pct = (x) => (x === null || x === undefined || !Number.isFinite(x) ? "n/a"
        : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(x !== 0 && Math.abs(x) < 0.01 ? 3 : 2)}%`);
      const cash = (x) => x.toLocaleString("en-GB", {minimumFractionDigits: 2, maximumFractionDigits: 2});
      const scash = (x) => `${x >= 0 ? "+" : "−"}${cash(Math.abs(x))}`;
      const tone = (x) => (x > 0 ? "gain" : x < 0 ? "loss" : "");
      // cur is the series on screen: the daily history, or the last day or week at minutes' resolution.
      let cur = d;
      const isIntra = () => cur.res === "intraday";
      let view = null, hovering = false;
      const readout = (i) => {
        const rest = i === null || i === undefined;
        const at = rest ? view.eq.length - 1 : i;
        legend.innerHTML = "";
        const item = (cls, name, value, t) => {
          const s = document.createElement("span");
          if (cls !== null) s.append(Object.assign(document.createElement("i"), {className: cls}));
          s.append(name);
          s.append(Object.assign(document.createElement("b"), {textContent: value, className: t || ""}));
          legend.append(s);
        };
        if (compare) {
          item("", labels[0], pct(view.ret[at]), tone(view.ret[at]));
          item("bench", labels[1], pct(view.bench[at]));
        } else if (rest) {
          const first = view.eq[0], last = view.eq[at], ch = last - first;
          item("", labels[0], cash(last));
          item(null, "Change", `${scash(ch)} (${pct(first ? (ch / first) * 100 : NaN)})`, tone(ch));
        } else {
          item("", labels[0], cash(view.eq[at]));
          const prev = at > 0 ? view.eq[at - 1] : (view.prior ?? null);
          if (prev !== null) {
            const ch = view.eq[at] - prev;
            item(null, isIntra() ? "Since the previous point" : "On the day", `${scash(ch)} (${pct(prev ? (ch / prev) * 100 : NaN)})`, tone(ch));
          }
        }
        item(null, rest ? "Worst drawdown" : "Drawdown", pct(rest ? view.worst : view.dd[at]), (rest ? view.worst : view.dd[at]) < 0 ? "loss" : "");
        legend.append(Object.assign(document.createElement("span"), {textContent: rest ? `${view.full[0]} to ${view.full[view.full.length - 1]}` : view.full[at], className: "faint"}));
      };
      const hover = (_e, els) => readout(els.length ? els[0].index : null);
      eq.options.onHover = hover; dd.options.onHover = hover;
      // Hovering either chart moves the hairline on both.
      const sync = (i) => [eq, dd].forEach((c) => {
        c.setActiveElements(i === null ? [] : [{datasetIndex: 0, index: i}]);
        c.tooltip.setActiveElements(i === null ? [] : [{datasetIndex: 0, index: i}], {x: 0, y: 0});
        c.update("none");
      });
      [eqEl, ddEl].forEach((el) => {
        el.addEventListener("mouseenter", () => { hovering = true; });
        el.addEventListener("mouseleave", () => { hovering = false; readout(null); sync(null); });
      });
      eqEl.addEventListener("mousemove", () => { const a = eq.getActiveElements(); if (a.length) { dd.setActiveElements([{datasetIndex: 0, index: a[0].index}]); dd.tooltip.setActiveElements([{datasetIndex: 0, index: a[0].index}], {x: 0, y: 0}); dd.update("none"); } });
      ddEl.addEventListener("mousemove", () => { const a = dd.getActiveElements(); if (a.length) { eq.setActiveElements([{datasetIndex: 0, index: a[0].index}]); eq.tooltip.setActiveElements([{datasetIndex: 0, index: a[0].index}], {x: 0, y: 0}); eq.update("none"); } });

      let times = [];
      // Place each fill on the curve at the first point at or after it; fills before the first point are off the chart.
      const fillIdx = (side) => {
        const out = new Set();
        (cur.fills || []).filter((f) => f.side === side && Date.parse(f.t) >= times[0]).forEach((f) => {
          const ft = Date.parse(f.t);
          let i = times.findIndex((x) => x >= ft);
          if (i < 0) i = times.length - 1;
          out.add(i);
        });
        return out;
      };
      const rebase = (arr, from) => {
        const b = arr.slice(from).find((v) => v !== null && v !== undefined && v > 0);
        return arr.slice(from).map((v) => (v === null || v === undefined || !b ? null : (v / b - 1) * 100));
      };
      // en-GB writes September as "Sept"; trading screens use three letters throughout.
      const fmtDate = (t, o) => new Date(t).toLocaleString("en-GB", {...o, timeZone: "UTC"}).replace("Sept", "Sep");
      // Boundaries for the date axis, thinned so labels never touch at any width.
      const boundaries = (ts) => {
        const span = (ts[ts.length - 1] - ts[0]) / 864e5;
        const out = {};
        if (isIntra() && span < 8) {
          // A day reads in hours; a week in days, the date at each midnight.
          const every = span <= 0.5 ? 1 : span <= 1.5 ? 3 : 24;  // hours between labels
          let prev = null;
          ts.forEach((t, i) => {
            const k = Math.floor(t / (every * 36e5));
            if (prev !== null && k !== prev) out[i] = every === 24 ? fmtDate(t, {weekday: "short", day: "2-digit"}) : fmtDate(t, {hour: "2-digit", minute: "2-digit"});
            prev = k;
          });
        } else {
          const unit = span <= 45 ? "week" : span <= 420 ? "month" : "quarter";
          let prev = null, first = true;
          ts.forEach((t, i) => {
            const dt = new Date(t), y = dt.getUTCFullYear(), m = dt.getUTCMonth();
            const k = unit === "week" ? Math.floor((t / 864e5 + 3) / 7) : unit === "month" ? y * 12 + m : y * 4 + Math.floor(m / 3);
            if (prev !== null && k !== prev) {
              out[i] = unit === "week" ? fmtDate(t, {day: "2-digit", month: "short"})
                : first || m === 0 ? fmtDate(t, {month: "short", year: "numeric"}) : fmtDate(t, {month: "short"});
              first = false;
            }
            prev = k;
          });
        }
        const keys = Object.keys(out).map(Number);
        // Measured from the box, not the last drawn chart area, which is stale while the layout settles.
        const room = Math.max(1, Math.floor(((ddEl.parentElement.clientWidth || 600) - 64) / 72));
        const stride = Math.ceil(keys.length / room);
        if (stride > 1) keys.forEach((k, n) => { if (n % stride) delete out[k]; });
        return out;
      };
      const ddSteps = [-1, -2, -4, -6, -8, -10, -20, -30, -40, -60, -80, -100];
      // The range on screen: a number of daily points (0 for all), or days of minute-level marks.
      let range = {n: 0};
      const show = () => {
        const from = range.n ? Math.max(0, cur.t.length - range.n) : 0;
        times = cur.t.map((x) => Date.parse(x));
        const buys = fillIdx("BUY"), sells = fillIdx("SELL");
        const ts = times.slice(from);
        const full = ts.map((t) => fmtDate(t, isIntra() ? {day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"} : {day: "2-digit", month: "short", year: "numeric"}));
        const eqv = cur.equity.slice(from), ret = rebase(cur.equity, from), bench = rebase(cur.benchmark, from);
        view = {full, eq: eqv, ret, bench, prior: from > 0 ? cur.equity[from - 1] : (cur.prior ?? null), dd: cur.drawdown.slice(from).map((x) => -x * 100)};
        // The whole run's worst comes from every mark, as the tables show it; the curve is daily or
        // thinned and can miss a fall that recovered between its points. A shorter range reads the curve.
        view.worst = cur === d && from === 0 && d.worst !== undefined ? -d.worst * 100 : Math.min(...view.dd);
        const top = compare ? ret : eqv;
        eq.data.labels = full; eq.data.datasets[0].data = top;
        eq.data.datasets[1].data = compare ? bench : [];
        eq.data.datasets[0].fill = compare ? "origin" : "start";
        eq.data.datasets[2].data = top.map((v, i) => (buys.has(i + from) ? v : null));
        eq.data.datasets[3].data = top.map((v, i) => (sells.has(i + from) ? v : null));
        const vals = (compare ? [...ret, ...bench] : eqv).filter((v) => v !== null && v !== undefined);
        const lo = Math.min(...vals), hi = Math.max(...vals);
        // Compare mode keeps 0% in view; money mode fits the equity line, padded so a flat line isn't a wall.
        const r = compare ? niceRange(Math.min(0, lo), Math.max(0, hi)) : niceRange(lo, hi, Math.max(hi * 0.0005, 0.5));
        step = r.step;
        Object.assign(eq.options.scales.y, {min: r.min, max: r.max}); eq.options.scales.y.ticks.stepSize = r.step;
        // Drawdown axis fits the worst point in view, on round steps: 0, half way, the floor.
        const worst = Math.min(...view.dd);
        const floor = ddSteps.find((s) => s <= worst * 1.1) ?? -100;
        Object.assign(dd.options.scales.y, {min: floor}); dd.options.scales.y.ticks.stepSize = -floor / 2;
        dd.data.labels = full; dd.data.datasets[0].data = view.dd;
        tickAt = boundaries(ts);
        eq.update(); dd.update();
        if (!hovering) readout(null);
      };
      const live = typeof url === "string";
      const withDays = (days) => url + (url.includes("?") ? "&" : "?") + "days=" + days;
      // Fetches what the current range needs. Live charts call this again every minute, so today's
      // move shows as it happens; a backtest's embedded result never changes.
      const load = () => {
        if (!live) { cur = d; return Promise.resolve(); }
        return fetch(range.days ? withDays(range.days) : url).then((r) => r.json()).then((x) => {
          if (range.days) cur = x; else { d = x; cur = x; }
        });
      };
      const buttons = scope.querySelectorAll("[data-range], [data-days]");
      buttons.forEach((b) => b.addEventListener("click", () => {
        buttons.forEach((o) => o.setAttribute("aria-pressed", o === b ? "true" : "false"));
        range = b.dataset.days ? {days: parseInt(b.dataset.days, 10), n: 0} : {n: parseInt(b.dataset.range, 10)};
        if (!range.days) cur = d;
        (range.days ? load() : Promise.resolve()).then(() => { if (cur.t.length) show(); });
      }));
      scope.querySelectorAll("[data-compare]").forEach((b) => {
        b.setAttribute("aria-pressed", String(compare));
        b.addEventListener("click", () => { compare = !compare; b.setAttribute("aria-pressed", String(compare)); show(); });
      });
      if (live) setInterval(() => {
        if (document.hidden || hovering) return;
        load().then(() => { if (cur.t.length) show(); }).catch(() => {});  // the strip already says when updates stop
      }, 60000);
      // Charts drawn inside a hidden tab get their width when it opens; recount the date labels then.
      new ResizeObserver(() => { if (view) { tickAt = boundaries(times.slice(range.n ? Math.max(0, cur.t.length - range.n) : 0)); dd.update("none"); } }).observe(ddEl.parentElement);
      show();
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
      timeScale: {borderVisible: false, rightOffset: 6, fixLeftEdge: true},
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
      items.push(...exitFields(form)());
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

  // Exits: one kind of stop and one kind of target, with only the chosen one's inputs live, and a plan
  // line pricing it in R with the costs the backtest and paper charge (the form's data-costs).
  // Returns the plain-English exit sentences for the new-sleeve summary.
  function exitFields(form) {
    const box = form && form.querySelector(".exit-fields");
    if (!box) return () => [];
    if (box.exitDescribe) return box.exitDescribe;
    const costs = box.dataset.costs ? JSON.parse(box.dataset.costs) : null;
    const line = box.querySelector(".exit-plan");
    const kind = {stop: box.querySelector("[data-kind=stop]"), tp: box.querySelector("[data-kind=tp]")};
    const num = (n) => { const i = form.elements[n]; return i && !i.disabled && i.value !== "" ? Number(i.value) : null; };
    const p2 = (x) => `${(x * 100).toFixed(2)}%`;
    const rr = (x) => `${x >= 0 ? "+" : ""}${x.toFixed(2)}R`;
    const sync = () => {
      for (const k of ["stop", "tp"]) {
        box.querySelectorAll(`[data-${k}]`).forEach((f) => {
          const on = f.dataset[k] === kind[k].value;
          f.hidden = !on;
          f.querySelectorAll("input").forEach((i) => { i.disabled = !on; });
        });
      }
      box.querySelector("[data-needs-stop]").hidden = !!kind.stop.value;
      const rOpt = kind.tp.querySelector("option[value=r]");
      rOpt.disabled = !kind.stop.value;  // a target in R needs a stop
      if (rOpt.disabled && kind.tp.value === "r") { kind.tp.value = ""; sync(); return; }
      if (!line || !costs) return;
      const pair = ((form.elements.instrument && form.elements.instrument.value) || "").toUpperCase();
      const half = pair in costs.spreads ? costs.spreads[pair] : costs.default_spread;
      const leg = costs.taker + half;  // each way: the taker fee and half the spread
      const s = num("stop_loss_pct") !== null ? num("stop_loss_pct") / 100 : null;
      const r = num("take_profit_r");
      // A target in R pays R times the stop-out's loss, both after costs (strategies/base.py r_target).
      const rTarget = (k, stop) => (k * (stop * (1 - leg) + 2 * leg) + 2 * leg) / (1 - leg);
      const t = num("take_profit_pct") !== null ? num("take_profit_pct") / 100 : null;
      const parts = [];
      let warn = false;
      const loss = s !== null ? s + leg + (1 - s) * leg : null;  // 1R: what a stop-out loses, costs included
      if (loss !== null) parts.push(`A stop-out loses 1R = ${p2(loss)} of the position: the ${p2(s)} stop plus ${p2(loss - s)} in fees and spread.`);
      else if (kind.stop.value === "atr" || kind.stop.value === "swing") parts.push("The stop is set from the market at each entry, so 1R differs from trade to trade; each trade's R shows in the backtest.");
      if (t !== null) {
        const trip = leg + (1 + t) * leg, gain = t - trip;
        if (gain <= 0) { warn = true; parts.push(`The ${p2(t)} target doesn't cover the ${p2(trip)} round trip, so every target hit would lose money. It can't be saved.`); }
        else if (loss !== null) {
          const R = gain / loss;
          parts.push(`The target makes ${rr(R)} after costs (${p2(t)} less a ${p2(trip)} round trip), ${(t / s).toFixed(1)}:1 before costs.`);
          if (R < 0.25) { warn = true; parts.push(`That is almost nothing for the risk: costs alone take ${p2(trip)}.`); }
        } else parts.push(`The target makes ${p2(gain)} after a ${p2(trip)} round trip.`);
      } else if (r !== null) {
        if (s !== null) parts.push(`The target makes ${rr(r)} after costs: ${p2(rTarget(r, s))} above the entry.`);
        else parts.push(`The target makes ${rr(r)} after costs, so it sits further out than ${r} stop distances: on a 3% stop, ${p2(rTarget(r, 0.03))} above the entry.`);
        if (r < 0.25) { warn = true; parts.push("That is almost nothing for the risk."); }
      }
      line.hidden = !parts.length;
      line.classList.toggle("warn", warn);
      line.textContent = parts.join(" ");
    };
    kind.stop.addEventListener("change", sync); kind.tp.addEventListener("change", sync);
    form.addEventListener("input", sync);
    sync();
    box.exitDescribe = () => {
      sync();  // the summary may be asked before this form's own input listener has run
      const out = [], r = num("take_profit_r"), atr = num("stop_atr"), swing = num("stop_swing_bars");
      const stop = num("stop_loss_pct") !== null ? `${num("stop_loss_pct")}% below entry`
        : atr !== null ? `${atr} average true ranges (over ${num("atr_bars") || 14} bars) below entry`
        : swing !== null ? `at the lowest low of the last ${swing} bars` : null;
      const tp = num("take_profit_pct") !== null ? `${num("take_profit_pct")}% above entry`
        : r !== null ? `a target that makes ${r}R after costs` : null;
      if (stop || tp) out.push(`Exit any trade ${[stop, tp].filter(Boolean).join(" or ")}.`);
      if (line && !line.hidden) out.push(line.textContent);
      const rpt = num("risk_per_trade_pct");
      if (rpt) out.push(stop ? `Size each trade to lose about ${rpt}% of capital if the stop is hit, fees and spread included.` : "Risk per trade needs a stop-loss; add one or clear it.");
      return out;
    };
    return box.exitDescribe;
  }

  // Order type: the wait only matters for maker-first orders, so it shows only then.
  function orderFields(formId) {
    const form = document.getElementById(formId);
    exitFields(form);
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


  // Tabs on a strategy page: one panel at a time, chosen by the URL hash so links and the back
  // button work. A hash that points inside a panel (an activity filter link) opens that panel.
  // Without JavaScript every panel shows, one after another.
  function tabs() {
    const bar = document.querySelector("[data-tabs]");
    if (!bar) return;
    const links = [...bar.querySelectorAll("[data-tab]")];
    const panels = [...document.querySelectorAll("[data-panel]")];
    const show = () => {
      const id = decodeURIComponent(location.hash.slice(1));
      // Panels' ids differ from their tab names, so the browser never jumps past the header to one.
      const named = id && document.querySelector(`[data-panel="${CSS.escape(id)}"]`);
      const target = named || (id && document.getElementById(id));
      const panel = named || (target && target.closest("[data-panel]")) || panels[0];
      panels.forEach((p) => { p.hidden = p !== panel; });
      links.forEach((a) => {
        const on = a.dataset.tab === panel.dataset.panel;
        a.setAttribute("aria-selected", String(on));
        if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
      });
      // Charts sized while hidden need a nudge once their panel is visible.
      window.dispatchEvent(new Event("resize"));
      if (target && target !== panel) target.scrollIntoView({block: "start"});
    };

    bar.addEventListener("click", (e) => {
      const a = e.target.closest("[data-tab]");
      if (!a) return;
      e.preventDefault();
      history.replaceState(null, "", "#" + a.dataset.tab);
      show();
    });
    window.addEventListener("hashchange", show);
    show();
  }

  return {sortable, tabs, sortBy, dialogs, whys, strategyPicker, priceChart, sleeveForm, orderFields, bookCharts: (url) => pair(url, "eq", "dd", ["Book", "Buy-and-hold"]), pair};
})();
