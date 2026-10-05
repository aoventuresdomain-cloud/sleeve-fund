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
          // The whole history's change is from the starting capital, which is never a point on it (M12-F1).
          const first = view.whole && Number.isFinite(d.start) ? d.start : view.eq[0], last = view.eq[at], ch = last - first;
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
        view.whole = cur === d && from === 0;
        view.worst = view.whole && d.worst !== undefined ? -d.worst * 100 : Math.min(...view.dd);
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

  // Indicators, worked out in the browser from the candles on screen. Each instance has its own
  // settings and colour; the list is remembered per browser. Oscillators get a strip under the price.
  const nulls = (n) => Array(n).fill(null);
  // Runs fn on the part of xs after its leading gaps, so indicators can be stacked (an EMA of a MACD).
  const tail = (fn) => (xs, ...a) => { const f = xs.findIndex((v) => v != null); return f < 0 ? nulls(xs.length) : nulls(f).concat(fn(xs.slice(f), ...a)); };
  const sma = tail((xs, n) => { let sum = 0; return xs.map((x, i) => { sum += x; if (i >= n) sum -= xs[i - n]; return i >= n - 1 ? sum / n : null; }); });
  const ema = tail((xs, n) => {
    const k = 2 / (n + 1), seed = sma(xs, n);
    let prev = null;
    return xs.map((x, i) => (prev = i < n - 1 ? null : prev == null ? seed[i] : x * k + prev * (1 - k)));
  });
  const wma = tail((xs, n) => xs.map((_, i) => {
    if (i < n - 1) return null;
    let s = 0; for (let j = 0; j < n; j++) s += xs[i - j] * (n - j);
    return s / (n * (n + 1) / 2);
  }));
  const rolling = (xs, n, fn) => xs.map((_, i) => (i < n - 1 ? null : fn(xs.slice(i - n + 1, i + 1))));
  const stdev = (xs, n, mid) => xs.map((_, i) => {
    if (mid[i] == null) return null;
    let v = 0; for (let j = i - n + 1; j <= i; j++) v += (xs[j] - mid[i]) ** 2;
    return Math.sqrt(v / n);
  });
  const wilder = (xs, n) => {  // Wilder's smoothing, as RSI and ATR use it
    const out = nulls(xs.length);
    let avg = 0;
    xs.forEach((x, i) => { if (i < n) { avg += x / n; if (i === n - 1) out[i] = avg; } else out[i] = avg = (avg * (n - 1) + x) / n; });
    return out;
  };
  const rsi = (c, n) => {
    const d = c.map((x, i) => (i ? x - c[i - 1] : 0)).slice(1);
    const up = wilder(d.map((x) => Math.max(x, 0)), n), dn = wilder(d.map((x) => Math.max(-x, 0)), n);
    return [null].concat(up.map((u, i) => (u == null ? null : dn[i] === 0 ? (u > 0 ? 100 : 50) : 100 - 100 / (1 + u / dn[i]))));  // as sleeve_fund/strategies/indicators.py Rsi
  };
  const trueRange = (k) => k.close.map((_, i) => (i ? Math.max(k.high[i] - k.low[i], Math.abs(k.high[i] - k.close[i - 1]), Math.abs(k.low[i] - k.close[i - 1])) : k.high[i] - k.low[i]));
  const lo = (xs) => Math.min(...xs), hi = (xs) => Math.max(...xs);
  const minus = (a, b) => a.map((v, i) => (v == null || b[i] == null ? null : v - b[i]));
  const plus = (a, b, k = 1) => a.map((v, i) => (v == null || b[i] == null ? null : v + k * b[i]));
  // pane: false draws on the price; otherwise a strip below. lines() returns [{vals, style?, alpha?, hist?}].
  const LIB = {
    sma: {name: "Simple moving average", short: "SMA", params: [["Length", 20]], lines: (k, [n]) => [{vals: sma(k.close, n)}]},
    ema: {name: "Exponential moving average", short: "EMA", params: [["Length", 50]], lines: (k, [n]) => [{vals: ema(k.close, n)}]},
    wma: {name: "Weighted moving average", short: "WMA", params: [["Length", 20]], lines: (k, [n]) => [{vals: wma(k.close, n)}]},
    vwap: {name: "Rolling VWAP", short: "VWAP", params: [["Length", 20]], lines: (k, [n]) => {
      const tp = k.close.map((c, i) => (c + k.high[i] + k.low[i]) / 3);
      return [{vals: k.close.map((_, i) => {
        if (i < n - 1) return null;
        let pv = 0, v = 0; for (let j = i - n + 1; j <= i; j++) { pv += tp[j] * k.volume[j]; v += k.volume[j]; }
        return v > 0 ? pv / v : null;
      })}];
    }},
    bb: {name: "Bollinger bands", short: "BB", params: [["Length", 20], ["Width (σ)", 2]], lines: (k, [n, w]) => {
      const mid = sma(k.close, n), sd = stdev(k.close, n, mid);
      return [{vals: plus(mid, sd, w), alpha: 0.85}, {vals: mid, style: 2, alpha: 0.6}, {vals: plus(mid, sd, -w), alpha: 0.85}];
    }},
    kc: {name: "Keltner channels", short: "KC", params: [["Length", 20], ["Width (ATR)", 2]], lines: (k, [n, w]) => {
      const mid = ema(k.close, n), atr = wilder(trueRange(k), n);
      return [{vals: plus(mid, atr, w), alpha: 0.85}, {vals: mid, style: 2, alpha: 0.6}, {vals: plus(mid, atr, -w), alpha: 0.85}];
    }},
    dc: {name: "Donchian channels", short: "DC", params: [["Length", 20]], lines: (k, [n]) => {
      const up = rolling(k.high, n, hi), dn = rolling(k.low, n, lo);
      return [{vals: up, alpha: 0.85}, {vals: up.map((u, i) => (u == null ? null : (u + dn[i]) / 2)), style: 2, alpha: 0.6}, {vals: dn, alpha: 0.85}];
    }},
    rsi: {name: "Relative strength index", short: "RSI", pane: {min: 0, max: 100, guides: [30, 70], digits: 0}, params: [["Length", 14]], lines: (k, [n]) => [{vals: rsi(k.close, n)}]},
    stoch: {name: "Stochastic", short: "Stoch", pane: {min: 0, max: 100, guides: [20, 80], digits: 0}, params: [["%K", 14], ["%D", 3]], lines: (k, [n, d]) => {
      const kk = k.close.map((c, i) => { if (i < n - 1) return null; const l = lo(k.low.slice(i - n + 1, i + 1)), h = hi(k.high.slice(i - n + 1, i + 1)); return h > l ? (100 * (c - l)) / (h - l) : 50; });
      return [{vals: kk}, {vals: sma(kk, d), style: 2, alpha: 0.7}];
    }},
    macd: {name: "MACD", short: "MACD", pane: {guides: [0]}, params: [["Fast", 12], ["Slow", 26], ["Signal", 9]], lines: (k, [f, sl, sg]) => {
      const m = minus(ema(k.close, f), ema(k.close, sl)), sig = ema(m, sg);
      return [{vals: minus(m, sig), hist: true}, {vals: m}, {vals: sig, style: 2, alpha: 0.7}];
    }},
    atr: {name: "Average true range", short: "ATR", pane: {}, params: [["Length", 14]], lines: (k, [n]) => [{vals: wilder(trueRange(k), n)}]},
    roc: {name: "Rate of change %", short: "ROC", pane: {guides: [0], digits: 2}, params: [["Length", 10]], lines: (k, [n]) => [{vals: k.close.map((c, i) => (i >= n ? (c / k.close[i - n] - 1) * 100 : null))}]},
  };
  const IND_KEY = "pc-indicators-v2", IND_MAX = 20;
  const PALETTE = ["--ind-1", "--ind-2", "--ind-3", "--ind-4", "--ind-5", "--ind-6", "--ind-7", "--ind-8"];
  const loadInd = () => {
    try {
      const v = JSON.parse(localStorage.getItem(IND_KEY) || "[]");
      return Array.isArray(v) ? v.filter((x) => x && LIB[x.type] && Array.isArray(x.p)).slice(0, IND_MAX) : [];
    } catch (e) { return []; }
  };
  const saveInd = (v) => { try { localStorage.setItem(IND_KEY, JSON.stringify(v)); } catch (e) { /* private window: kept for this visit */ } };

  // TradingView-style price chart: candles, volume, a marker on every fill, and entry/stop/target
  // lines. Clicking a marker shows the reason journaled when the order was sent.
  // src is the data itself, or an endpoint that takes ?interval= and ?pair= (another instrument, for comparison).
  function priceChart(boxId, src) {
    const box = document.getElementById(boxId);
    if (!box || !window.LightweightCharts) return;
    const $ = (sel) => box.querySelector(sel);
    const accent = css("--accent");
    const fmt = (v) => (Math.abs(v) >= 100 ? v.toLocaleString("en-GB", {minimumFractionDigits: 2, maximumFractionDigits: 2}) : v.toPrecision(5));
    const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
    const base = {
      autoSize: true,
      layout: {background: {type: "solid", color: css("--panel")}, textColor: css("--muted"), fontSize: 11, fontFamily: getComputedStyle(document.body).fontFamily, attributionLogo: true},
      grid: {vertLines: {visible: false}, horzLines: {color: css("--line")}},
      timeScale: {borderVisible: false, rightOffset: 6, fixLeftEdge: true},
      crosshair: {mode: 0, vertLine: {color: css("--line-strong"), labelBackgroundColor: css("--raised")}, horzLine: {color: css("--line-strong"), labelBackgroundColor: css("--raised")}},
    };
    // The axis keeps counting down through the volume band; a price at or below zero there is no price, so leave it blank.
    const chart = LightweightCharts.createChart($(".pc-canvas"), {...base,
      localization: {priceFormatter: (v) => (v > 0 ? fmt(v) : "")},
      rightPriceScale: {borderVisible: false, minimumWidth: 76, scaleMargins: {top: 0.08, bottom: 0.08}},
    });
    const candles = chart.addCandlestickSeries({upColor: css("--gain"), downColor: css("--loss"), borderVisible: false, wickUpColor: css("--gain"), wickDownColor: css("--loss")});
    // Candles built from the sleeve's own marks have no range inside the bar, so they draw as a line instead.
    const area = chart.addAreaSeries({lineColor: accent, topColor: rgba(accent, 0.22), bottomColor: rgba(accent, 0), lineWidth: 2, visible: false});
    const vol = chart.addHistogramSeries({priceScaleId: "vol", color: rgba(css("--muted"), 0.3), priceFormat: {type: "volume"}, lastValueVisible: false, priceLineVisible: false});
    chart.priceScale("vol").applyOptions({scaleMargins: {top: 0.86, bottom: 0}});
    let main = candles;
    let lines = [], data = null;

    // Indicators: config is the saved list; built holds each one's series and, for oscillators, its strip.
    let config = loadInd(), built = [];
    let syncing = false, mirroring = false;
    const quiet = {priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false};
    const colorOf = (c) => (c.color && /^#[0-9a-f]{6}$/i.test(c.color) ? c.color : css(PALETTE[0]));
    const labelOf = (c) => `${LIB[c.type].short} ${c.p.join(", ")}`;
    const strips = () => built.filter((b) => b.chart !== chart);
    const teardown = () => {
      built.forEach((b) => { if (b.chart === chart) b.series.forEach((s) => chart.removeSeries(s)); else { b.chart.remove(); b.box.remove(); } });
      built = [];
    };
    const build = () => {
      teardown();
      config.forEach((c) => {
        const def = LIB[c.type], color = colorOf(c);
        let target = chart, wrap = null, legend = null;
        if (def.pane) {
          wrap = el("div", "pc-sub"); legend = el("div", "pc-sub-legend");
          wrap.append(legend); $(".pc-subs").append(wrap);
          const d = def.pane.digits ?? 2;
          target = LightweightCharts.createChart(wrap, {...base, layout: {...base.layout, attributionLogo: false},
            localization: {priceFormatter: (v) => v.toFixed(d)},
            rightPriceScale: {borderVisible: false, minimumWidth: 76, scaleMargins: {top: 0.12, bottom: 0.08}}});
        }
        const shape = def.lines({close: [1], high: [1], low: [1], volume: [1]}, c.p);  // how many lines, and their style
        const series = shape.map((ln) => (ln.hist
          ? target.addHistogramSeries({...quiet, priceFormat: {type: "price", precision: def.pane?.digits ?? 2, minMove: 0.01}})
          : target.addLineSeries({...quiet, color: rgba(color, ln.alpha ?? 1), lineWidth: def.pane ? 1.5 : 1.5, lineStyle: ln.style || 0,
              ...(def.pane && def.pane.min != null ? {autoscaleInfoProvider: () => ({priceRange: {minValue: def.pane.min, maxValue: def.pane.max}})} : {})})));
        if (def.pane) (def.pane.guides || []).forEach((g) => series[series.length - 1].createPriceLine({price: g, color: css("--line-strong"), lineWidth: 1, lineStyle: 2, axisLabelVisible: false}));
        built.push({c, def, color, series, chart: target, box: wrap, legend, vals: []});
      });
      // Every strip follows the price chart's scroll and zoom, and the other way round.
      strips().forEach((b) => {
        b.chart.timeScale().subscribeVisibleLogicalRangeChange((r) => {
          if (!r || syncing) return;
          syncing = true; chart.timeScale().setVisibleLogicalRange(r); strips().forEach((o) => o !== b && o.chart.timeScale().setVisibleLogicalRange(r)); syncing = false;
        });
        b.chart.subscribeCrosshairMove((p) => { if (!mirroring) mirror(p.time, b); });
      });
      const last = strips().length;
      chart.applyOptions({timeScale: {visible: !last}});
      strips().forEach((b, i) => b.chart.applyOptions({timeScale: {visible: i === last - 1}}));
      const badge = $(".pc-ind-btn .n");
      badge.textContent = config.length; badge.hidden = !config.length;
      if (data) fill();
    };
    // Drawn while its tab was hidden, the chart fitted the candles to a zero-width box and kept that bar
    // spacing once shown, crammed into the left edge. Fit again when it first gets a real width.
    let wasHidden = $(".pc-canvas").clientWidth < 50;
    new ResizeObserver(([e]) => {
      const hidden = e.contentRect.width < 50;
      if (wasHidden && !hidden && data) chart.timeScale().fitContent();
      wasHidden = hidden;
    }).observe($(".pc-canvas"));
    chart.timeScale().subscribeVisibleLogicalRangeChange((r) => {
      if (!r || syncing) return;
      syncing = true; strips().forEach((b) => b.chart.timeScale().setVisibleLogicalRange(r)); syncing = false;
    });
    const fill = () => {
      const t = data.candles.map((c) => c.time);
      const k = {close: data.candles.map((c) => c.close), high: data.candles.map((c) => c.high), low: data.candles.map((c) => c.low),
        volume: data.candles.map((c, i) => (data.volume[i] ? data.volume[i].value : 0))};
      built.forEach((b) => {
        const out = b.def.lines(k, b.c.p);
        b.vals = out.map((ln) => ln.vals);
        out.forEach((ln, j) => {
          // Strips get a point per candle, blank where the indicator isn't defined yet, so all charts count bars alike.
          const pts = b.chart === chart ? ln.vals.map((v, i) => (v == null ? null : {time: t[i], value: v})).filter(Boolean)
            : ln.vals.map((v, i) => (v == null ? {time: t[i]} : ln.hist ? {time: t[i], value: v, color: rgba(v >= 0 ? css("--gain") : css("--loss"), 0.5)} : {time: t[i], value: v}));
          b.series[j].setData(pts);
        });
        if (b.chart !== chart) b.chart.applyOptions({timeScale: {timeVisible: data.interval < 1440, secondsVisible: false}});
      });
      const r = chart.timeScale().getVisibleLogicalRange();
      if (r) strips().forEach((b) => b.chart.timeScale().setVisibleLogicalRange(r));
      legends(-1);
    };
    // Legends: overlays in the price legend, each strip's values in its own corner.
    const valueText = (b, i) => {
      const d = b.def.pane?.digits ?? null;
      const at = b.vals.map((v) => (i >= 0 ? v[i] : v[v.length - 1]));
      if (b.def.lines.length && b.series[0] && b.series[0].seriesType() === "Histogram") at.push(at.shift());  // MACD reads line, signal, histogram
      const shown = at.filter((v) => v != null);
      if (!shown.length) return "";
      const f = (v) => (d != null ? v.toFixed(d) : b.def.pane ? +v.toPrecision(4) + "" : fmt(v));
      return b.def.pane ? shown.map(f).join("  ") : shown.length === 3 ? `${f(shown[2])} – ${f(shown[0])}` : f(shown[0]);
    };
    const tag = (b, i) => {
      const sp = el("span"), sw = el("i"); sw.style.background = b.color;
      sp.append(sw, labelOf(b.c) + "  ", el("b", null, valueText(b, i)));
      return sp;
    };
    let barText = "";
    const legends = (i) => {
      $(".pc-legend").replaceChildren(...(barText ? [el("span", null, barText)] : []), ...built.filter((b) => b.chart === chart).map((b) => tag(b, i)));
      strips().forEach((b) => b.legend.replaceChildren(tag(b, i)));
    };
    const indexOf = (time) => (time && data ? data.candles.findIndex((c) => c.time === time) : -1);
    // Moving over any strip or the price moves the crosshair on all of them.
    const mirror = (time, from) => {
      mirroring = true;
      const i = indexOf(time);
      if (from) { const c = data && data.candles[i]; if (c) chart.setCrosshairPosition(c.close, time, main); else chart.clearCrosshairPosition(); }
      strips().forEach((b) => {
        if (b === from) return;
        const v = i >= 0 ? b.vals.map((x) => x[i]).find((x) => x != null) : null;
        const s = b.series[b.vals.findIndex((x) => i >= 0 && x[i] != null)];
        if (v != null && s) b.chart.setCrosshairPosition(v, time, s); else b.chart.clearCrosshairPosition();
      });
      mirroring = false;
    };

    // The indicator menu: what's on, each with its settings, colour and a remove button; then a list to add more.
    const indBtn = $(".pc-ind-btn"), menu = $(".pc-ind-menu");
    const commit = () => { saveInd(config); build(); drawMenu(); };
    const drawMenu = () => {
      const rows = config.map((c, idx) => {
        const def = LIB[c.type], row = el("div", "row");
        const col = el("input"); col.type = "color"; col.value = colorOf(c); col.setAttribute("aria-label", `${def.name} colour`);
        col.addEventListener("change", () => { c.color = col.value; commit(); });
        row.append(col, el("span", "nm", def.short));
        row.querySelector(".nm").title = def.name;
        def.params.forEach(([label], j) => {
          const n = el("input"); n.type = "number"; n.min = label.includes("Width") ? 0.1 : 1; n.max = 500; n.step = label.includes("Width") ? 0.1 : 1;
          n.value = c.p[j]; n.title = label; n.setAttribute("aria-label", `${def.name} ${label}`);
          n.addEventListener("change", () => { const v = +n.value; if (v > 0 && v <= 500) { c.p[j] = label.includes("Width") ? v : Math.round(v); commit(); } else n.value = c.p[j]; });
          row.append(n);
        });
        const x = el("button", "x", "×"); x.type = "button"; x.setAttribute("aria-label", `Remove ${labelOf(c)}`);
        x.addEventListener("click", () => { config.splice(idx, 1); commit(); });
        row.append(x);
        return row;
      });
      const add = el("select");
      add.setAttribute("aria-label", "Add an indicator");
      add.append(new Option(config.length >= IND_MAX ? `Up to ${IND_MAX} indicators` : "+ Add indicator", ""));
      [["On the price", false], ["Below the price", true]].forEach(([label, pane]) => {
        const g = el("optgroup"); g.label = label;
        Object.entries(LIB).filter(([, d]) => !!d.pane === pane).forEach(([key, d]) => g.append(new Option(d.name, key)));
        add.append(g);
      });
      add.disabled = config.length >= IND_MAX;
      add.addEventListener("change", () => {
        const def = LIB[add.value];
        if (!def) return;
        config.push({type: add.value, p: def.params.map(([, v]) => v), color: css(PALETTE[config.length % PALETTE.length])});
        commit();
      });
      menu.replaceChildren(...(rows.length ? rows : [el("div", "none", "No indicators yet.")]), add,
        el("p", null, "Lengths count candles of the interval shown. Add the same indicator more than once with different settings."));
    };
    drawMenu();

    // Pop-overs (instrument list, indicator menu): one open at a time, closed by Escape or a click elsewhere.
    const pops = [];
    const popover = (btn, pop, onOpen) => {
      const set = (open) => { pop.hidden = !open; btn.setAttribute("aria-expanded", String(open)); if (open && onOpen) onOpen(); };
      btn.addEventListener("click", () => { const open = pop.hidden; pops.forEach((p) => p.set(false)); set(open); });
      pops.push({set, btn, pop});
      return set;
    };
    popover(indBtn, menu);
    document.addEventListener("click", (e) => pops.forEach((p) => { if (!p.pop.hidden && !p.pop.contains(e.target) && !p.btn.contains(e.target)) p.set(false); }));
    box.addEventListener("keydown", (e) => { if (e.key === "Escape") pops.forEach((p) => { if (!p.pop.hidden) { p.set(false); p.btn.focus(); } }); });

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
    let current = "", pair = "";
    const marksNote = $(".pc-source").textContent;
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
      vol.setData(hasVol ? d.volume : []);
      // Keep the candles clear of the volume band when there is one.
      chart.priceScale("right").applyOptions({scaleMargins: {top: 0.08, bottom: hasVol ? 0.18 : 0.06}});
      main.setMarkers(d.markers.map((m) => ({...m, color: m.position === "belowBar" ? css("--gain") : css("--loss")})));
      lines.forEach((l) => main.removePriceLine(l));
      lines = d.lines.map((l) => main.createPriceLine({price: l.price, title: l.title, lineWidth: 1, lineStyle: 2, axisLabelVisible: true,
        color: l.kind === "stop" ? css("--loss") : l.kind === "target" ? css("--gain") : accent}));
      if (!keepView) chart.timeScale().fitContent();
      fill();
      $(".pc-source").hidden = d.source !== "marks" && !d.note;
      $(".pc-source").textContent = d.note || marksNote;
      $(".pc-empty").hidden = d.candles.length > 0;
      if (!keepView) showNote([]);
      // Another instrument than the strategy's own: say so, and offer the way back.
      const other = d.home && d.pair && d.pair !== d.home;
      if (d.pair) { $(".pc-title").textContent = d.pair; if (symBtn) symBtn.setAttribute("aria-label", `Instrument: ${d.pair}. Choose another`); }
      const home = $(".pc-home"), otherNote = $(".pc-other");
      if (home) { home.hidden = !other; home.textContent = other ? `Back to ${d.home}` : ""; }
      if (otherNote) { otherNote.hidden = !other; otherNote.textContent = other ? `Comparison only. This strategy trades ${d.home}, so its trades and exits aren't drawn here.` : ""; }
      if (d.pairs) ownPairs = d.pairs;
      const tabs = $(".pc-intervals");
      if (tabs && d.intervals) {
        tabs.replaceChildren(...d.intervals.map((k) => {
          const b = el("button", null, k); b.type = "button"; b.setAttribute("aria-pressed", String(k === d.chosen));
          b.addEventListener("click", () => load(k)); return b;
        }));
      }
    };
    const load = (interval, keepView = false) => {
      if (typeof src !== "string") { render(src); return Promise.resolve(); }
      current = interval;
      const q = new URLSearchParams();
      if (interval) q.set("interval", interval);
      if (pair) q.set("pair", pair);
      const asked = pair;
      return fetch(src + (q.toString() ? `?${q}` : ""), {cache: "no-store"})
        .then((r) => r.json().then((d) => { if (!r.ok) throw new Error(d.detail || "Couldn't load that instrument"); return d; }))
        .then((d) => { if (asked === pair) render(d, keepView); });
    };

    // The instrument dropdown: this strategy's instrument, the book's, then everything the venue lists, with a search box.
    const symBtn = $(".pc-sym-btn");
    let ownPairs = [], allPairs = null;
    if (symBtn && typeof src === "string") {
      const pop = $(".pc-sym-pop"), search = pop.querySelector("input"), list = pop.querySelector("ul");
      let active = 0;
      const pick = (v) => {
        setOpen(false);
        pair = data && v === data.home ? "" : v;
        load(current).catch((e) => { $(".pc-source").hidden = false; $(".pc-source").textContent = e.message; });
      };
      const drawList = () => {
        const q = search.value.trim().toUpperCase().replace(/[-_ ]/, "/");
        const match = (p) => !q || p.includes(q) || p.replace("/", "").includes(q.replace("/", ""));
        const shown = data ? data.pair : "";
        const items = [];
        const group = (label, xs) => {
          const hits = xs.filter(match);
          if (!hits.length) return;
          const g = el("li", "grp", label); g.setAttribute("role", "presentation"); items.push(g);
          hits.forEach((p) => {
            const li = el("li", "opt", p); li.setAttribute("role", "option"); li.setAttribute("aria-selected", String(p === shown));
            if (data && p === data.home) li.append(el("small", null, "this strategy"));
            li.addEventListener("click", () => pick(p)); items.push(li);
          });
        };
        group("This strategy and the book", ownPairs);
        group(allPairs ? `All instruments (${allPairs.length})` : "Loading the venue's list…", (allPairs || []).filter((p) => !ownPairs.includes(p)));
        // Something typed that isn't in any list: still offer it, the venue decides.
        if (/^[A-Z0-9]{1,12}\/[A-Z0-9]{2,6}$/.test(q) && !ownPairs.includes(q) && !(allPairs || []).includes(q)) group("Other", [q]);
        list.replaceChildren(...items);
        active = 0; mark();
      };
      const opts = () => [...list.querySelectorAll("li.opt")];
      const mark = () => opts().forEach((li, i) => { li.classList.toggle("on", i === active); if (i === active) li.scrollIntoView({block: "nearest"}); });
      const setOpen = popover(symBtn, pop, () => {
        search.value = ""; drawList(); search.focus({preventScroll: true});
        if (!allPairs) fetch("/api/instruments", {cache: "no-store"}).then((r) => r.json()).then((d) => { allPairs = d.instruments || []; drawList(); }).catch(() => { allPairs = []; drawList(); });
      });
      search.addEventListener("input", drawList);
      search.addEventListener("keydown", (e) => {
        const o = opts();
        if (e.key === "ArrowDown") { active = Math.min(active + 1, o.length - 1); mark(); e.preventDefault(); }
        else if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); mark(); e.preventDefault(); }
        else if (e.key === "Enter") { e.preventDefault(); if (o[active]) pick(o[active].firstChild.textContent); }
      });
      $(".pc-home").addEventListener("click", () => { pair = ""; load(current); });
    }
    // A sleeve's chart follows the market: new candles and fills appear without a reload.
    if (typeof src === "string") setInterval(() => { if (!document.hidden) load(current, true).catch(() => {}); }, 30000);
    chart.subscribeCrosshairMove((p) => {
      const bar = p.seriesData && p.seriesData.get(main);
      barText = !bar ? "" : main === area ? `Price ${fmt(bar.value)}` : `O ${fmt(bar.open)}  H ${fmt(bar.high)}  L ${fmt(bar.low)}  C ${fmt(bar.close)}`;
      const i = bar ? indexOf(p.time) : -1;
      legends(i);
      if (!mirroring) mirror(p.time, null);
    });
    chart.subscribeClick((p) => {
      if (!data || !p.time) return;
      const ids = p.hoveredObjectId && data.notes[p.hoveredObjectId] ? [p.hoveredObjectId]
        : data.markers.filter((m) => m.time === p.time).map((m) => m.id);
      if (ids.length) showNote(ids);
    });
    build();
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
        (() => {
          const mk = form.elements.market ? form.elements.market.value : "spot";
          if (mk === "spot") return `Hold at most ${pct(Number(prof.cap))} of its capital in ${pair.split("/")[0] || "the instrument"}, long only.`;
          const shorts = form.elements.allow_short && form.elements.allow_short.checked;
          return `Trade the ${form.elements.market.selectedOptions[0].textContent.split(":")[0].toLowerCase()}, ${shorts ? "long and short" : "long only"}, with positions up to ${prof.lev}x its capital.`;
        })(),
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
      ["name", "reason", "warmup_bars", "tested_bar_spec", "from", "account", "demo_mirror"].forEach((k) => q.delete(k));
      q.set("run", "1");
      bt.href = `/backtest?${q}`;
    });
  }

  // A trade's exits priced after costs, as the strategy prices them (strategies/base.py loss_at_stop,
  // gain_at_target, r_target): each leg pays `leg` on its own notional, and a short (side -1) buys back
  // above its entry at the stop and below it at the target.
  const exitMath = {
    loss: (stop, leg, side) => stop + leg + (1 - side * stop) * leg,
    gain: (tp, leg, side) => tp - leg - (1 + side * tp) * leg,
    rTarget: (r, stop, leg, side) => (r * exitMath.loss(stop, leg, side) + 2 * leg) / (1 - side * leg),
  };

  // Exits: one kind of stop and one kind of target, with only the chosen one's inputs live, and a plan
  // line pricing it in R with the costs the backtest and paper charge (the form's data-costs).
  // Returns the plain-English exit sentences for the new-sleeve summary.
  function exitFields(form) {
    const box = form && form.querySelector(".exit-fields");
    if (!box) return () => [];
    if (box.exitDescribe) return box.exitDescribe;
    const costs = box.dataset.costs ? JSON.parse(box.dataset.costs) : null;
    const line = box.querySelector(".exit-plan");
    // The side the exits are for: the position held, else every side the strategy can take.
    const market = () => (form.elements.market ? form.elements.market.value : box.dataset.market) || "spot";
    const shorts = () => market() !== "spot" && (form.elements.allow_short ? form.elements.allow_short.checked : !!box.dataset.shorts);
    const heldSide = () => Number(box.dataset.side || 0);
    const ways = () => {
      const side = heldSide();
      if (side < 0) return {stop: "above", tp: "below", swing: "highest high", swing_short: "high"};
      if (!side && shorts()) return {stop: "below (a short: above)", tp: "above (a short: below)",
        swing: "lowest low (a short: highest high)", swing_short: "low (a short: high)"};
      return {stop: "below", tp: "above", swing: "lowest low", swing_short: "low"};
    };
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
      const w = ways();
      box.querySelectorAll("[data-exit-tpl]").forEach((el) => {
        el.textContent = el.dataset.exitTpl.replace(/\{(\w+)\}/g, (_, k) => w[k]);
      });
      if (!line || !costs) return;
      const pair = ((form.elements.instrument && form.elements.instrument.value) || "").toUpperCase();
      const mk = (costs.markets || {})[market()] || {};
      const half = "half_spread" in mk ? mk.half_spread : pair in costs.spreads ? costs.spreads[pair] : costs.default_spread;
      const leg = ("taker" in mk ? mk.taker : costs.taker) + half;  // each way: the taker fee and half the spread
      // Priced for the side held; with none, for a long (a short's differs by a fraction of the leg cost).
      const side = heldSide() < 0 ? -1 : 1, who = heldSide() < 0 ? "the short" : "the position";
      const s = num("stop_loss_pct") !== null ? num("stop_loss_pct") / 100 : null;
      const r = num("take_profit_r");
      // A target in R pays R times the stop-out's loss, both after costs (strategies/base.py r_target).
      const rTarget = (k, stop) => exitMath.rTarget(k, stop, leg, side);
      const t = num("take_profit_pct") !== null ? num("take_profit_pct") / 100 : null;
      const parts = [];
      let warn = false;
      const loss = s !== null ? exitMath.loss(s, leg, side) : null;  // 1R: what a stop-out loses, costs included
      if (loss !== null) parts.push(`A stop-out loses 1R = ${p2(loss)} of ${who}: the ${p2(s)} stop ${w.stop} the entry plus ${p2(loss - s)} in fees and spread.`);
      else if (kind.stop.value === "atr" || kind.stop.value === "swing") parts.push("The stop is set from the market at each entry, so 1R differs from trade to trade; each trade's R shows in the backtest.");
      if (t !== null) {
        const gain = exitMath.gain(t, leg, side), trip = t - gain;
        if (gain <= 0) { warn = true; parts.push(`The ${p2(t)} target doesn't cover the ${p2(trip)} round trip, so every target hit would lose money. It can't be saved.`); }
        else if (loss !== null) {
          const R = gain / loss;
          parts.push(`The target makes ${rr(R)} after costs (${p2(t)} less a ${p2(trip)} round trip), ${(t / s).toFixed(1)}:1 before costs.`);
          if (R < 0.25) { warn = true; parts.push(`That is almost nothing for the risk: costs alone take ${p2(trip)}.`); }
        } else parts.push(`The target makes ${p2(gain)} after a ${p2(trip)} round trip.`);
      } else if (r !== null) {
        if (s !== null) parts.push(`The target makes ${rr(r)} after costs: ${p2(rTarget(r, s))} ${w.tp} the entry.`);
        else parts.push(`The target makes ${rr(r)} after costs, so it sits further out than ${r} stop distances: on a 3% stop, ${p2(rTarget(r, 0.03))} ${w.tp} the entry.`);
        if (r < 0.25) { warn = true; parts.push("That is almost nothing for the risk."); }
      }
      line.hidden = !parts.length;
      line.classList.toggle("warn", warn);
      line.textContent = parts.join(" ");
    };
    kind.stop.addEventListener("change", sync); kind.tp.addEventListener("change", sync);
    form.addEventListener("input", sync); form.addEventListener("change", sync); form.addEventListener("market-change", sync);
    sync();
    box.exitDescribe = () => {
      sync();  // the summary may be asked before this form's own input listener has run
      const out = [], r = num("take_profit_r"), atr = num("stop_atr"), swing = num("stop_swing_bars"), w = ways();
      const stop = num("stop_loss_pct") !== null ? `${num("stop_loss_pct")}% ${w.stop} entry`
        : atr !== null ? `${atr} average true ranges (over ${num("atr_bars") || 14} bars) ${w.stop} entry`
        : swing !== null ? `at the ${w.swing} of the last ${swing} bars` : null;
      const tp = num("take_profit_pct") !== null ? `${num("take_profit_pct")}% ${w.tp} entry`
        : r !== null ? `a target that makes ${r}R after costs` : null;
      if (stop || tp) out.push(`Exit any trade ${[stop, tp].filter(Boolean).join(" or ")}.`);
      if (line && !line.hidden) out.push(line.textContent);
      const rpt = num("risk_per_trade_pct");
      if (rpt) out.push(stop ? `Size each trade to lose about ${rpt}% of capital if the stop is hit, fees and spread included.` : "Risk per trade needs a stop-loss; add one or clear it.");
      return out;
    };
    return box.exitDescribe;
  }

  // The venue: its instruments are the ones suggested, and a perpetual venue lists no spot, so its strategies
  // trade its own perpetual. A page showing one venue's stored history reloads on another.
  function venueField(form) {
    const sel = form && form.elements.venue;
    if (!(sel instanceof HTMLSelectElement)) return;  // the research study's venue switch runs itself
    if (sel.hasAttribute("data-reload")) {
      sel.addEventListener("change", () => { location.href = `${location.pathname}?venue=${encodeURIComponent(sel.value)}`; });
      return;
    }
    const list = document.getElementById("pairs"), inst = form.elements.instrument, mk = form.elements.market;
    const perpOpt = mk ? [...mk.options].find((o) => o.value === "perp") : null;
    if (perpOpt) perpOpt.dataset.label = perpOpt.textContent;
    const sync = (first) => {
      const opt = sel.selectedOptions[0];
      if (!opt) return;
      const pairs = (opt.dataset.pairs || "").split(",").filter(Boolean), perp = !!opt.dataset.perp;
      if (list) list.replaceChildren(...pairs.map((p) => Object.assign(document.createElement("option"), {value: p})));
      if (!first && inst && pairs.length && !pairs.includes(inst.value.trim().toUpperCase())) inst.value = pairs[0];
      if (!mk) return;
      [...mk.options].forEach((o) => { o.disabled = perp && o.value !== "perp"; });
      if (perpOpt) perpOpt.textContent = perp ? `${opt.textContent.split(":")[0]}: the venue's own fees and settled funding` : perpOpt.dataset.label;
      if (perp && mk.value !== "perp") { mk.value = "perp"; mk.dispatchEvent(new Event("change")); }
    };
    sel.addEventListener("change", () => sync(false)); sync(true);
  }

  // Order type: the wait only matters for maker-first orders, so it shows only then.
  function orderFields(formId) {
    const form = document.getElementById(formId);
    exitFields(form);
    venueField(form);
    if (form && form.elements.market) {  // shorts and the testnet mirror only on a perpetual
      const perp = [...form.querySelectorAll("[data-when-perp]")];
      const syncMarket = () => {
        const spot = form.elements.market.value === "spot";
        perp.forEach((el) => { el.hidden = spot; if (spot) el.querySelectorAll("input[type=checkbox]").forEach((c) => { c.checked = false; }); });
        form.dispatchEvent(new Event("market-change"));
      };
      form.elements.market.addEventListener("change", syncMarket); syncMarket();
    }
    if (!form || !form.elements.execution) return;
    const wait = form.querySelector("[data-when-maker]");
    if (!wait) return;  // maker-first orders switched off: market only
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

  // --- Action reasons and the strategy page's switches (combined build F2, F4) ---------------------------------
  // Reason pickers (templates/_reasons.html): a dialog's buttons marked data-needs-reason wait for a pick, and
  // Other for a note of data-min characters. data-blocked keeps a button off whatever the pick.
  function reasons() {
    const check = (fs) => {
      const form = fs.closest("form");
      if (!form) return;
      const min = Number(fs.dataset.min || 10);
      const pick = fs.querySelector("input[name=reason_pick]:checked")?.value || "";
      const note = (fs.querySelector("textarea[name=reason_note]")?.value || "").trim().replace(/\s+/g, " ");
      const ok = Boolean(pick) && (pick !== "Other" || note.length >= min);
      form.querySelectorAll("[data-needs-reason]").forEach((b) => { b.disabled = !ok || b.hasAttribute("data-blocked"); });
      fs.querySelectorAll("label.opt").forEach((l) => l.classList.toggle("sel", Boolean(l.querySelector("input")?.checked)));
      const hint = fs.querySelector("[data-note-hint]");
      if (hint) hint.textContent = pick === "Other" ? `(needed, at least ${min} characters)` : "(optional)";
      const line = fs.querySelector("[data-logline]");
      if (line) line.textContent = "Logged as: " + (pick ? pick + (note ? ": " + note : "") : "no reason yet");
    };
    const all = () => document.querySelectorAll("[data-reasons]").forEach(check);
    once("reasonsBound", () => {
      const on = (e) => { const fs = e.target.closest && e.target.closest("[data-reasons]"); if (fs) check(fs); };
      document.addEventListener("change", on);
      document.addEventListener("input", on);
      document.addEventListener("live:swap", all);
    });
    all();
  }

  // Filter chips over a list (data-chips names the list's id): rows carry data-kind, one or more kinds. The
  // choice is kept through live updates.
  function chips() {
    const chosen = {};
    const apply = (group, k) => {
      const list = document.getElementById(group.dataset.chips);
      group.querySelectorAll("[data-chip]").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.chip === k)));
      list?.querySelectorAll("[data-kind]").forEach((r) => { r.hidden = k !== "all" && !r.dataset.kind.split(" ").includes(k); });
      const none = list?.querySelector("[data-none]");
      if (none) none.hidden = k === "all" || [...list.querySelectorAll("[data-kind]")].some((r) => !r.hidden);
    };
    once("chipsBound", () => {
      document.addEventListener("click", (e) => {
        const b = e.target.closest("[data-chip]");
        if (!b) return;
        const group = b.closest("[data-chips]");
        chosen[group.dataset.chips] = b.dataset.chip;
        apply(group, b.dataset.chip);
      });
      document.addEventListener("live:swap", () => document.querySelectorAll("[data-chips]").forEach((g) => {
        if (chosen[g.dataset.chips]) apply(g, chosen[g.dataset.chips]);
      }));
    });
  }

  // A two-way switch (Price / Equity on the overview chart): the box's data-mode picks what shows.
  function modes() {
    once("modesBound", () => document.addEventListener("click", (e) => {
      const b = e.target.closest("[data-mode-to]");
      if (!b) return;
      const box = b.closest("[data-mode]");
      box.dataset.mode = b.dataset.modeTo;
      box.querySelectorAll("[data-mode-to]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      window.dispatchEvent(new Event("resize"));  // a chart drawn while hidden sizes itself now
    }));
  }

  // Settings: Save first lists every change, before and after, in its confirm dialog, with a warning when the
  // risk profile changes (its option carries the new limits in data-limits).
  function settingsDiff(formId) {
    const form = document.getElementById(formId);
    const dlg = form && form.querySelector("dialog[data-diff]");
    const open = form && form.querySelector("[data-diff-open]");
    if (!dlg || !open) return;
    const words = (t) => (t || "").replace(/\s+/g, " ").trim();
    const label = (f) => words(form.querySelector(`label[for="${CSS.escape(f.id)}"]`)?.textContent || f.name);
    open.addEventListener("click", () => {
      const rows = [];
      let profile = null;
      form.querySelectorAll("input[name], select[name], textarea[name]").forEach((f) => {
        if (f.closest("dialog") || f.type === "hidden" || f.name.startsWith("reason")) return;
        if (f.type === "checkbox") {
          if (f.checked !== f.defaultChecked) rows.push([words(f.closest("label")?.textContent || f.name), f.defaultChecked ? "yes" : "no", f.checked ? "yes" : "no"]);
        } else if (f.tagName === "SELECT") {
          const was = [...f.options].find((o) => o.defaultSelected) || f.options[0];
          if (was && f.value !== was.value) {
            rows.push([label(f), words(was.textContent).split(":")[0], words(f.selectedOptions[0].textContent).split(":")[0]]);
            if (f.name === "risk_profile") profile = f.selectedOptions[0];
          }
        } else if (f.value !== f.defaultValue) {
          rows.push([label(f), f.defaultValue || "none", f.value || "none"]);
        }
      });
      const box = dlg.querySelector("[data-diff-rows]");
      box.replaceChildren(...(rows.length ? rows.map(([k, a, b]) => {
        const d = document.createElement("div");
        d.append(Object.assign(document.createElement("span"), {textContent: k}),
                 Object.assign(document.createElement("span"), {textContent: `${a} → ${b}`}));
        return d;
      }) : [Object.assign(document.createElement("div"), {textContent: "Nothing has changed yet.", className: "muted"})]));
      const warn = dlg.querySelector("[data-diff-warn]");
      if (warn) {
        warn.hidden = !profile;
        if (profile) warn.querySelector("span").textContent = profile.dataset.limits || "";
      }
      dlg.querySelectorAll("[data-needs-change]").forEach((b) => { b.toggleAttribute("data-blocked", !rows.length); });
      document.querySelectorAll("[data-reasons]").forEach((fs) => fs.dispatchEvent(new Event("change", {bubbles: true})));
      dlg.showModal();
    });
  }

  return {sortable, tabs, sortBy, dialogs, whys, strategyPicker, priceChart, sleeveForm, orderFields, reasons, chips, modes, settingsDiff, bookCharts: (url) => pair(url, "eq", "dd", ["Book", "Buy-and-hold"]), pair};
})();
