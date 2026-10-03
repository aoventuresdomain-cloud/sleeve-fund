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
    fetch(url).then((r) => r.json()).then((d) => {
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
          scales: {x: {ticks: {color: muted, maxTicksLimit: window.innerWidth < 760 ? 4 : 7, maxRotation: 0}, grid: {display: false}},
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

  return {sortable, dialogs, bookCharts: (url) => pair(url, "eq", "dd", ["Book", "Buy-and-hold"]), pair};
})();
