// Live updates: every few seconds the page fetches itself and swaps in the panels whose figures
// changed, so prices, P&L, positions, orders, status and alerts stay current without a reload.
// Panels with charts or forms are left alone (charts refresh themselves). If updates stop, the
// header says so.
(() => {
  const EVERY = 5000, STALE_AFTER = 20000, MAX_BACKOFF = 60000;
  const indicator = document.getElementById("live");
  if (!indicator || document.body.dataset.live === "off") return;

  // A region is a panel the server renders: an explicit data-live, or a labelled section.
  const keyOf = (el) => el.dataset.live || el.getAttribute("aria-labelledby") || el.getAttribute("aria-label");
  const SELECTOR = "[data-live], main section[aria-labelledby], main section[aria-label]";
  const swappable = (el) => el.dataset.live !== "off" && !el.querySelector("canvas, .pc-canvas")
    && (el.hasAttribute("data-live-forms") || !el.querySelector("input:not([type=hidden]), select, textarea"))
    && !el.parentElement.closest(SELECTOR);
  const regions = (root) => [...root.querySelectorAll(SELECTOR)].filter(swappable);

  // The server's HTML for each region as last applied, so changes the page made itself (sorting,
  // an open Why row) don't count as differences.
  const seen = new Map(regions(document).map((el) => [keyOf(el), el.innerHTML]));

  const busyIn = (el) => {
    if (el.querySelector("dialog[open], .help[aria-expanded=true]")) return true;
    // Never wipe something the PM is typing.
    if (el.contains(document.activeElement) && document.activeElement.matches("input, select, textarea")) return true;
    // A choice or text the PM has made (a radio or box counts once ticked; its value is always there).
    if ([...el.querySelectorAll("input:not([type=hidden]), textarea")].some((i) => (i.type === "radio" || i.type === "checkbox") ? i.checked : i.value)) return true;
    const sel = getSelection();
    return sel && !sel.isCollapsed && el.contains(sel.anchorNode);
  };

  const swap = (el, html) => {
    const open = [...el.querySelectorAll("[data-why][aria-expanded=true]")].map((b) => b.dataset.why);
    const sorts = [...el.querySelectorAll("table.sortable")].map((t) => {
      const heads = [...t.querySelectorAll("thead th")];
      const i = heads.findIndex((h) => h.hasAttribute("aria-sort"));
      return i < 0 ? null : [i, heads[i].getAttribute("aria-sort")];
    });
    const before = [...el.querySelectorAll(".v, td.num")].map((c) => c.textContent);
    el.innerHTML = html;
    open.forEach((id) => el.querySelector(`[data-why="${CSS.escape(id)}"]`)?.click());
    el.querySelectorAll("table.sortable").forEach((t, n) => { if (sorts[n] && window.Console) Console.sortBy(t, ...sorts[n]); });
    // A brief highlight on figures that moved, so a change is noticed without being distracting.
    const after = [...el.querySelectorAll(".v, td.num")];
    if (after.length === before.length) {
      after.forEach((c, n) => {
        if (c.textContent !== before[n]) { c.classList.remove("flash"); void c.offsetWidth; c.classList.add("flash"); }
      });
    }
  };

  let last = Date.now(), delay = EVERY, timer = null, inflight = false;
  const show = () => {
    const age = Date.now() - last;
    const stale = age > STALE_AFTER;
    indicator.hidden = false;
    indicator.dataset.state = stale ? "stale" : "ok";
    const at = new Date(last).toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC"}) + " UTC";
    indicator.textContent = stale ? `Not updating since ${at}` : "Live";
    indicator.title = stale ? "The dashboard can't reach the server. Figures may be out of date; it keeps retrying."
      : `Figures update every ${EVERY / 1000} seconds. Last update ${at}.`;
  };

  async function pull() {
    timer = null;
    if (document.hidden) return;  // resumes on visibilitychange
    inflight = true;
    try {
      const r = await fetch(location.href, {headers: {"X-Live": "1"}, cache: "no-store", credentials: "same-origin"});
      if (!r.ok) throw new Error(String(r.status));
      const doc = new DOMParser().parseFromString(await r.text(), "text/html");
      const fresh = new Map(regions(doc).map((el) => [keyOf(el), el.innerHTML]));
      let changed = false;
      for (const el of regions(document)) {
        const key = keyOf(el), html = fresh.get(key);
        if (html === undefined || html === seen.get(key) || busyIn(el)) continue;
        swap(el, html);
        seen.set(key, html);
        changed = true;
      }
      if (changed) document.dispatchEvent(new CustomEvent("live:swap"));
      last = Date.now();
      delay = EVERY;
    } catch {
      delay = Math.min(delay * 2, MAX_BACKOFF);
    } finally {
      inflight = false;
      show();
      schedule();
    }
  }
  const schedule = () => { if (!timer && !inflight && !document.hidden) timer = setTimeout(pull, delay); };

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { clearTimeout(timer); timer = null; } else { delay = EVERY; clearTimeout(timer); timer = null; pull(); }
  });
  setInterval(show, 5000);
  show();
  schedule();
})();
