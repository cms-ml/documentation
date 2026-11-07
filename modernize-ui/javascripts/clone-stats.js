/* Clone tracker for the skills marketplace page.
 *
 * Fills any <div class="cms-clones" data-src="..."> with summary numbers, an
 * SVG column chart of unique cloners per day, and a table of the data. The
 * data comes from scripts/update_clone_stats.py.
 */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var DAY_MS = 86400000;
  var WEEKLY_AFTER_DAYS = 120; // switch to weekly columns past this many days
  var PLOT_HEIGHT = 180;
  var MARGIN = { top: 22, right: 4, bottom: 26, left: 34 };

  var fmtShort = new Intl.DateTimeFormat("en", { month: "short", day: "numeric", timeZone: "UTC" });
  var fmtLong = new Intl.DateTimeFormat("en", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
  var fmtNum = new Intl.NumberFormat("en");

  function parseDay(s) {
    return new Date(s + "T00:00:00Z");
  }

  function svg(tag, attrs, parent) {
    var node = document.createElementNS(SVG_NS, tag);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function html(tag, className, text, parent) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  /* Columns: one per day, or one per week (starting Monday) for long records.
   * Days missing from the record stay null so gaps are not drawn as zeros. */
  function buildColumns(days) {
    var byDate = {};
    days.forEach(function (d) { byDate[d.date] = d; });
    var first = parseDay(days[0].date).getTime();
    var last = parseDay(days[days.length - 1].date).getTime();
    var weekly = (last - first) / DAY_MS + 1 > WEEKLY_AFTER_DAYS;
    var columns = [];
    var current = null;

    for (var t = first; t <= last; t += DAY_MS) {
      var day = new Date(t);
      var key = day.toISOString().slice(0, 10);
      var startsWeek = day.getUTCDay() === 1;
      if (!current || !weekly || startsWeek) {
        current = { start: day, end: day, weekly: weekly, uniques: null, count: null };
        columns.push(current);
      }
      current.end = day;
      var d = byDate[key];
      if (d) {
        current.uniques = (current.uniques || 0) + d.uniques;
        current.count = (current.count || 0) + d.count;
      }
    }
    return columns;
  }

  function columnLabel(c) {
    if (!c.weekly) return fmtLong.format(c.start);
    return "Week of " + fmtLong.format(c.start);
  }

  function niceStep(max, ticks) {
    var raw = Math.max(max, 1) / ticks;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
    return Math.max(1, step * mag);
  }

  /* Column with a 4px rounded top and a square base */
  function columnPath(x, y, w, h) {
    var r = Math.min(4, w / 2, h);
    return "M" + x + "," + (y + h) +
      "V" + (y + r) +
      "Q" + x + "," + y + " " + (x + r) + "," + y +
      "H" + (x + w - r) +
      "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) +
      "V" + (y + h) + "Z";
  }

  function renderStats(root, record) {
    var days = record.days;
    var total = days.reduce(function (s, d) { return s + d.count; }, 0);
    var peak = days.reduce(function (p, d) { return d.uniques > p.uniques ? d : p; }, days[0]);
    var stats = html("div", "cms-clones__stats", null, root);

    function tile(value, label) {
      var t = html("div", "cms-clones__stat", null, stats);
      html("span", "cms-clones__value", fmtNum.format(value), t);
      html("span", "cms-clones__label", label, t);
    }

    if (record.window) {
      tile(record.window.uniques,
        "Unique cloners in the 14 days to " + fmtShort.format(parseDay(record.window.end)));
    }
    tile(total, "Clones since " + fmtLong.format(parseDay(days[0].date)));
    tile(peak.uniques, "Most unique cloners in a day (" + fmtShort.format(parseDay(peak.date)) + ")");
  }

  function renderChart(frame, tooltip, columns) {
    var old = frame.querySelector("svg");
    if (old) old.remove();

    var width = Math.max(frame.clientWidth, 260);
    var height = MARGIN.top + PLOT_HEIGHT + MARGIN.bottom;
    var plotW = width - MARGIN.left - MARGIN.right;
    var n = columns.length;
    var band = plotW / n;
    var barW = Math.max(1, Math.min(24, band - 2)); // 2px surface gap between columns

    var maxU = columns.reduce(function (m, c) { return Math.max(m, c.uniques || 0); }, 0);
    var step = niceStep(maxU, 4);
    var yMax = Math.max(step, Math.ceil(maxU / step) * step);
    function y(v) { return MARGIN.top + PLOT_HEIGHT - (v / yMax) * PLOT_HEIGHT; }

    var root = svg("svg", {
      width: width,
      height: height,
      viewBox: "0 0 " + width + " " + height,
      role: "group",
      "aria-label": "Unique cloners per " + (columns[0].weekly ? "week" : "day")
    });
    frame.insertBefore(root, tooltip);

    // Gridlines and y-axis labels
    var grid = svg("g", { "aria-hidden": "true" }, root);
    for (var v = 0; v <= yMax; v += step) {
      var gy = Math.round(y(v)) + 0.5;
      svg("line", { class: "cms-clones__grid", x1: MARGIN.left, x2: width - MARGIN.right, y1: gy, y2: gy }, grid);
      svg("text", { class: "cms-clones__axis", x: MARGIN.left - 8, y: gy, "text-anchor": "end", "dominant-baseline": "middle" }, grid)
        .textContent = fmtNum.format(v);
    }

    // X-axis labels: spaced at least 64px apart, always including the newest column
    var every = Math.max(1, Math.ceil(64 / band));
    var axis = svg("g", { "aria-hidden": "true" }, root);
    columns.forEach(function (c, i) {
      if ((n - 1 - i) % every !== 0) return;
      var cx = MARGIN.left + band * (i + 0.5);
      var anchor = "middle";
      if (cx - MARGIN.left < 24) anchor = "start";
      if (width - MARGIN.right - cx < 24) anchor = "end";
      svg("text", { class: "cms-clones__axis", x: anchor === "start" ? cx - barW / 2 : anchor === "end" ? cx + barW / 2 : cx, y: height - 6, "text-anchor": anchor }, axis)
        .textContent = fmtShort.format(c.start);
    });

    // Columns, each with a full-height hit area for hover and keyboard focus
    var peakIndex = -1;
    columns.forEach(function (c, i) { if (c.uniques === maxU && maxU > 0) peakIndex = i; });

    columns.forEach(function (c, i) {
      var x0 = MARGIN.left + band * i;
      var g = svg("g", { class: "cms-clones__column" }, root);
      if (c.uniques != null) {
        g.setAttribute("tabindex", "0");
        g.setAttribute("aria-label", columnLabel(c) + ": " + fmtNum.format(c.uniques) +
          " unique cloners, " + fmtNum.format(c.count) + " clones");
      }
      svg("rect", { class: "cms-clones__hit", x: x0, y: MARGIN.top, width: band, height: PLOT_HEIGHT }, g);
      if (c.uniques) {
        var top = y(c.uniques);
        svg("path", { class: "cms-clones__bar", d: columnPath(x0 + (band - barW) / 2, top, barW, MARGIN.top + PLOT_HEIGHT - top) }, g);
      }
      if (i === peakIndex) {
        // Label only the tallest column; the tooltip and table carry the rest
        svg("text", { class: "cms-clones__peak", x: x0 + band / 2, y: y(c.uniques) - 6, "text-anchor": "middle", "aria-hidden": "true" }, g)
          .textContent = fmtNum.format(c.uniques);
      }
      if (c.uniques == null) return;

      function show() {
        root.querySelectorAll(".is-active").forEach(function (a) { a.classList.remove("is-active"); });
        g.classList.add("is-active");
        tooltip.replaceChildren();
        var line = html("div", null, null, tooltip);
        html("strong", null, fmtNum.format(c.uniques), line);
        line.appendChild(document.createTextNode(" unique cloners"));
        html("div", null, fmtNum.format(c.count) + " clones", tooltip);
        html("div", "cms-clones__tooltip-date", columnLabel(c), tooltip);
        tooltip.hidden = false;
        var tx = x0 + band / 2;
        var half = tooltip.offsetWidth / 2;
        tx = Math.min(Math.max(tx, half), width - half);
        tooltip.style.left = tx + "px";
        tooltip.style.top = (c.uniques ? y(c.uniques) : MARGIN.top + PLOT_HEIGHT) - 8 + "px";
      }
      function hide() {
        g.classList.remove("is-active");
        tooltip.hidden = true;
      }
      g.addEventListener("pointerenter", show);
      g.addEventListener("pointerleave", hide);
      g.addEventListener("focus", show);
      g.addEventListener("blur", hide);
    });
  }

  function renderTable(root, record) {
    var details = html("details", "cms-clones__data", null, root);
    html("summary", null, "Show the data", details);
    var table = html("table", null, null, details);
    var head = html("tr", null, null, html("thead", null, null, table));
    ["Date", "Unique cloners", "Clones"].forEach(function (h) { html("th", null, h, head); });
    var body = html("tbody", null, null, table);
    record.days.slice().reverse().forEach(function (d) {
      var row = html("tr", null, null, body);
      html("td", null, fmtLong.format(parseDay(d.date)), row);
      html("td", null, fmtNum.format(d.uniques), row);
      html("td", null, fmtNum.format(d.count), row);
    });
  }

  function setup(root) {
    if (root.dataset.ready) return;
    root.dataset.ready = "true";
    var src = new URL(root.dataset.src, window.location.href);

    fetch(src)
      .then(function (r) {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      })
      .then(function (record) {
        if (!record.days || !record.days.length) throw new Error("no data");
        root.replaceChildren();
        renderStats(root, record);

        var card = html("div", "cms-clones__card", null, root);
        html("p", "cms-clones__title", "Unique cloners per " +
          (buildColumns(record.days)[0].weekly ? "week" : "day"), card);
        var frame = html("div", "cms-clones__frame", null, card);
        var tooltip = html("div", "cms-clones__tooltip", null, frame);
        tooltip.hidden = true;
        tooltip.setAttribute("role", "status");
        var columns = buildColumns(record.days);
        renderChart(frame, tooltip, columns);

        if (record.updated) {
          html("p", "cms-clones__updated",
            "Last updated " + fmtLong.format(new Date(record.updated)), card);
        }
        renderTable(root, record);

        var width = frame.clientWidth;
        var observer = new ResizeObserver(function () {
          if (!frame.isConnected) return observer.disconnect();
          if (frame.clientWidth === width) return;
          width = frame.clientWidth;
          renderChart(frame, tooltip, columns);
        });
        observer.observe(frame);
      })
      .catch(function () {
        root.replaceChildren();
        var p = html("p", "cms-clones__error", "Couldn't load the clone data. ", root);
        var a = html("a", null, "Open the raw data", p);
        a.href = src.href;
      });
  }

  function init() {
    document.querySelectorAll(".cms-clones[data-src]").forEach(setup);
  }

  // Material's instant navigation swaps pages without a reload: rerun on each page
  if (window.document$) window.document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
