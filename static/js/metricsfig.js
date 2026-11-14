// Quantitative results as three small line charts (one per metric, each with its own axis):
// x = number of lines, one line per abstraction level (ordinal blue ramp, light = geometric, dark = semantic).
// Hovering a chart shows a crosshair synced across all three and a tooltip with every level's value;
// hovering a legend entry highlights that level everywhere, clicking pins it. A table view keeps the numbers.
import { rich } from './bezierfig.js';

const DATA = './static/data/metrics.json';
const RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281'];   // validated ordinal ramp (light surface)
const W = 300, H = 210, M = { l: 44, r: 12, t: 12, b: 34 };
const NS = 'http://www.w3.org/2000/svg';

function el(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.append(e);
  return e;
}
function svgEl(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}
function niceTicks(lo, hi, n = 4) {
  const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => span / s <= n + 0.5) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

export async function metricsCharts(root) {
  const m = await fetch(DATA).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (!m) { el('p', 'sg-caption', root, 'Quantitative results will appear here when all runs are finished.'); return; }
  root.classList.add('mc');
  const rows = m.rows, cols = m.cols;
  const color = Object.fromEntries(rows.map((r, i) => [r.key, RAMP[Math.round((i * (RAMP.length - 1)) / Math.max(rows.length - 1, 1))]]));
  const st = { col: null, hover: null, pinned: null };

  // toolbar: chart / table
  const top = el('div', 'sg-toolbar', root);
  const views = el('div', 'buttons has-addons', top);
  const bChart = el('button', 'button is-small is-dark is-selected', views, 'chart');
  const bTable = el('button', 'button is-small', views, 'table');

  // legend (shared by the three charts)
  const legend = el('div', 'mc-legend', root);
  const legendItems = rows.map((r) => {
    const it = el('button', 'mc-legend-item', legend);
    const sw = el('span', 'mc-swatch', it); sw.style.background = color[r.key];
    el('span', 'mc-legend-label', it, r.label);
    if (r.sub) el('span', 'mc-legend-sub', it, r.sub);
    it.addEventListener('mouseenter', () => { st.hover = r.key; render(); });
    it.addEventListener('mouseleave', () => { st.hover = null; render(); });
    it.addEventListener('click', () => { st.pinned = st.pinned === r.key ? null : r.key; render(); });
    return [r.key, it];
  });

  const chartsWrap = el('div', 'mc-charts', root);
  const tableWrap = el('div', 'mc-tables is-hidden', root);
  bChart.addEventListener('click', () => { bChart.classList.add('is-dark', 'is-selected'); bTable.classList.remove('is-dark', 'is-selected'); chartsWrap.classList.remove('is-hidden'); legend.classList.remove('is-hidden'); tableWrap.classList.add('is-hidden'); });
  bTable.addEventListener('click', () => { bTable.classList.add('is-dark', 'is-selected'); bChart.classList.remove('is-dark', 'is-selected'); tableWrap.classList.remove('is-hidden'); legend.classList.add('is-hidden'); chartsWrap.classList.add('is-hidden'); });

  // ------------------------------------------------------------------ charts
  const charts = m.metrics.map((metric) => {
    const card = el('div', 'mc-card', chartsWrap);
    const title = el('div', 'mc-title', card);
    el('span', '', title, metric.label);
    el('span', 'mc-better', title, metric.higher_is_better ? '↑ higher is better' : '↓ lower is better');
    const box = el('div', 'mc-plot', card);
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'mc-svg', role: 'img', 'aria-label': `${metric.label} by number of lines and abstraction level` }, box);
    const tip = el('div', 'mc-tip', box);

    const vals = rows.flatMap((r) => cols.map((c) => metric.values[`${r.key}|${c.key}`])).filter((v) => v != null);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const ticks = niceTicks(lo, hi);
    lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
    const x = (j) => M.l + ((W - M.l - M.r) * j) / (cols.length - 1);
    const y = (v) => M.t + (H - M.t - M.b) * (1 - (v - lo) / (hi - lo));

    // grid and axes (recessive)
    const g = svgEl('g', {}, svg);
    for (const t of ticks) {
      svgEl('line', { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), class: 'mc-grid' }, g);
      svgEl('text', { x: M.l - 6, y: y(t) + 3.5, class: 'mc-tick', 'text-anchor': 'end' }, g).textContent = t.toFixed(2);
    }
    cols.forEach((c, j) => { svgEl('text', { x: x(j), y: H - M.b + 16, class: 'mc-tick', 'text-anchor': 'middle' }, g).textContent = c.label; });
    svgEl('text', { x: (M.l + W - M.r) / 2, y: H - 3, class: 'mc-axis', 'text-anchor': 'middle' }, g).textContent = 'number of lines';

    const cross = svgEl('line', { y1: M.t, y2: H - M.b, class: 'mc-cross' }, svg);
    const series = rows.map((r) => {
      const pts = cols.map((c, j) => [x(j), metric.values[`${r.key}|${c.key}`]]).filter(([, v]) => v != null).map(([px, v]) => [px, y(v)]);
      const sg = svgEl('g', { class: 'mc-series' }, svg);
      const path = svgEl('path', { d: 'M' + pts.map((p) => p.join(',')).join(' L'), class: 'mc-line', stroke: color[r.key] }, sg);
      const dots = pts.map(([px, py]) => svgEl('circle', { cx: px, cy: py, r: 4, class: 'mc-dot', fill: color[r.key] }, sg));
      return { key: r.key, g: sg, path, dots };
    });

    // hover: nearest column, tooltip with all levels
    const hit = svgEl('rect', { x: M.l - 14, y: 0, width: W - M.l - M.r + 28, height: H - M.b + 4, class: 'mc-hit' }, svg);
    const toCol = (e) => {
      const b = svg.getBoundingClientRect(), sx = ((e.clientX - b.left) / b.width) * W;
      return Math.max(0, Math.min(cols.length - 1, Math.round(((sx - M.l) / (W - M.l - M.r)) * (cols.length - 1))));
    };
    hit.addEventListener('pointermove', (e) => { st.col = toCol(e); render(); });
    hit.addEventListener('pointerleave', () => { st.col = null; render(); });

    return { metric, card, svg, tip, series, cross, x, y, box };
  });

  function render() {
    const focus = st.hover || st.pinned;
    legendItems.forEach(([k, it]) => {
      it.classList.toggle('is-dim', !!focus && focus !== k);
      it.classList.toggle('is-pinned', st.pinned === k);
    });
    for (const ch of charts) {
      for (const s of ch.series) {
        s.g.classList.toggle('is-dim', !!focus && focus !== s.key);
        s.g.classList.toggle('is-focus', focus === s.key);
        s.dots.forEach((d, j) => d.setAttribute('r', st.col === j ? 6 : 4));
      }
      if (st.col === null) { ch.cross.style.opacity = 0; ch.tip.classList.remove('is-on'); continue; }
      const j = st.col, c = cols[j], cx = ch.x(j);
      ch.cross.setAttribute('x1', cx); ch.cross.setAttribute('x2', cx); ch.cross.style.opacity = 1;
      const entries = rows.map((r) => [r, ch.metric.values[`${r.key}|${c.key}`]]).filter(([, v]) => v != null);
      const best = (ch.metric.higher_is_better ? Math.max : Math.min)(...entries.map(([, v]) => v));
      ch.tip.innerHTML = '';
      el('div', 'mc-tip-head', ch.tip, c.label === '1' ? '1 line' : `${c.label} lines`);
      for (const [r, v] of entries) {
        const row = el('div', 'mc-tip-row', ch.tip);
        row.classList.toggle('is-focus', focus === r.key);
        row.classList.toggle('is-dim', !!focus && focus !== r.key);
        el('span', 'mc-swatch', row).style.background = color[r.key];
        el('span', 'mc-tip-label', row, r.label);
        const val = el('span', 'mc-tip-val', row, v.toFixed(3));
        if (v === best) { val.classList.add('is-best'); val.title = 'best at this line count'; }
      }
      // place the tooltip beside the crosshair, flipping near the right edge
      const b = ch.box.getBoundingClientRect(), px = (cx / W) * b.width;
      ch.tip.classList.add('is-on');
      const tw = ch.tip.offsetWidth;
      const left = px + 14 + tw > b.width ? px - 14 - tw : px + 14;
      ch.tip.style.left = `${Math.max(0, Math.min(left, b.width - tw))}px`;
    }
  }

  // draw-in when first visible
  new IntersectionObserver((es, ob) => {
    if (!es[0].isIntersecting) return;
    root.classList.add('is-drawn');
    ob.disconnect();
  }, { threshold: 0.3 }).observe(chartsWrap);
  for (const ch of charts) for (const s of ch.series) {
    const len = s.path.getTotalLength ? s.path.getTotalLength() : 600;
    s.path.style.setProperty('--len', Math.ceil(len) + 2);
  }

  // ------------------------------------------------------------------ table view (exact numbers)
  for (const metric of m.metrics) {
    const t = el('table', 'table is-narrow sg-metric', el('div', 'sg-metric-wrap', tableWrap));
    el('caption', '', t, `${metric.label} ${metric.higher_is_better ? '(higher is better)' : '(lower is better)'}`);
    const head = el('tr', '', t);
    el('th', '', head, '');
    cols.forEach((c) => el('th', '', head, c.label));
    for (const r of rows) {
      const tr = el('tr', '', t);
      const th = el('th', '', tr, r.label);
      if (r.sub) el('span', 'sg-sub', th, r.sub);
      cols.forEach((c) => { const v = metric.values[`${r.key}|${c.key}`]; el('td', '', tr, v == null ? '–' : v.toFixed(3)); });
    }
  }
  render();
}
