// Interactive figure: a cubic Bezier curve -> spherical Gaussians along its arc -> splatted stroke.
// Drag the control points; the sliders change the Gaussian radius r and the spacing d between centers.
// Panel 3 is computed per pixel exactly like the rasterizer: V = 1 - prod_n (1 - w_n).

export const COLORS = { control: '#e8743b', curve: '#2f6fdf', radius: '#d6453d', ink: '#111111', guide: '#b9bfca' };

const ASPECT = 0.72;            // panel height / width
const ZOOM = 5;                 // magnification of the middle panel
const FOCUS_T = 0.42;           // curve parameter the magnifier looks at
const DEFAULT = { P: [[0.1, 0.5], [0.27, 0.07], [0.63, 0.7], [0.9, 0.16]], r: 0.008, s: 0.5 };

function bezier(P, t) {
  const u = 1 - t, b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
  return [b0 * P[0][0] + b1 * P[1][0] + b2 * P[2][0] + b3 * P[3][0],
          b0 * P[0][1] + b1 * P[1][1] + b2 * P[2][1] + b3 * P[3][1]];
}

// Centers spaced uniformly in arc length, at most d apart (N = ceil(L / d) + 1).
function arcSamples(P, d) {
  const M = 600, pts = [], cum = [0];
  for (let i = 0; i <= M; i++) pts.push(bezier(P, i / M));
  for (let i = 1; i <= M; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const L = cum[M], N = Math.ceil(L / d) + 1, out = [];
  let j = 0;
  for (let n = 0; n < N; n++) {
    const s = (L * n) / (N - 1);
    while (j < M - 1 && cum[j + 1] < s) j++;
    const f = (s - cum[j]) / Math.max(cum[j + 1] - cum[j], 1e-9);
    out.push([pts[j][0] + f * (pts[j + 1][0] - pts[j][0]), pts[j][1] + f * (pts[j + 1][1] - pts[j][1])]);
  }
  return { centers: out, L };
}

export function tex(el, src) {
  if (window.katex) window.katex.render(src, el, { throwOnError: false });
  else el.textContent = src;
}

// Text with inline \( math \) pieces.
export function rich(el, src) {
  src.split(/\\\((.*?)\\\)/).forEach((part, i) => {
    const span = document.createElement('span');
    if (i % 2) tex(span, part); else span.textContent = part;
    el.append(span);
  });
}

function make(tag, cls, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (parent) parent.append(e);
  return e;
}

class Panel {
  constructor(parent, caption) {
    this.wrap = make('div', 'bz-panel', parent);
    this.box = make('div', 'bz-canvas', this.wrap);
    this.canvas = make('canvas', '', this.box);
    this.ctx = this.canvas.getContext('2d');
    this.labels = make('div', 'bz-labels', this.box);
    rich(make('div', 'bz-cap', this.wrap), caption);
  }

  resize() {
    const w = this.box.clientWidth, h = Math.round(w * ASPECT), dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.box.style.height = `${h}px`;
    this.w = w; this.h = h; this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${w}px`; this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  label(src, color) {
    const e = make('div', 'bz-label', this.labels);
    if (color) e.style.color = color;
    tex(e, src);
    return e;
  }
}

function place(el, x, y, visible = true) {
  const box = el.parentElement.getBoundingClientRect();
  x = Math.min(Math.max(x, 12), box.width - 12); y = Math.min(Math.max(y, 12), box.height - 12);
  el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
  el.style.opacity = visible ? 1 : 0;
}

export function bezierFigure(root) {
  const st = { P: DEFAULT.P.map((p) => p.slice()), r: DEFAULT.r, s: DEFAULT.s, drag: -1, hover: -1, t: 0 };
  root.classList.add('bz-fig');

  const row = make('div', 'bz-row', root);
  const p1 = new Panel(row, 'cubic Bézier curve \\(\\textcolor{#2f6fdf}{\\mathbf{B}(t)}\\)');
  make('div', 'bz-arrow', row).innerHTML = '<span>sample</span>';
  const p2 = new Panel(row, `Gaussians at the centers \\(\\textcolor{#2f6fdf}{\\mathbf{c}_n}\\), zoomed ×${ZOOM}`);
  make('div', 'bz-arrow', row).innerHTML = '<span>splat</span>';
  const p3 = new Panel(row, 'splatted stroke \\(\\mathbf{V} = 1 - \\textstyle\\prod_n (1 - w_n)\\)');

  // labels of panel 1 and 2
  const pl = [0, 1, 2, 3].map((i) => p1.label(`\\mathbf{P}_${i}`, COLORS.control));
  const bl = p1.label('\\mathbf{B}(t)', COLORS.curve);
  const cl = p2.label('\\mathbf{c}_n', COLORS.curve);
  const rl = p2.label('r', COLORS.radius);
  const dl = p2.label('d', '#666');

  // controls
  const ctl = make('div', 'bz-controls', root);
  const slider = (label, min, max, step, value, fmt, on) => {
    const g = make('label', 'bz-slider', ctl);
    tex(make('span', 'bz-slider-name', g), label);
    const inp = make('input', '', g);
    Object.assign(inp, { type: 'range', min, max, step, value });
    const out = make('span', 'bz-slider-value', g);
    const upd = () => { on(Number(inp.value)); out.textContent = fmt(Number(inp.value)); };
    inp.addEventListener('input', () => { upd(); redraw(); });
    upd();
    return inp;
  };
  const rIn = slider('\\textcolor{#d6453d}{r}', 0.004, 0.016, 0.0005, st.r, (v) => `${(v * 1000).toFixed(1)}`, (v) => { st.r = v; });
  const sIn = slider('d / r', 0.25, 4, 0.05, st.s, (v) => v.toFixed(2), (v) => { st.s = v; });
  const info = make('span', 'bz-info', ctl);
  const reset = make('button', 'button is-small is-rounded', ctl);
  reset.textContent = 'reset';
  reset.addEventListener('click', () => {
    st.P = DEFAULT.P.map((p) => p.slice());
    rIn.value = DEFAULT.r; sIn.value = DEFAULT.s;
    rIn.dispatchEvent(new Event('input')); sIn.dispatchEvent(new Event('input'));
  });
  make('p', 'bz-hint', root).textContent = 'Drag the orange control points. Raise d / r to see why the centers must be dense.';

  // ---------------------------------------------------------------- drawing

  let samples = null;

  function drawCurve(ctx, P, map, width, color, dash = []) {
    ctx.beginPath();
    for (let i = 0; i <= 120; i++) {
      const [x, y] = map(bezier(P, i / 120));
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.setLineDash(dash); ctx.lineWidth = width; ctx.strokeStyle = color; ctx.stroke(); ctx.setLineDash([]);
  }

  function drawPanel1() {
    const { ctx, w, h } = p1, P = st.P, m = (p) => [p[0] * w, p[1] * w];
    ctx.clearRect(0, 0, w, h);
    // magnifier footprint
    const F = m(bezier(P, FOCUS_T)), half = w / ZOOM / 2;
    ctx.strokeStyle = COLORS.guide; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.strokeRect(F[0] - half, F[1] - half * ASPECT, 2 * half, 2 * half * ASPECT); ctx.setLineDash([]);
    // control polygon and curve
    ctx.beginPath(); P.forEach((p, i) => (i ? ctx.lineTo(...m(p)) : ctx.moveTo(...m(p))));
    ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4; ctx.strokeStyle = COLORS.control; ctx.globalAlpha = 0.8; ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
    drawCurve(ctx, P, m, 3, COLORS.curve);
    // de Casteljau construction at the animated t
    const t = st.t;
    let lv = P.map(m);
    const shades = ['rgba(232,116,59,0.55)', 'rgba(120,120,140,0.55)'];
    for (let k = 0; k < 2; k++) {
      const nx = [];
      for (let i = 0; i + 1 < lv.length; i++) nx.push([lv[i][0] + t * (lv[i + 1][0] - lv[i][0]), lv[i][1] + t * (lv[i + 1][1] - lv[i][1])]);
      ctx.beginPath(); nx.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
      ctx.strokeStyle = shades[k]; ctx.lineWidth = 1; ctx.stroke();
      nx.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 2.2, 0, 7); ctx.fillStyle = shades[k]; ctx.fill(); });
      lv = nx;
    }
    const B = m(bezier(P, t));
    ctx.beginPath(); ctx.arc(B[0], B[1], 5, 0, 7); ctx.fillStyle = COLORS.curve; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
    // handles
    const cx = P.reduce((a, p) => a + p[0], 0) / 4 * w, cy = P.reduce((a, p) => a + p[1], 0) / 4 * w;
    P.forEach((p, i) => {
      const [x, y] = m(p), big = i === st.hover || i === st.drag;
      ctx.beginPath(); ctx.arc(x, y, big ? 8 : 6, 0, 7);
      ctx.fillStyle = '#fff'; ctx.fill(); ctx.lineWidth = 2.2; ctx.strokeStyle = COLORS.control; ctx.stroke();
      const dx = x - cx, dy = y - cy, n = Math.hypot(dx, dy) || 1;
      let lx = x + (dx / n) * 20, ly = y + (dy / n) * 20;
      if (ly < 12 || ly > h - 12) { lx = x + (dx < 0 ? -20 : 20); ly = y; }      // no room above/below: go sideways
      if (lx < 12 || lx > w - 12) { lx = x; ly = y + (dy < 0 ? -20 : 20); }
      place(pl[i], lx, ly);
    });
    const B2 = m(bezier(P, Math.min(t + 0.01, 1))), B1 = m(bezier(P, Math.max(t - 0.01, 0)));
    let nx = -(B2[1] - B1[1]), ny = B2[0] - B1[0];
    const nn = Math.hypot(nx, ny) || 1; nx /= nn; ny /= nn;
    if (nx * (B[0] - cx) + ny * (B[1] - cy) < 0) { nx = -nx; ny = -ny; }
    place(bl, B[0] + nx * 20, B[1] + ny * 20);
  }

  function drawPanel2() {
    const { ctx, w, h } = p2, P = st.P, F = bezier(P, FOCUS_T);
    const m = (p) => [(p[0] - F[0]) * w * ZOOM + w / 2, (p[1] - F[1]) * w * ZOOM + h / 2];
    ctx.clearRect(0, 0, w, h);
    drawCurve(ctx, P, m, 1, COLORS.guide, [4, 4]);
    const rad = st.r * w * ZOOM, vis = samples.centers.map(m).filter(([x, y]) => x > -3 * rad && x < w + 3 * rad && y > -3 * rad && y < h + 3 * rad);
    const a = Math.min(1, 0.25 + st.s / 2);
    ctx.globalCompositeOperation = 'multiply';
    for (const [x, y] of vis) {
      const g = ctx.createRadialGradient(x, y, 0, x, y, 2.5 * rad);
      g.addColorStop(0, `rgba(47,111,223,${0.55 * a})`); g.addColorStop(0.45, `rgba(47,111,223,${0.22 * a})`); g.addColorStop(1, 'rgba(47,111,223,0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 2.5 * rad, 0, 7); ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';
    for (const [x, y] of vis) {
      ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fillStyle = '#1d3f8a'; ctx.fill();
      ctx.lineWidth = 0.8; ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.stroke();
    }
    // annotate the center closest to the middle: radius r and spacing d to its neighbour
    const all = samples.centers.map(m);
    let k = 0, best = Infinity;
    all.forEach(([x, y], i) => { const dd = (x - w / 2) ** 2 + (y - h / 2) ** 2; if (dd < best && i + 1 < all.length) { best = dd; k = i; } });
    const [x, y] = all[k], [x2, y2] = all[k + 1];
    const tx = (x2 - x), ty = (y2 - y), tn = Math.hypot(tx, ty) || 1, nx = -ty / tn, ny = tx / tn;
    ctx.strokeStyle = COLORS.radius; ctx.fillStyle = COLORS.radius; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + nx * rad, y + ny * rad); ctx.stroke();
    ctx.beginPath(); ctx.arc(x + nx * rad, y + ny * rad, 2, 0, 7); ctx.fill();
    place(rl, x + nx * (rad + 11), y + ny * (rad + 11));
    place(cl, x - nx * 20 - (tx / tn) * 8, y - ny * 20 - (ty / tn) * 8);
    const off = -2.6 * rad;   // spacing bracket on the other side of the curve
    ctx.strokeStyle = tn > 6 ? '#666' : 'rgba(0,0,0,0)'; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + nx * (off + 4), y + ny * (off + 4)); ctx.lineTo(x + nx * off, y + ny * off);
    ctx.lineTo(x2 + nx * off, y2 + ny * off); ctx.lineTo(x2 + nx * (off + 4), y2 + ny * (off + 4)); ctx.stroke();
    place(dl, (x + x2) / 2 + nx * (off - 10), (y + y2) / 2 + ny * (off - 10), tn > 6);
  }

  // Exact splatting, windowed to +-4 radii around every center (like the sparse rasterizer).
  function drawPanel3() {
    const { ctx, w, h, dpr } = p3, W = p3.canvas.width, H = p3.canvas.height;
    const logT = new Float32Array(W * H), rpx = st.r * w * dpr, win = Math.ceil(4 * rpx), inv = 1 / (2 * rpx * rpx);
    for (const [cx0, cy0] of samples.centers) {
      const cx = cx0 * w * dpr, cy = cy0 * w * dpr;
      const x0 = Math.max(0, Math.floor(cx - win)), x1 = Math.min(W - 1, Math.ceil(cx + win));
      const y0 = Math.max(0, Math.floor(cy - win)), y1 = Math.min(H - 1, Math.ceil(cy + win));
      for (let y = y0; y <= y1; y++) {
        const dy = y + 0.5 - cy, row = y * W;
        for (let x = x0; x <= x1; x++) {
          const dx = x + 0.5 - cx, wgt = Math.exp(-(dx * dx + dy * dy) * inv);
          logT[row + x] += Math.log(Math.max(1 - wgt, 1e-6));
        }
      }
    }
    const img = ctx.createImageData(W, H), d = img.data;
    for (let i = 0; i < W * H; i++) {
      d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = 17;
      d[4 * i + 3] = Math.round(255 * (1 - Math.exp(logT[i])));
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H); ctx.putImageData(img, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function redraw() {
    samples = arcSamples(st.P, st.r * st.s);
    info.innerHTML = '';
    tex(info, `N = ${samples.centers.length}\\ \\text{centers}`);
    drawPanel2(); drawPanel3();
  }

  // ---------------------------------------------------------------- interaction

  const toP = (e) => { const b = p1.box.getBoundingClientRect(); return [(e.clientX - b.left) / p1.w, (e.clientY - b.top) / p1.w]; };
  const pick = (q) => {
    let k = -1, best = (14 / p1.w) ** 2;
    st.P.forEach((p, i) => { const dd = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2; if (dd < best) { best = dd; k = i; } });
    return k;
  };
  p1.box.addEventListener('pointerdown', (e) => {
    st.drag = pick(toP(e));
    if (st.drag >= 0) { p1.box.setPointerCapture(e.pointerId); e.preventDefault(); }
  });
  p1.box.addEventListener('pointermove', (e) => {
    const q = toP(e);
    if (st.drag >= 0) {
      st.P[st.drag] = [Math.min(Math.max(q[0], 0.03), 0.97), Math.min(Math.max(q[1], 0.03), ASPECT - 0.03)];
      redraw();
    } else {
      st.hover = pick(q);
    }
    p1.box.style.cursor = st.drag >= 0 ? 'grabbing' : st.hover >= 0 ? 'grab' : 'default';
  });
  const up = () => { st.drag = -1; };
  p1.box.addEventListener('pointerup', up);
  p1.box.addEventListener('pointercancel', up);

  // ---------------------------------------------------------------- loop

  const resize = () => { [p1, p2, p3].forEach((p) => p.resize()); redraw(); };
  new ResizeObserver(resize).observe(row);
  let visible = false, last = performance.now(), clock = 0;
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { rootMargin: '100px' }).observe(root);
  const loop = (now) => {
    const dt = Math.min((now - last) / 1000, 0.1); last = now;
    if (visible && p1.w) {
      clock += dt;
      st.t = 0.5 - 0.5 * Math.cos((clock * 2 * Math.PI) / 7);   // t sweeps 0 -> 1 -> 0 every 7 s
      drawPanel1();
    }
    requestAnimationFrame(loop);
  };
  resize();
  requestAnimationFrame(loop);
}
