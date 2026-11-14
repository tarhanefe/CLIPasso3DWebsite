// Connected vs. not connected: the same objects trained without and with the connectivity loss (+ weld).
// Top row: every separate piece in its own color, each floating on its own.
// Bottom row: one piece; a wave of light runs through the contact graph and sparkles twinkle on it.
// Pieces are the connected components of the contact graph (curves touch if closer than CONTACT),
// computed here from the exported curves like clipasso3d/source/connectivity.py.
import * as THREE from 'three';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { loadRuns, controlPoints, curvesToSegments, Orbit, FOV_DEG, SEGMENTS } from './sketchgrid.js';

const CONTACT = 2 * 0.008 * 1.25;      // contact distance of training (two stroke radii), small tolerance for sampling
const N_PTS = 32;                       // points per curve for the distances (as in connectivity_stats)
const PALETTE = ['#e8743b', '#2f6fdf', '#2a9d8f', '#c0398b', '#7b5ea7', '#d6a21a', '#3d8f4a', '#d6453d',
  '#4a90a8', '#a8642a', '#8c9a2a', '#6b6fd6'];
const GOLD = new THREE.Color('#b8862b'), GLOW = new THREE.Color('#fff2c4');
const ROWS = [
  { key: 'off', label: 'not connected', sub: 'no constraint' },
  { key: 'loss01', label: 'connected', sub: 'loss 0.1 + weld' },
];

function el(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.append(e);
  return e;
}

function bezier(cp, c, t, out) {
  const o = c * 12, u = 1 - t, b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
  for (let k = 0; k < 3; k++) out[k] = b0 * cp[o + k] + b1 * cp[o + 3 + k] + b2 * cp[o + 6 + k] + b3 * cp[o + 9 + k];
  return out;
}

// Contact graph of the curves: adjacency lists and component labels.
function contactGraph(cp) {
  const C = cp.length / 12, P = new Float32Array(C * N_PTS * 3), lo = [], hi = [], p = [0, 0, 0];
  for (let c = 0; c < C; c++) {
    const l = [Infinity, Infinity, Infinity], h = [-Infinity, -Infinity, -Infinity];
    for (let s = 0; s < N_PTS; s++) {
      bezier(cp, c, s / (N_PTS - 1), p);
      P.set(p, (c * N_PTS + s) * 3);
      for (let k = 0; k < 3; k++) { l[k] = Math.min(l[k], p[k]); h[k] = Math.max(h[k], p[k]); }
    }
    lo.push(l); hi.push(h);
  }
  const adj = Array.from({ length: C }, () => []), e2 = CONTACT * CONTACT;
  for (let i = 0; i < C; i++) for (let j = i + 1; j < C; j++) {
    if (lo[i].some((v, k) => v > hi[j][k] + CONTACT) || lo[j].some((v, k) => v > hi[i][k] + CONTACT)) continue;
    let touch = false;
    for (let a = 0; a < N_PTS && !touch; a++) {
      const ia = (i * N_PTS + a) * 3;
      for (let b = 0; b < N_PTS; b++) {
        const jb = (j * N_PTS + b) * 3, dx = P[ia] - P[jb], dy = P[ia + 1] - P[jb + 1], dz = P[ia + 2] - P[jb + 2];
        if (dx * dx + dy * dy + dz * dz < e2) { touch = true; break; }
      }
    }
    if (touch) { adj[i].push(j); adj[j].push(i); }
  }
  const label = new Int32Array(C).fill(-1);
  let n = 0;
  for (let s = 0; s < C; s++) {
    if (label[s] >= 0) continue;
    const q = [s]; label[s] = n;
    while (q.length) { const i = q.pop(); for (const j of adj[i]) if (label[j] < 0) { label[j] = n; q.push(j); } }
    n++;
  }
  // order components by size (largest first) so colors are stable
  const sizes = Array.from({ length: n }, (_, k) => [k, 0]);
  label.forEach((k) => sizes[k][1]++);
  sizes.sort((a, b) => b[1] - a[1]);
  const rank = new Int32Array(n); sizes.forEach(([k], r) => { rank[k] = r; });
  return { adj, label: label.map((k) => rank[k]), n };
}

// Hop distance of every curve from the lowest curve (where the light wave starts).
function hops(cp, adj) {
  const C = adj.length, d = new Float32Array(C).fill(Infinity);
  let start = 0, zmin = Infinity;
  for (let c = 0; c < C; c++) { const z = Math.min(cp[c * 12 + 2], cp[c * 12 + 11]); if (z < zmin) { zmin = z; start = c; } }
  d[start] = 0;
  const q = [start];
  while (q.length) { const i = q.shift(); for (const j of adj[i]) if (d[j] === Infinity) { d[j] = d[i] + 1; q.push(j); } }
  const max = Math.max(1, ...[...d].filter(Number.isFinite));
  return d.map((v) => (Number.isFinite(v) ? v / max : 1));
}

// small round sprite for the sparkles
let sparkTex = null;
function sparkTexture() {
  if (!sparkTex) {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d'), r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.25, 'rgba(255,240,200,0.9)'); r.addColorStop(1, 'rgba(255,220,150,0)');
    g.fillStyle = r; g.fillRect(0, 0, 64, 64);
    g.strokeStyle = 'rgba(255,255,255,0.9)'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(32, 6); g.lineTo(32, 58); g.moveTo(6, 32); g.lineTo(58, 32); g.stroke();
    sparkTex = new THREE.CanvasTexture(c);
  }
  return sparkTex;
}

export class ConnectCompare {
  constructor(root, { objects, lines = ['5', '20', '50', '200'], initialLines = '20' }) {
    this.objects = objects; this.n = initialLines;
    root.classList.add('sg-root', 'cc');
    const top = el('div', 'sg-toolbar', root);
    const btns = el('div', 'buttons has-addons', top);
    for (const n of lines) {
      const b = el('button', 'button is-small', btns, `${n} lines`);
      if (n === this.n) b.classList.add('is-dark', 'is-selected');
      b.addEventListener('click', () => {
        btns.querySelectorAll('button').forEach((x) => x.classList.remove('is-dark', 'is-selected'));
        b.classList.add('is-dark', 'is-selected');
        this.n = n; this.load();
      });
    }

    this.stage = el('div', 'sg-stage', root);
    const grid = el('div', 'sg-grid cc-grid', this.stage);
    grid.style.gridTemplateColumns = `auto repeat(${objects.length}, 1fr)`;
    el('div', 'sg-corner', grid);
    objects.forEach((o) => el('div', 'sg-colhead', grid, o.label));
    this.cells = {};
    for (const r of ROWS) {
      const head = el('div', `sg-rowhead cc-rowhead cc-${r.key}`, grid, r.label);
      el('span', 'sg-sub', head, r.sub);
      for (const o of objects) {
        const cell = el('div', 'sg-cell cc-cell', grid);
        const badge = el('div', 'cc-badge', cell);
        this.cells[`${r.key}|${o.key}`] = { cell, badge, view: null };
      }
    }

    // renderer: one canvas, a viewport per cell, one shared orbit
    this.canvas = el('canvas', 'sg-canvas', null);
    this.stage.prepend(this.canvas);
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setScissorTest(true);
    this.camera = new THREE.PerspectiveCamera(FOV_DEG, 1, 0.05, 50);
    this.orbit = new Orbit(this.stage);
    this.clock = new THREE.Clock(); this.time = 0; this.visible = false;
    new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; }, { rootMargin: '200px' }).observe(this.stage);
    this.renderer.setAnimationLoop(() => this.frame());
    this.load();
  }

  async load() {
    const n = this.n;
    await Promise.all(this.objects.map(async (o) => {
      const runs = await loadRuns(o.key, 'connect');
      if (this.n !== n) return;
      for (const r of ROWS) {
        const c = this.cells[`${r.key}|${o.key}`];
        const cp = runs && controlPoints(runs, `${r.key}|${n}`);
        this.setView(c, cp, r.key === 'loss01');
      }
    }));
  }

  setView(c, cp, connected) {
    if (c.view) { c.view.lines.geometry.dispose(); c.view = null; }
    if (!cp) { c.badge.textContent = 'coming soon'; return; }
    const { adj, label, n } = contactGraph(cp);
    const C = cp.length / 12, segs = C * SEGMENTS;
    const pos = curvesToSegments(cp), colors = new Float32Array(segs * 6);
    const geo = new LineSegmentsGeometry();
    geo.setPositions(pos); geo.setColors(colors);
    const mat = new LineMaterial({ vertexColors: true, linewidth: 1.7 });
    const lines = new LineSegments2(geo, mat);
    const scene = new THREE.Scene();
    scene.add(lines);
    // keep an untouched copy: the geometry uses `pos` itself as its buffer and the drift rewrites it
    const view = { scene, lines, mat, pos: pos.slice(), label, n, connected, C, phase: Math.random() * 6 };
    view.pieceColor = Array.from({ length: n }, (_, k) => new THREE.Color(connected || n === 1 ? GOLD : PALETTE[k % PALETTE.length]));
    view.piecePhase = Float32Array.from({ length: n }, () => Math.random());
    view.hop = connected ? hops(cp, adj) : new Float32Array(C);
    // sparkles at random points of the curves
    const S = Math.min(60, 8 + C * 2), sp = new Float32Array(S * 3);
    view.sparkSeeds = Float32Array.from({ length: S }, () => Math.random());
    const p = [0, 0, 0];
    for (let k = 0; k < S; k++) sp.set(bezier(cp, Math.floor(Math.random() * C), Math.random(), p), k * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(sp, 3));
    view.cp = cp; view.sparkCycle = new Int32Array(S).fill(-1);
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(S * 3).fill(1), 3));
    view.sparks = new THREE.Points(g, new THREE.PointsMaterial({
      map: sparkTexture(), size: 0.11, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, vertexColors: true,
    }));
    view.sparkMats = view.sparks.material;
    scene.add(view.sparks);
    if (!connected) {
      // pieces drift apart a little, each straight away from the sketch's center
      const cen = Array.from({ length: n }, () => [0, 0, 0, 0]), all = [0, 0, 0];
      for (let c = 0; c < C; c++) for (let k = 0; k < 4; k++) for (let a = 0; a < 3; a++) {
        const v = cp[c * 12 + k * 3 + a]; cen[label[c]][a] += v; all[a] += v / (4 * C);
      }
      for (let c = 0; c < C; c++) cen[label[c]][3] += 4;
      view.drift = cen.map(([x, y, z, w]) => {
        const d = new THREE.Vector3(x / w - all[0], y / w - all[1], z / w - all[2]);
        return d.lengthSq() > 1e-6 ? d.normalize() : new THREE.Vector3(0, 0, 1);
      });
    }
    c.badge.textContent = n === 1 ? '1 piece' : `${n} pieces`;
    c.badge.classList.toggle('is-one', n === 1);
    c.view = view;
  }

  animate(v, t) {
    const geo = v.lines.geometry, arr = geo.attributes.instanceColorStart.data.array, col = new THREE.Color();
    // shimmer: connected = one wave running through the contact graph; separate = every piece on its own rhythm
    for (let c = 0; c < v.C; c++) {
      const k = v.label[c], base = v.pieceColor[k], ph = v.connected ? v.hop[c] : v.piecePhase[k];
      const rate = v.connected ? 0.35 : 0.22 + 0.12 * v.piecePhase[k];
      for (let s = 0; s < SEGMENTS; s++) {
        for (let e = 0; e < 2; e++) {
          const u = (s + e) / SEGMENTS;
          const wave = Math.max(0, Math.cos(2 * Math.PI * (rate * t - ph - 0.08 * u)));
          const tw = 0.12 * Math.max(0, Math.sin(9 * u + 2 * t + c));
          col.copy(base).lerp(GLOW, Math.min(1, wave ** 6 * 0.85 + tw));
          arr.set([col.r, col.g, col.b], ((c * SEGMENTS + s) * 2 + e) * 3);
        }
      }
    }
    geo.attributes.instanceColorStart.data.needsUpdate = true;

    // separate pieces detach: a slow, small outward drift (about 2% of the object's size)
    const off = v.connected || v.n === 1 ? null
      : v.drift.map((d, k) => d.clone().multiplyScalar(0.022 * (0.5 - 0.5 * Math.cos(2 * Math.PI * (t / 7 + 0.1 * v.piecePhase[k])))));
    if (off) {
      const pa = geo.attributes.instanceStart.data.array, base = v.pos;
      for (let c = 0; c < v.C; c++) {
        const o = off[v.label[c]];
        for (let s = 0; s < SEGMENTS * 2; s++) {
          const i = (c * SEGMENTS * 2 + s) * 3;
          pa[i] = base[i] + o.x; pa[i + 1] = base[i + 1] + o.y; pa[i + 2] = base[i + 2] + o.z;
        }
      }
      geo.attributes.instanceStart.data.needsUpdate = true;
    }

    // sparkles: each jumps to a new random point once per own cycle, tinted like the piece it lands on
    const pa = v.sparks.geometry.attributes.position, pc = v.sparks.geometry.attributes.color, p = [0, 0, 0];
    v.sparkSeeds.forEach((seed, k) => {
      const cyc = Math.floor(t / 1.1 + seed * 7);
      if (cyc === v.sparkCycle[k]) return;
      v.sparkCycle[k] = cyc;
      const c = Math.floor(Math.random() * v.C), o = off ? off[v.label[c]] : null;
      bezier(v.cp, c, Math.random(), p);
      if (o) { p[0] += o.x; p[1] += o.y; p[2] += o.z; }
      pa.array.set(p, k * 3);
      col.copy(v.pieceColor[v.label[c]]).lerp(GLOW, 0.6);
      pc.array.set([col.r, col.g, col.b], k * 3);
      pa.needsUpdate = pc.needsUpdate = true;
    });
    v.sparkMats.size = 0.075 + 0.03 * Math.sin(3.1 * t + v.phase);
    v.sparkMats.opacity = 0.7 + 0.3 * Math.sin(2.3 * t + v.phase);
  }

  frame() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    this.time += dt;
    this.orbit.update(dt, this.camera);
    if (!this.visible) return;
    const r = this.stage.getBoundingClientRect(), w = Math.round(r.width), h = Math.round(r.height), pr = this.renderer.getPixelRatio();
    if (this.canvas.width !== Math.round(w * pr) || this.canvas.height !== Math.round(h * pr)) this.renderer.setSize(w, h, true);
    this.renderer.setScissor(0, 0, w, h); this.renderer.setClearColor(0xffffff, 0); this.renderer.clear();
    for (const { cell, view } of Object.values(this.cells)) {
      if (!view) continue;
      const c = cell.getBoundingClientRect();
      if (c.width < 2 || c.bottom < r.top || c.top > r.bottom) continue;
      this.animate(view, this.time);
      const x = c.left - r.left, y = r.bottom - c.bottom;
      this.renderer.setViewport(x, y, c.width, c.height); this.renderer.setScissor(x, y, c.width, c.height);
      view.mat.resolution.set(c.width, c.height);
      view.mat.linewidth = THREE.MathUtils.clamp(c.width / 95, 1.2, 2.4);
      this.camera.aspect = c.width / c.height; this.camera.updateProjectionMatrix();
      this.renderer.render(view.scene, this.camera);
    }
  }
}
