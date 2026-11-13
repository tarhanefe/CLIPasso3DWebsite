// Interactive 3D sketch grids for the CLIPasso3D page.
//
// Every grid draws all of its cells with one WebGL canvas (scissored viewports) and one shared
// camera: dragging any cell orbits all of them; when released they keep circling the object at
// a 20 degree elevation. Data comes from static/data (written by clipasso3d/tools/export_web.py).
import * as THREE from 'three';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const DATA = './static/data';
const FOV_DEG = 39.6;                 // horizontal FOV of the training cameras
const DIST = 3.25 / 1.12;             // training camera distance, zoomed in a little (unit sphere still fits)
const ELEV = THREE.MathUtils.degToRad(20);
const SPIN = (2 * Math.PI) / 18;      // one turn every 18 s
const SEGMENTS = 24;                  // line segments per Bezier curve

// ------------------------------------------------------------------ data

const jsonCache = new Map();
const binCache = new Map();

function fetchJSON(url) {
  if (!jsonCache.has(url)) {
    jsonCache.set(url, fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return jsonCache.get(url);
}

function fetchInt16(url) {
  if (!binCache.has(url)) {
    binCache.set(url, fetch(url).then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((b) => (b ? new Int16Array(b) : null)).catch(() => null));
  }
  return binCache.get(url);
}

export function loadIndex() {
  return fetchJSON(`${DATA}/index.json`);
}

// One experiment family of one object: { meta, finals, history (lazy) }.
// meta.runs[key] = { curves, snapshots: [batches...], final_offset, history_offset }
export async function loadRuns(objectKey, family = 'ablation') {
  const base = `${DATA}/${family}/${objectKey}`;
  const meta = await fetchJSON(`${base}/runs.json`);
  if (!meta) return null;
  const finals = await fetchInt16(`${base}/finals.bin`);
  return {
    meta,
    finals,
    history: () => fetchInt16(`${base}/history.bin`),
  };
}

// Control points (C*4*3 floats) of a run at a snapshot index (or the final state).
function controlPoints(runs, key, snapshot = null, history = null) {
  const run = runs.meta.runs[key];
  if (!run) return null;
  const n = run.curves * 12;
  const scale = runs.meta.scale / 32767;
  let src, off;
  if (snapshot === null || !history) {
    src = runs.finals; off = run.final_offset;
  } else {
    src = history; off = run.history_offset + snapshot * n;
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = src[off + i] * scale;
  return out;
}

// Sample cubic Bezier curves into line-segment endpoints (x0,y0,z0,x1,y1,z1 per segment).
function curvesToSegments(cp) {
  const C = cp.length / 12;
  const pos = new Float32Array(C * SEGMENTS * 6);
  let k = 0;
  for (let c = 0; c < C; c++) {
    const o = c * 12;
    let px = cp[o], py = cp[o + 1], pz = cp[o + 2];
    for (let s = 1; s <= SEGMENTS; s++) {
      const t = s / SEGMENTS, u = 1 - t;
      const b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
      const x = b0 * cp[o] + b1 * cp[o + 3] + b2 * cp[o + 6] + b3 * cp[o + 9];
      const y = b0 * cp[o + 1] + b1 * cp[o + 4] + b2 * cp[o + 7] + b3 * cp[o + 10];
      const z = b0 * cp[o + 2] + b1 * cp[o + 5] + b2 * cp[o + 8] + b3 * cp[o + 11];
      pos[k++] = px; pos[k++] = py; pos[k++] = pz; pos[k++] = x; pos[k++] = y; pos[k++] = z;
      px = x; py = y; pz = z;
    }
  }
  return pos;
}

// ------------------------------------------------------------------ theme

export const theme = {
  dark: false,   // false: black strokes on white, true: white strokes on black
  listeners: [],
  set(dark) {
    this.dark = dark;
    document.documentElement.classList.toggle('sketch-dark', dark);
    this.listeners.forEach((f) => f(dark));
  },
};

// ------------------------------------------------------------------ orbit

class Orbit {
  constructor(el) {
    this.az = Math.PI / 4;
    this.el = ELEV;
    this.dragging = false;
    el.addEventListener('pointerdown', (e) => {
      this.dragging = true; this.x = e.clientX; this.y = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      this.az -= (e.clientX - this.x) * 0.008;
      this.el = THREE.MathUtils.clamp(this.el + (e.clientY - this.y) * 0.008, -1.2, 1.4);
      this.x = e.clientX; this.y = e.clientY;
    });
    const up = () => { this.dragging = false; };
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  }

  update(dt, camera, lookZ = 0, dist = DIST) {
    if (!this.dragging) {
      this.az += SPIN * dt;                                  // keep circling ...
      this.el += (ELEV - this.el) * (1 - Math.exp(-2 * dt)); // ... and settle back to 20 degrees
    }
    const ce = Math.cos(this.el);
    camera.position.set(dist * ce * Math.cos(this.az), dist * ce * Math.sin(this.az), lookZ + dist * Math.sin(this.el));
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, lookZ);
  }
}

// ------------------------------------------------------------------ multi-view renderer

// Soft round shadow (white texture, tinted per theme) under floating objects.
let shadowTex = null;
function shadowTexture() {
  if (!shadowTex) {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const g = c.getContext('2d');
    const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.5, 'rgba(255,255,255,0.45)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 128);
    shadowTex = new THREE.CanvasTexture(c);
  }
  return shadowTex;
}

// Lowest z of a set of control points (the curves stay within their control polygon's hull).
function minZ(cp) {
  let z = Infinity;
  for (let i = 2; i < cp.length; i += 3) z = Math.min(z, cp[i]);
  return z;
}

class MultiView {
  // transparent: no per-view background (objects drawn on the page itself)
  // float: every view bobs gently above a soft shadow, each with its own phase
  // lookZ: height the camera aims at (below 0 leaves room for the shadows); dist: camera distance
  constructor(container, { transparent = false, float = false, lineScale = 1, lookZ = 0, dist = DIST } = {}) {
    this.container = container;
    this.transparent = transparent; this.float = float; this.lineScale = lineScale;
    this.lookZ = lookZ; this.dist = dist;
    this.time = 0;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'sg-canvas';
    container.prepend(this.canvas);
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setScissorTest(true);
    this.camera = new THREE.PerspectiveCamera(FOV_DEG, 1, 0.05, 50);
    this.orbit = new Orbit(container);
    this.views = [];          // { el, scene, material? }
    this.visible = false;
    this.onFrame = null;
    new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; }, { rootMargin: '200px' })
      .observe(container);
    this.clock = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  addView(el) {
    const scene = new THREE.Scene();
    const pivot = new THREE.Group();
    scene.add(pivot);
    const view = { el, scene, pivot, lines: null, material: null, floor: null, phase: this.views.length * 1.7 };
    if (this.float) {
      view.shadow = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 1.7),
        new THREE.MeshBasicMaterial({ map: shadowTexture(), transparent: true, depthWrite: false }));
      view.shadow.visible = false;
      scene.add(view.shadow);
    }
    this.views.push(view);
    return view;
  }

  setLines(view, cp) {
    if (view.lines) {
      view.pivot.remove(view.lines);
      view.lines.geometry.dispose();
      view.lines = null;
    }
    view.floor = cp ? minZ(cp) : null;
    if (!cp) return;
    if (!view.material) {
      view.material = new LineMaterial({ color: theme.dark ? 0xffffff : 0x111111, linewidth: 1.6 });
    }
    const geo = new LineSegmentsGeometry();
    geo.setPositions(curvesToSegments(cp));
    view.lines = new LineSegments2(geo, view.material);
    view.pivot.add(view.lines);
  }

  setObject(view, object3d) {
    if (view.object) view.pivot.remove(view.object);
    view.object = object3d;
    view.floor = null;
    if (object3d) {
      view.pivot.add(object3d);
      object3d.updateMatrixWorld(true);
      view.floor = new THREE.Box3().setFromObject(object3d).min.z;
    }
  }

  // Bob every view up and down; all shadows share one floor below the lowest object.
  floatViews() {
    const floors = this.views.map((v) => v.floor).filter((z) => z !== null && Number.isFinite(z));
    const floor = (floors.length ? Math.min(...floors) : -1) - 0.1;
    for (const v of this.views) {
      const h = 0.06 * Math.sin(1.1 * this.time + v.phase);
      v.pivot.position.z = h;
      v.shadow.visible = v.floor !== null;
      v.shadow.position.z = floor;
      v.shadow.scale.setScalar(1 - 1.5 * h);
      v.shadow.material.color.set(theme.dark ? 0xffffff : 0x000000);
      v.shadow.material.opacity = (theme.dark ? 0.12 : 0.2) * (1 - 4 * h);
    }
  }

  frame() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    this.time += dt;
    this.orbit.update(dt, this.camera, this.lookZ, this.dist);
    if (this.float) this.floatViews();
    if (this.onFrame) this.onFrame(dt);
    if (!this.visible) return;
    const r = this.container.getBoundingClientRect();
    const w = Math.round(r.width), h = Math.round(r.height);
    const pr = this.renderer.getPixelRatio();
    if (this.canvas.width !== Math.round(w * pr) || this.canvas.height !== Math.round(h * pr)) {
      this.renderer.setSize(w, h, true);
    }
    const bg = new THREE.Color(theme.dark ? 0x111111 : 0xffffff);
    this.renderer.setScissor(0, 0, w, h);
    this.renderer.setClearColor(bg, 0);
    this.renderer.clear();
    for (const v of this.views) {
      const c = v.el.getBoundingClientRect();
      const x = c.left - r.left, y = r.bottom - c.bottom;   // WebGL origin is bottom-left
      if (c.width < 2 || c.bottom < r.top || c.top > r.bottom) continue;
      this.renderer.setViewport(x, y, c.width, c.height);
      this.renderer.setScissor(x, y, c.width, c.height);
      if (!this.transparent) {
        this.renderer.setClearColor(v.isGT ? 0xffffff : bg, 1);
        this.renderer.clear();
      }
      if (v.material) {
        v.material.resolution.set(c.width, c.height);
        v.material.linewidth = THREE.MathUtils.clamp(c.width / 120, 1.0, 2.2) * this.lineScale;
        v.material.color.set(theme.dark ? 0xffffff : 0x111111);
      }
      this.camera.aspect = c.width / c.height;
      this.camera.updateProjectionMatrix();
      this.renderer.render(v.scene, this.camera);
    }
  }
}

// ------------------------------------------------------------------ ground truth meshes

const gltfLoader = new GLTFLoader();
const gtCache = new Map();

function loadGT(objectKey) {
  if (!gtCache.has(objectKey)) {
    gtCache.set(objectKey, new Promise((resolve) => {
      gltfLoader.load(`${DATA}/gt/${objectKey}.glb`, (g) => {
        const root = new THREE.Group();
        g.scene.rotation.x = Math.PI / 2;   // glTF is Y-up, the sketches are Z-up (Blender frame)
        root.add(g.scene);
        root.add(new THREE.HemisphereLight(0xffffff, 0x888888, 2.2));
        const sun = new THREE.DirectionalLight(0xffffff, 1.6);
        sun.position.set(2, -3, 4);
        root.add(sun);
        resolve(root);
      }, undefined, () => resolve(null));
    }));
  }
  return gtCache.get(objectKey).then((r) => (r ? r.clone() : null));
}

// ------------------------------------------------------------------ helpers

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function objectPicker(objects, onPick, selected) {
  const bar = el('div', 'sg-picker buttons is-centered');
  for (const o of objects) {
    const b = el('button', 'button is-small is-rounded', o.label);
    if (o.key === selected) b.classList.add('is-dark');
    b.addEventListener('click', () => {
      bar.querySelectorAll('button').forEach((x) => x.classList.remove('is-dark'));
      b.classList.add('is-dark');
      onPick(o.key);
    });
    bar.append(b);
  }
  return bar;
}

// Row label with an optional second line (e.g. the loss ratio of an abstraction level).
function rowHead(r, tag = 'div', cls = 'sg-rowhead') {
  const h = el(tag, cls, r.label);
  if (r.sub) h.append(el('span', 'sg-sub', r.sub));
  return h;
}

function pending(view, text) {
  view.el.classList.add('sg-pending');
  view.el.dataset.pending = text;
}

// ------------------------------------------------------------------ rows x columns grid
// Used for the lines x abstraction ablation and the connectivity ablation.
// rows/cols: [{key,label}], run key = `${row}|${col}`.

export class RunGrid {
  constructor(root, { family, rows, cols, objects, initial, rowTitle, colTitle }) {
    this.family = family; this.rows = rows; this.cols = cols; this.objects = objects;
    this.mode = 'done'; this.t = 1; this.playing = true;
    root.classList.add('sg-root');

    // top: done / train switch
    const top = el('div', 'sg-toolbar');
    const modes = el('div', 'buttons has-addons sg-modes');
    for (const [m, label] of [['done', 'Done'], ['train', 'Training']]) {
      const b = el('button', 'button is-small', label);
      if (m === this.mode) b.classList.add('is-dark', 'is-selected');
      b.addEventListener('click', () => {
        modes.querySelectorAll('button').forEach((x) => x.classList.remove('is-dark', 'is-selected'));
        b.classList.add('is-dark', 'is-selected');
        this.setMode(m);
      });
      modes.append(b);
    }
    top.append(modes);
    root.append(top);

    // grid of cells: [GT] [row labels | cells...]
    this.stage = el('div', 'sg-stage');
    const gtWrap = el('div', 'sg-gt');
    gtWrap.append(el('div', 'sg-label', 'ground truth'));
    const gtCell = el('div', 'sg-cell sg-gt-cell');
    gtWrap.append(gtCell);
    const grid = el('div', 'sg-grid');
    grid.style.gridTemplateColumns = `auto repeat(${cols.length}, 1fr)`;
    grid.append(el('div', 'sg-corner', rowTitle || ''));
    cols.forEach((c) => grid.append(el('div', 'sg-colhead', c.label)));
    this.cells = {};
    for (const r of rows) {
      grid.append(rowHead(r));
      for (const c of cols) {
        const cell = el('div', 'sg-cell');
        grid.append(cell);
        this.cells[`${r.key}|${c.key}`] = cell;
      }
    }
    this.stage.append(gtWrap, grid);
    root.append(this.stage);

    // timeline (training mode)
    this.timeline = el('div', 'sg-timeline is-hidden');
    this.playBtn = el('button', 'button is-small is-rounded', 'Pause');
    this.slider = el('input', 'slider is-fullwidth is-small is-circle');
    Object.assign(this.slider, { type: 'range', min: 0, max: 1000, value: 1000 });
    this.stepLabel = el('span', 'sg-step', '');
    this.playBtn.addEventListener('click', () => {
      this.playing = !this.playing;
      this.playBtn.textContent = this.playing ? 'Pause' : 'Play';
    });
    this.slider.addEventListener('input', () => {
      this.playing = false; this.playBtn.textContent = 'Play';
      this.t = this.slider.value / 1000; this.refreshTraining();
    });
    this.timeline.append(this.playBtn, this.slider, this.stepLabel);
    root.append(this.timeline);

    root.append(objectPicker(objects, (k) => this.setObject(k), initial));
    if (colTitle) root.append(el('p', 'sg-caption', colTitle));

    // renderer
    this.mv = new MultiView(this.stage);
    this.gtView = this.mv.addView(gtCell); this.gtView.isGT = true;
    this.views = {};
    for (const [k, cell] of Object.entries(this.cells)) this.views[k] = this.mv.addView(cell);
    this.mv.onFrame = (dt) => this.tick(dt);
    this.setObject(initial);
  }

  async setObject(key) {
    this.object = key;
    for (const v of Object.values(this.views)) { this.mv.setLines(v, null); pending(v, 'loading'); }
    loadGT(key).then((o) => { if (this.object === key) this.mv.setObject(this.gtView, o); });
    const runs = await loadRuns(key, this.family);
    if (this.object !== key) return;
    this.runs = runs; this.history = null;
    if (!runs) {
      for (const v of Object.values(this.views)) pending(v, 'results coming soon');
      return;
    }
    this.maxBatch = Math.max(...Object.values(runs.meta.runs).map((r) => r.snapshots[r.snapshots.length - 1]));
    if (this.mode === 'train') await this.loadHistory();
    this.refresh();
  }

  async loadHistory() {
    const key = this.object;
    const h = await this.runs.history();
    if (this.object === key) this.history = h;
  }

  async setMode(m) {
    this.mode = m;
    this.timeline.classList.toggle('is-hidden', m !== 'train');
    if (m === 'train') {
      this.t = 0; this.playing = true; this.playBtn.textContent = 'Pause';
      if (this.runs && !this.history) await this.loadHistory();
    }
    this.refresh();
  }

  refresh() {
    if (!this.runs) return;
    if (this.mode === 'train' && this.history) { this.lastIdx = {}; this.refreshTraining(); return; }
    for (const [k, v] of Object.entries(this.views)) {
      const cp = controlPoints(this.runs, k);
      v.el.classList.toggle('sg-pending', !cp);
      if (!cp) v.el.dataset.pending = 'training ...';
      this.mv.setLines(v, cp);
    }
  }

  refreshTraining(t = this.t) {
    if (!this.runs || !this.history) return;
    const batch = Math.min(t, 1) * this.maxBatch;
    this.stepLabel.textContent = `step ${Math.round(batch)} / ${this.maxBatch}`;
    this.slider.value = Math.round(Math.min(t, 1) * 1000);
    for (const [k, v] of Object.entries(this.views)) {
      const run = this.runs.meta.runs[k];
      if (!run) continue;
      let i = 0;
      while (i + 1 < run.snapshots.length && run.snapshots[i + 1] <= batch) i++;
      if (this.lastIdx[k] === i) continue;
      this.lastIdx[k] = i;
      v.el.classList.remove('sg-pending');
      this.mv.setLines(v, controlPoints(this.runs, k, i, this.history));
    }
  }

  tick(dt) {
    if (this.mode !== 'train' || !this.playing || !this.history) return;
    this.t += dt / 14;                       // whole training in 14 s, then a short pause at the end
    if (this.t > 1.12) this.t = 0;
    this.refreshTraining();
  }
}

// ------------------------------------------------------------------ banner
// The object and its sketches with more and more lines, floating side by side on the page.

export class Banner {
  constructor(root, { object, family = 'ablation', level = 'normal', lines = [5, 20, 50, 200] }) {
    root.classList.add('sg-root');
    this.stage = el('div', 'sg-stage sg-banner-stage');
    const row = el('div', 'sg-banner-row');
    const add = (title) => {
      const cell = el('div', 'sg-cell');
      cell.title = title;
      row.append(cell);
      return cell;
    };
    const gtCell = add('object');
    const cells = lines.map((n) => [`${level}|${n}`, add(`${n} lines`)]);
    this.stage.append(row);
    root.append(this.stage);

    this.mv = new MultiView(this.stage, { transparent: true, float: true, lineScale: 1.25, lookZ: -0.25, dist: DIST * 1.12 });
    const gtView = this.mv.addView(gtCell);
    const views = cells.map(([key, cell]) => [key, this.mv.addView(cell)]);
    loadGT(object).then((o) => this.mv.setObject(gtView, o));
    loadRuns(object, family).then((runs) => {
      for (const [key, v] of views) {
        const cp = runs && controlPoints(runs, key);
        if (!cp) pending(v, 'coming soon');
        this.mv.setLines(v, cp);
      }
    });
  }
}

// ------------------------------------------------------------------ gallery (one cell per object)

export class Gallery {
  constructor(root, { objects, rows, cols, initialRow, initialCol, columns = 5 }) {
    this.objects = objects;
    this.showGT = false;
    root.classList.add('sg-root');
    const top = el('div', 'sg-toolbar');
    const mk = (opts, init, onChange) => {
      const sel = el('div', 'select is-small');
      const s = el('select');
      opts.forEach((o) => { const op = el('option', '', o.sub ? `${o.label} (${o.sub})` : o.label); op.value = o.key; if (o.key === init) op.selected = true; s.append(op); });
      s.addEventListener('change', () => onChange(s.value));
      sel.append(s);
      return sel;
    };
    this.row = initialRow; this.col = initialCol;
    top.append(el('span', 'sg-toolbar-label', 'abstraction'), mk(rows, initialRow, (v) => { this.row = v; this.refresh(); }));
    top.append(el('span', 'sg-toolbar-label', 'lines'), mk(cols, initialCol, (v) => { this.col = v; this.refresh(); }));
    const gt = el('button', 'button is-small is-rounded', 'Show ground truth');
    gt.addEventListener('click', () => {
      this.showGT = !this.showGT;
      gt.textContent = this.showGT ? 'Show sketches' : 'Show ground truth';
      gt.classList.toggle('is-dark', this.showGT);
      this.refresh();
    });
    top.append(gt);
    root.append(top);

    this.stage = el('div', 'sg-stage');
    const grid = el('div', 'sg-gallery');
    grid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
    this.cells = {};
    for (const o of objects) {
      const wrap = el('div', 'sg-gallery-item');
      const cell = el('div', 'sg-cell');
      wrap.append(cell, el('div', 'sg-label', o.label));
      grid.append(wrap);
      this.cells[o.key] = cell;
    }
    this.stage.append(grid);
    root.append(this.stage);
    this.mv = new MultiView(this.stage);
    this.views = {};
    for (const o of objects) this.views[o.key] = this.mv.addView(this.cells[o.key]);
    this.refresh();
  }

  async refresh() {
    const key = `${this.row}|${this.col}`;
    await Promise.all(this.objects.map(async (o) => {
      const v = this.views[o.key];
      v.isGT = this.showGT;
      if (this.showGT) {
        this.mv.setLines(v, null);
        v.el.classList.remove('sg-pending');
        this.mv.setObject(v, await loadGT(o.key));
        return;
      }
      this.mv.setObject(v, null);
      const runs = await loadRuns(o.key, 'ablation');
      const cp = runs && controlPoints(runs, key);
      v.el.classList.toggle('sg-pending', !cp);
      if (!cp) v.el.dataset.pending = runs ? 'training ...' : 'results coming soon';
      this.mv.setLines(v, cp);
    }));
  }
}

// ------------------------------------------------------------------ metrics heat maps

export async function metricsTables(root) {
  const m = await fetchJSON(`${DATA}/metrics.json`);
  if (!m) { root.append(el('p', 'sg-caption', 'Quantitative results will appear here when all runs are finished.')); return; }
  const wrap = el('div', 'columns is-multiline is-centered');
  for (const metric of m.metrics) {
    const col = el('div', 'column is-full');
    const t = el('table', 'table is-narrow sg-metric');
    const cap = el('caption', '', `${metric.label} ${metric.higher_is_better ? '(higher is better)' : '(lower is better)'}`);
    t.append(cap);
    const head = el('tr');
    head.append(el('th', '', ''));
    m.cols.forEach((c) => head.append(el('th', '', c.label)));
    t.append(head);
    const vals = m.rows.flatMap((r) => m.cols.map((c) => metric.values[`${r.key}|${c.key}`])).filter((x) => x != null);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    for (const r of m.rows) {
      const tr = el('tr');
      tr.append(rowHead(r, 'th', ''));
      for (const c of m.cols) {
        const v = metric.values[`${r.key}|${c.key}`];
        const td = el('td', '', v == null ? '–' : v.toFixed(3));
        if (v != null) {
          let q = hi > lo ? (v - lo) / (hi - lo) : 0.5;
          if (!metric.higher_is_better) q = 1 - q;
          td.style.background = `rgba(50, 115, 220, ${0.08 + 0.55 * q})`;
        }
        tr.append(td);
      }
      t.append(tr);
    }
    col.append(t);
    wrap.append(col);
  }
  root.append(wrap);
}
