// Animated training loop on a real run: the 3D curves, 8 sampled training cameras, the sketch and photo of one
// view, CLIP and the two losses, the gradient flowing back, and the curves moving to the next saved snapshot.
// Curves, step numbers and the red update arrows come from the run's training history (static/data).
import * as THREE from 'three';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { loadRuns, loadGT, controlPoints, curvesToSegments } from './sketchgrid.js';
import { rich } from './bezierfig.js';

const ORANGE = 0xe8743b, RED = 0xd6453d, GREY = 0xb9bfca, INK = 0x111111;
const VIEW_DIST = 3.25 / 1.12;   // camera distance of the sketch / photo cells (as in the grids)
const RING = 1.55;               // radius at which the camera frustums are drawn in the 3D scene
const N_CAMS = 24, N_PICK = 8;
const FOV = 39.6;
const N_STEPS = 22;             // the animation walks through the run in about this many steps

const STAGES = [
  { key: 'views', label: 'pick 8 views', dur: 1.1,
    text: '8 of the training cameras are drawn at random for this step.' },
  { key: 'render', label: 'render', dur: 1.3,
    text: 'The curves are projected into each camera and splatted into a sketch \\(\\mathbf{S}_v\\); the photo \\(\\mathbf{I}_v\\) of the same view comes from the dataset.' },
  { key: 'clip', label: 'compare in CLIP', dur: 1.6,
    text: 'CLIP compares sketch and photo: intermediate layers give the geometric loss, the final embedding the semantic loss.' },
  { key: 'back', label: 'backpropagate', dur: 1.3,
    text: 'The gradient of \\(\\mathcal{L}\\) flows back through CLIP, the splatting and the projection to every control point.' },
  { key: 'update', label: 'update', dur: 1.1,
    text: 'Adam moves the control points (red: their movement until the next saved snapshot), then the next step begins.' },
];

// wires (from, to) and when their pulse runs, as a fraction of a stage
const PULSES = {
  render: [['scene>sketch', 0.1, 0.8]],
  clip: [['sketch>clip', 0, 0.35], ['photo>clip', 0, 0.35], ['clip>geo', 0.35, 0.68], ['clip>sem', 0.35, 0.68],
    ['geo>loss', 0.68, 1], ['sem>loss', 0.68, 1]],
  back: [['back', 0, 1]],
};
const ACTIVE = {
  views: ['scene'], render: ['scene', 'sketch', 'photo'], clip: ['sketch', 'photo', 'clip', 'geo', 'sem', 'loss'],
  back: ['loss', 'scene'], update: ['scene'],
};

const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - (-2 * x + 2) ** 2 / 2);
const clamp01 = (x) => Math.min(Math.max(x, 0), 1);

function h(tag, cls, parent, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) rich(e, html);
  if (parent) parent.append(e);
  return e;
}

// Camera directions on the upper hemisphere (three rings), like the training views.
function cameraDirs() {
  const dirs = [];
  [[10, 9], [35, 9], [62, 6]].forEach(([el, n], ring) => {
    for (let i = 0; i < n; i++) {
      const az = (2 * Math.PI * (i + 0.5 * ring)) / n, e = THREE.MathUtils.degToRad(el);
      dirs.push(new THREE.Vector3(Math.cos(e) * Math.cos(az), Math.cos(e) * Math.sin(az), Math.sin(e)));
    }
  });
  return dirs.slice(0, N_CAMS);
}

export function frustum(dir, ring = RING) {
  const apex = dir.clone().multiplyScalar(ring), fwd = dir.clone().negate();
  const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 0, 1)).normalize();
  const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
  const base = apex.clone().addScaledVector(fwd, 0.32), s = 0.11;
  const c = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) => base.clone().addScaledVector(right, a * s).addScaledVector(up, b * s));
  const p = [];
  c.forEach((q, i) => { p.push(apex, q, q, c[(i + 1) % 4]); });
  const geo = new THREE.BufferGeometry().setFromPoints(p);
  const mat = new THREE.LineBasicMaterial({ color: GREY, transparent: true, opacity: 0.55 });
  return new THREE.LineSegments(geo, mat);
}

export function pipelineFigure(root, { object = 'bicycle', key = 'normal|20' } = {}) {
  root.classList.add('pf');

  // ------------------------------------------------------------------ layout
  const flow = h('div', 'pf-flow', root);
  const node = (name, cls, parent) => { const n = h('div', `pf-node ${cls || ''}`, parent); n.dataset.node = name; return n; };
  const scene = node('scene', 'pf-scene', flow);
  const sceneCell = h('div', 'pf-cell pf-cell-3d', scene);
  h('div', 'pf-cap', scene, '3D Bézier curves, control points \\(\\mathbf{P}\\)');
  const imgs = h('div', 'pf-col', flow);
  const sk = node('sketch', '', imgs);
  const skCell = h('div', 'pf-cell', sk);
  h('div', 'pf-cap', sk, 'sketch \\(\\mathbf{S}_v\\)');
  const ph = node('photo', '', imgs);
  const phCell = h('div', 'pf-cell', ph);
  h('div', 'pf-cap', ph, 'photo \\(\\mathbf{I}_v\\)');
  const clip = node('clip', 'pf-clip', flow);
  h('div', 'pf-trap', clip, 'CLIP');
  h('div', 'pf-cap', clip, 'ViT-B/32, frozen');
  const losses = h('div', 'pf-col pf-losses', flow);
  const geo = node('geo', 'pf-box pf-geo', losses);
  h('div', 'pf-box-title', geo, 'geometric');
  h('div', 'pf-box-eq', geo, '\\(\\sum_{l} \\|f_l(\\mathbf{S}) - f_l(\\mathbf{I})\\|^2\\)');
  const sem = node('sem', 'pf-box pf-sem', losses);
  h('div', 'pf-box-title', sem, 'semantic');
  h('div', 'pf-box-eq', sem, '\\(1 - \\cos\\bigl(e(\\mathbf{S}), e(\\mathbf{I})\\bigr)\\)');
  const loss = node('loss', 'pf-box pf-loss', flow);
  h('div', 'pf-box-eq', loss, '\\(\\mathcal{L} = w_g \\mathcal{L}_{\\mathrm{geo}} + w_s \\mathcal{L}_{\\mathrm{sem}}\\)');
  const nodes = { scene, sketch: sk, photo: ph, clip, geo, sem, loss };

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.classList.add('pf-wires');
  flow.append(svg);
  const canvas = h('canvas', 'pf-canvas', flow);

  const steps = h('div', 'pf-steps', root);
  const chips = STAGES.map((s, i) => {
    const c = h('button', 'pf-chip', steps);
    h('span', 'pf-chip-n', c).textContent = i + 1;
    h('span', '', c).textContent = s.label;
    c.addEventListener('click', () => { st.stage = i; st.time = 0; st.fast = false; syncMode(); });
    return c;
  });
  const text = h('p', 'pf-text', root);
  const bar = h('div', 'pf-bar', root);
  const play = h('button', 'button is-small is-rounded', bar);
  const modes = h('div', 'buttons has-addons pf-modes', bar);
  const mStep = h('button', 'button is-small', modes); mStep.textContent = 'step by step';
  const mFast = h('button', 'button is-small', modes); mFast.textContent = 'fast-forward';
  const scrub = h('input', 'pf-scrub', bar);
  Object.assign(scrub, { type: 'range', min: 0, max: 1, step: 1, value: 0 });
  const counter = h('span', 'pf-counter', bar);

  // ------------------------------------------------------------------ state
  const st = { stage: 0, time: 0, snap: 0, playing: true, fast: false, picks: [], active: 0, ready: false };
  play.addEventListener('click', () => { st.playing = !st.playing; syncMode(); });
  mStep.addEventListener('click', () => { st.fast = false; st.stage = 0; st.time = 0; syncMode(); });
  mFast.addEventListener('click', () => { st.fast = true; st.playing = true; syncMode(); });
  scrub.addEventListener('input', () => {
    st.snap = Number(scrub.value); st.stage = 4; st.time = 0; st.playing = false; syncMode(); setSnapshot(st.snap, 0);
  });
  function syncMode() {
    play.textContent = st.playing ? 'Pause' : 'Play';
    mStep.classList.toggle('is-dark', !st.fast); mFast.classList.toggle('is-dark', st.fast);
    root.classList.toggle('pf-is-fast', st.fast);
  }
  syncMode();

  // ------------------------------------------------------------------ three.js
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setScissorTest(true);
  const mainScene = new THREE.Scene(), skScene = new THREE.Scene(), phScene = new THREE.Scene();
  const mainCam = new THREE.PerspectiveCamera(34, 1, 0.05, 50), viewCam = new THREE.PerspectiveCamera(FOV, 1, 0.05, 50);
  let az = 0.6, el = 0.42, drag = null;
  sceneCell.addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY]; sceneCell.setPointerCapture(e.pointerId); });
  sceneCell.addEventListener('pointermove', (e) => {
    if (!drag) return;
    az -= (e.clientX - drag[0]) * 0.008; el = THREE.MathUtils.clamp(el + (e.clientY - drag[1]) * 0.008, -0.2, 1.3);
    drag = [e.clientX, e.clientY];
  });
  ['pointerup', 'pointercancel'].forEach((ev) => sceneCell.addEventListener(ev, () => { drag = null; }));

  // floor disc and camera ring for depth
  const fade = document.createElement('canvas');
  fade.width = fade.height = 128;
  const fg = fade.getContext('2d'), grad = fg.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(160,170,190,0.35)'); grad.addColorStop(0.6, 'rgba(160,170,190,0.15)'); grad.addColorStop(1, 'rgba(160,170,190,0)');
  fg.fillStyle = grad; fg.fillRect(0, 0, 128, 128);
  const disc = new THREE.Mesh(new THREE.PlaneGeometry(RING * 2.2, RING * 2.2),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(fade), transparent: true, depthWrite: false }));
  disc.position.z = -0.7;
  mainScene.add(disc);
  const dirs = cameraDirs();
  const frusta = dirs.map((d) => { const f = frustum(d); mainScene.add(f); return f; });

  const mat = (color, width) => new LineMaterial({ color, linewidth: width, transparent: true });
  const mainMat = mat(INK, 1.8), skMat = mat(INK, 1.6), gradMat = mat(RED, 1.5);
  const curveGeo = new LineSegmentsGeometry();
  const mainLines = new LineSegments2(curveGeo, mainMat), skLines = new LineSegments2(curveGeo, skMat);
  mainScene.add(mainLines); skScene.add(skLines);
  const gradGeo = new LineSegmentsGeometry();
  const gradLines = new LineSegments2(gradGeo, gradMat);
  mainScene.add(gradLines);
  const tipGeo = new THREE.SphereGeometry(0.016, 10, 8);
  let tips = null;
  const rays = new THREE.LineSegments(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: ORANGE, transparent: true, opacity: 0 }));
  mainScene.add(rays);
  let gt = null;
  loadGT(object).then((o) => { if (o) { gt = o; phScene.add(o); } });

  // ------------------------------------------------------------------ data
  let runs = null, hist = null, batches = [], cpNow = null, cpA = null, cpB = null;
  let seq = [];                  // snapshot indices shown, evenly strided, always ending with the final one
  (async () => {
    runs = await loadRuns(object, 'ablation');
    if (!runs || !runs.meta.runs[key]) { text.textContent = 'Training data not available yet.'; return; }
    hist = await runs.history();
    batches = runs.meta.runs[key].snapshots;
    const stride = Math.max(1, Math.ceil((batches.length - 1) / N_STEPS));
    for (let i = 0; i < batches.length - 1; i += stride) seq.push(i);
    seq.push(batches.length - 1);
    scrub.max = seq.length - 1;
    tips = new THREE.InstancedMesh(tipGeo, new THREE.MeshBasicMaterial({ color: RED }), runs.meta.runs[key].curves * 4);
    mainScene.add(tips);
    setSnapshot(0, 0);
    pickViews();
    st.ready = true;
  })();

  const snapCP = (i) => controlPoints(runs, key, seq[Math.min(i, seq.length - 1)], hist);

  // curves at snapshot i moving towards i+1 by f (0..1); red arrows show the rest of the way
  function setSnapshot(i, f, arrows = 0) {
    cpA = snapCP(i); cpB = snapCP(i + 1);
    cpNow = new Float32Array(cpA.length);
    for (let k = 0; k < cpA.length; k++) cpNow[k] = cpA[k] + f * (cpB[k] - cpA[k]);
    curveGeo.setPositions(curvesToSegments(cpNow));
    // arrow scale: the largest movement is drawn 0.22 long
    let maxd = 1e-6;
    for (let k = 0; k < cpA.length; k += 3) maxd = Math.max(maxd, Math.hypot(cpB[k] - cpA[k], cpB[k + 1] - cpA[k + 1], cpB[k + 2] - cpA[k + 2]));
    const sc = Math.min(0.22 / maxd, 6) * arrows, seg = [], m = new THREE.Matrix4();
    for (let k = 0, j = 0; k < cpA.length; k += 3, j++) {
      const a = [cpNow[k], cpNow[k + 1], cpNow[k + 2]];
      const b = [a[0] + sc * (cpB[k] - cpNow[k]), a[1] + sc * (cpB[k + 1] - cpNow[k + 1]), a[2] + sc * (cpB[k + 2] - cpNow[k + 2])];
      seg.push(...a, ...b);
      m.makeScale(arrows, arrows, arrows).setPosition(b[0], b[1], b[2]);
      tips.setMatrixAt(j, m);
    }
    gradGeo.setPositions(seg);
    gradLines.visible = tips.visible = arrows > 0.01;
    tips.instanceMatrix.needsUpdate = true;
    scrub.value = i;
    counter.textContent = `training step ${batches[seq[Math.min(i + (f > 0.5 ? 1 : 0), seq.length - 1)]]} / ${batches[batches.length - 1]}`;
  }

  function pickViews() {
    const idx = [...dirs.keys()].sort(() => Math.random() - 0.5);
    st.picks = idx.slice(0, N_PICK);
    st.active = st.picks[0];
  }

  // ------------------------------------------------------------------ wires
  const NS = 'http://www.w3.org/2000/svg';
  const defs = document.createElementNS(NS, 'defs');
  defs.innerHTML = `<marker id="pf-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8a909c"/></marker>
    <marker id="pf-ar-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#d6453d"/></marker>`;
  svg.append(defs);
  const wires = {};
  const WIRE_NAMES = ['scene>sketch', 'sketch>clip', 'photo>clip', 'clip>geo', 'clip>sem', 'geo>loss', 'sem>loss', 'back'];
  for (const w of WIRE_NAMES) {
    const p = document.createElementNS(NS, 'path');
    p.setAttribute('class', w === 'back' ? 'pf-wire pf-wire-back' : 'pf-wire');
    p.setAttribute('marker-end', w === 'back' ? 'url(#pf-ar-red)' : 'url(#pf-ar)');
    const dot = document.createElementNS(NS, 'circle');
    dot.setAttribute('r', 4.5);
    dot.setAttribute('class', w === 'back' ? 'pf-pulse pf-pulse-back' : 'pf-pulse');
    svg.append(p, dot);
    wires[w] = { p, dot, len: 0 };
  }
  const backLabel = h('div', 'pf-back-label', flow, 'gradients \\(\\partial\\mathcal{L} / \\partial\\mathbf{P}\\)');

  function layoutWires() {
    const R = flow.getBoundingClientRect();
    svg.setAttribute('viewBox', `0 0 ${R.width} ${R.height}`);
    const box = (n) => { const b = (n === 'scene' ? sceneCell : n === 'sketch' ? skCell : n === 'photo' ? phCell : nodes[n]).getBoundingClientRect(); return { l: b.left - R.left, r: b.right - R.left, t: b.top - R.top, b: b.bottom - R.top, cx: (b.left + b.right) / 2 - R.left, cy: (b.top + b.bottom) / 2 - R.top }; };
    const curve = (a, b) => { const dx = Math.max((b[0] - a[0]) * 0.45, 12); return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`; };
    const link = (from, to, fy = 'cy', ty = 'cy') => { const a = box(from), b = box(to); return curve([a.r + 4, a[fy]], [b.l - 5, b[ty]]); };
    const set = { 'scene>sketch': link('scene', 'sketch'), 'sketch>clip': link('sketch', 'clip'), 'photo>clip': link('photo', 'clip'),
      'clip>geo': link('clip', 'geo'), 'clip>sem': link('clip', 'sem'), 'geo>loss': link('geo', 'loss'), 'sem>loss': link('sem', 'loss') };
    const L = box('loss'), S = box('scene'), top = Math.max(6, Math.min(L.t, S.t) - 52);
    set.back = `M${L.cx},${L.t - 4} C${L.cx},${top} ${S.cx},${top} ${S.cx},${S.t - 6}`;
    for (const [w, d] of Object.entries(set)) { wires[w].p.setAttribute('d', d); wires[w].len = wires[w].p.getTotalLength(); }
    backLabel.style.left = `${(L.cx + S.cx) / 2}px`; backLabel.style.top = `${top + 8}px`;
  }
  new ResizeObserver(layoutWires).observe(flow);

  // ------------------------------------------------------------------ animation
  let visible = false, last = performance.now(), clock = 0;
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { rootMargin: '100px' }).observe(root);

  function stageEffects(stage, f) {
    const s = STAGES[stage].key;
    // highlights
    for (const [n, e] of Object.entries(nodes)) e.classList.toggle('pf-on', !st.fast && ACTIVE[s].includes(n));
    chips.forEach((c, i) => c.classList.toggle('is-active', !st.fast && i === stage));
    // pulses
    for (const w of Object.values(wires)) w.dot.style.opacity = 0;
    if (!st.fast) {
      for (const [w, a, b] of PULSES[s] || []) {
        const g = clamp01((f - a) / (b - a));
        if (g <= 0 || g >= 1) continue;
        const pt = wires[w].p.getPointAtLength(ease(g) * wires[w].len);
        wires[w].dot.setAttribute('cx', pt.x); wires[w].dot.setAttribute('cy', pt.y);
        wires[w].dot.style.opacity = Math.sin(Math.PI * g) ** 0.4;
      }
    }
    svg.querySelector('.pf-wire-back').classList.toggle('pf-hot', !st.fast && s === 'back');
    // cameras: picked ones light up one after another during "views"
    const shown = s === 'views' ? Math.ceil(f * N_PICK * 1.3) : N_PICK;
    frusta.forEach((fr, i) => {
      const k = st.picks.indexOf(i), on = k >= 0 && k < shown;
      fr.material.color.setHex(on ? ORANGE : GREY);
      fr.material.opacity = on ? (i === st.active ? 1 : 0.85) : 0.4;
    });
    // projection rays from the active camera during "render"
    if (s === 'render' && cpNow && !st.fast) {
      const apex = dirs[st.active].clone().multiplyScalar(RING), pts = [];
      for (let k = 0; k < cpNow.length; k += 12) pts.push(apex, new THREE.Vector3(
        0.125 * cpNow[k] + 0.375 * cpNow[k + 3] + 0.375 * cpNow[k + 6] + 0.125 * cpNow[k + 9],
        0.125 * cpNow[k + 1] + 0.375 * cpNow[k + 4] + 0.375 * cpNow[k + 7] + 0.125 * cpNow[k + 10],
        0.125 * cpNow[k + 2] + 0.375 * cpNow[k + 5] + 0.375 * cpNow[k + 8] + 0.125 * cpNow[k + 11]));
      rays.geometry.setFromPoints(pts);
      rays.material.opacity = 0.55 * Math.sin(Math.PI * clamp01(f / 0.85));
    } else rays.material.opacity = 0;
    // curves and update arrows
    if (s === 'back') setSnapshot(st.snap, 0, ease(clamp01(f / 0.8)));
    else if (s === 'update') setSnapshot(st.snap, ease(f), 1 - ease(f));
    text.innerHTML = '';
    rich(text, st.fast ? `Fast-forward through the whole run in ${seq.length - 1} jumps, with 8 new views each time.` : STAGES[stage].text);
  }

  function advance(dt) {
    if (!st.ready || !st.playing) return;
    if (st.fast) {
      st.time += dt / 0.28;
      if (st.time >= 1) { st.time = 0; st.snap += 1; pickViews(); }
      if (st.snap >= seq.length - 1) { st.snap = 0; }
      setSnapshot(st.snap, ease(st.time), 0);
      return;
    }
    st.time += dt / STAGES[st.stage].dur;
    if (st.time >= 1) {
      st.time = 0;
      st.stage = (st.stage + 1) % STAGES.length;
      if (st.stage === 0) {
        st.snap = st.snap + 1 >= seq.length - 1 ? 0 : st.snap + 1;
        pickViews();
        setSnapshot(st.snap, 0);
      }
    }
  }

  function viewport(cell, R, clear) {
    const c = cell.getBoundingClientRect(), x = c.left - R.left, y = R.bottom - c.bottom;
    renderer.setViewport(x, y, c.width, c.height); renderer.setScissor(x, y, c.width, c.height);
    if (clear) { renderer.setClearColor(0xffffff, 0); renderer.clear(); }
    return c;
  }

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1); last = now;
    requestAnimationFrame(frame);
    if (!visible) return;
    clock += dt;
    advance(dt);
    if (st.ready) stageEffects(st.stage, st.time);
    if (!drag) az += dt * 0.12;

    const R = flow.getBoundingClientRect(), pr = renderer.getPixelRatio();
    if (canvas.width !== Math.round(R.width * pr) || canvas.height !== Math.round(R.height * pr)) renderer.setSize(R.width, R.height, true);
    renderer.setScissor(0, 0, R.width, R.height); renderer.setClearColor(0xffffff, 0); renderer.clear();

    // main 3D scene
    let c = viewport(sceneCell, R, false);
    const d = 5.2, cz = 0.25;
    mainCam.position.set(d * Math.cos(el) * Math.cos(az), d * Math.cos(el) * Math.sin(az), d * Math.sin(el) + cz);
    mainCam.up.set(0, 0, 1); mainCam.lookAt(0, 0, cz);
    mainCam.aspect = c.width / c.height; mainCam.updateProjectionMatrix();
    mainMat.resolution.set(c.width, c.height); gradMat.resolution.set(c.width, c.height);
    renderer.render(mainScene, mainCam);

    // the active training view: sketch and photo from the same camera
    viewCam.position.copy(dirs[st.active]).multiplyScalar(VIEW_DIST);
    viewCam.up.set(0, 0, 1); viewCam.lookAt(0, 0, 0); viewCam.aspect = 1; viewCam.updateProjectionMatrix();
    c = viewport(skCell, R, true);
    skMat.resolution.set(c.width, c.height);
    renderer.render(skScene, viewCam);
    viewport(phCell, R, true);
    if (gt) renderer.render(phScene, viewCam);
  }
  requestAnimationFrame(frame);
}
