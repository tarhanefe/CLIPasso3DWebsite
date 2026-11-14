// Animated initialization and respawning on a real run:
//   1 masks   - object silhouettes from training cameras (rendered here from the ground-truth mesh)
//   2 carve   - a voxel grid carved camera by camera into the visual hull (computed in the browser)
//   3 seed    - the run's actual initial short lines on the hull
//   4 train   - the run's training history; curves that jump far between snapshots were respawned
import * as THREE from 'three';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { loadRuns, loadGT, controlPoints, curvesToSegments } from './sketchgrid.js';
import { frustum } from './pipelinefig.js';
import { rich } from './bezierfig.js';

const ORANGE = 0xe8743b, RED = 0xd6453d, GREY = 0xb9bfca, BLUE = 0x5b8fec, PALE = 0xb7cdf6, INK = 0x111111;
const RES = 26;                 // voxels per side of the [-1, 1]^3 grid
const MASK = 96;                // mask resolution
const CAM_DIST = 3.25 / 1.12, FOV = 39.6, RING = 1.75;
const JUMP = 0.4;               // mean control point jump between snapshots that marks a respawn

const STAGES = [
  { key: 'masks', label: 'object masks', dur: 3.2,
    text: 'Every training view comes with the object mask: where the object is in that image.' },
  { key: 'carve', label: 'carve the visual hull', dur: 6.0,
    text: 'A voxel grid is carved view by view: a voxel survives only if it falls inside every silhouette. What remains is the visual hull.' },
  { key: 'seed', label: 'seed lines', dur: 3.2,
    text: 'Short lines are spread over the hull surface by farthest point sampling (the actual starting lines of this run).' },
  { key: 'train', label: 'train + respawn', dur: 0,
    text: 'During training, curves that leave the object or collapse are moved to the least covered part of the hull (red: removed, orange: respawned).' },
];

const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - (-2 * x + 2) ** 2 / 2);
const clamp01 = (x) => Math.min(Math.max(x, 0), 1);

function h(tag, cls, parent, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) rich(e, html);
  if (parent) parent.append(e);
  return e;
}

// 12 cameras: four elevation bands, including two from below
function cameraDirs() {
  const out = [];
  [[-25, 2, 0.4], [10, 4, 0], [40, 4, 0.8], [75, 2, 0.3]].forEach(([el, n, off]) => {
    for (let i = 0; i < n; i++) {
      const az = (2 * Math.PI * (i + off)) / n, e = THREE.MathUtils.degToRad(el);
      out.push(new THREE.Vector3(Math.cos(e) * Math.cos(az), Math.cos(e) * Math.sin(az), Math.sin(e)));
    }
  });
  // interleave so consecutive cameras look from different sides
  return [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11].map((i) => out[i]);
}

export function initFigure(root, { object = 'mug', key = 'middle_abstract|20' } = {}) {
  root.classList.add('if');

  // ------------------------------------------------------------------ layout
  const strip = h('div', 'if-strip', root);
  const thumbs = [];
  const stage3d = h('div', 'if-stage', root);
  const canvas = h('canvas', 'if-canvas', stage3d);
  const steps = h('div', 'pf-steps', root);
  const chips = STAGES.map((s, i) => {
    const c = h('button', 'pf-chip', steps);
    h('span', 'pf-chip-n', c).textContent = i + 1;
    h('span', '', c).textContent = s.label;
    c.addEventListener('click', () => goto(i));
    return c;
  });
  const text = h('p', 'pf-text', root);
  const bar = h('div', 'pf-bar', root);
  const play = h('button', 'button is-small is-rounded', bar);
  const counter = h('span', 'pf-counter', bar);

  const st = { stage: 0, time: 0, playing: true, ready: false, snapT: 0 };
  play.addEventListener('click', () => { st.playing = !st.playing; play.textContent = st.playing ? 'Pause' : 'Play'; });
  play.textContent = 'Pause';

  // ------------------------------------------------------------------ three.js
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa3b5, 2.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(2, -3, 4);
  scene.add(sun);
  const cam = new THREE.PerspectiveCamera(32, 2, 0.05, 50);
  let az = 0.9, el = 0.38, drag = null;
  stage3d.addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY]; stage3d.setPointerCapture(e.pointerId); });
  stage3d.addEventListener('pointermove', (e) => {
    if (!drag) return;
    az -= (e.clientX - drag[0]) * 0.008; el = THREE.MathUtils.clamp(el + (e.clientY - drag[1]) * 0.008, -0.6, 1.3);
    drag = [e.clientX, e.clientY];
  });
  ['pointerup', 'pointercancel'].forEach((ev) => stage3d.addEventListener(ev, () => { drag = null; }));

  const dirs = cameraDirs();
  const frusta = dirs.map((d) => { const f = frustum(d, RING); scene.add(f); return f; });

  // voxels (instanced cubes; only surface voxels are drawn)
  const N = RES ** 3, vs = 2 / RES;
  const center = (i) => [((i % RES) + 0.5) * vs - 1, ((Math.floor(i / RES) % RES) + 0.5) * vs - 1, (Math.floor(i / RES / RES) + 0.5) * vs - 1];
  const vox = new THREE.InstancedMesh(new THREE.BoxGeometry(vs * 0.92, vs * 0.92, vs * 0.92),
    new THREE.MeshLambertMaterial({ color: BLUE, transparent: true, opacity: 1 }), N);
  vox.count = 0;
  scene.add(vox);
  const alive = new Uint8Array(N).fill(1);
  const keepAfter = [];          // keepAfter[k][i]: voxel i survives cameras 0..k

  // lines
  const lineMat = new LineMaterial({ color: INK, linewidth: 2, transparent: true });
  const ghostMat = new LineMaterial({ color: RED, linewidth: 2.4, transparent: true });
  const bornMat = new LineMaterial({ color: ORANGE, linewidth: 2.6, transparent: true });
  const lines = new LineSegments2(new LineSegmentsGeometry(), lineMat);
  const ghosts = new LineSegments2(new LineSegmentsGeometry(), ghostMat);
  const born = new LineSegments2(new LineSegmentsGeometry(), bornMat);
  scene.add(lines, ghosts, born);
  lines.visible = ghosts.visible = born.visible = false;

  // pulse rings where lines are seeded / respawned
  const ringGeo = new THREE.RingGeometry(0.035, 0.05, 32);
  const rings = [];
  const ring = (p, color, t0) => {
    const m = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
    m.position.set(...p); scene.add(m); rings.push({ m, t0 });
  };

  let gt = null, runs = null, hist = null, batches = [], C = 0, events = [];
  const snapCP = (i) => controlPoints(runs, key, i, hist);

  (async () => {
    const [g, r] = await Promise.all([loadGT(object), loadRuns(object, 'ablation')]);
    if (!g || !r || !r.meta.runs[key]) { text.textContent = 'Data not available yet.'; return; }
    gt = g; runs = r;
    hist = await runs.history();
    batches = runs.meta.runs[key].snapshots;
    C = runs.meta.runs[key].curves;
    // respawn events: curves whose control points jump far between consecutive snapshots
    for (let i = 0; i + 1 < batches.length; i++) {
      const a = snapCP(i), b = snapCP(i + 1), ev = [];
      for (let c = 0; c < C; c++) {
        let d = 0;
        for (let k = 0; k < 4; k++) { const o = c * 12 + k * 3; d += Math.hypot(b[o] - a[o], b[o + 1] - a[o + 1], b[o + 2] - a[o + 2]) / 4; }
        if (d > JUMP) ev.push(c);
      }
      events.push(ev);
    }
    computeMasks();
    gt.traverse((o) => { if (o.isMesh) { o.material = o.material.clone(); o.material.transparent = true; } });
    scene.add(gt);
    st.ready = true;
  })();

  // silhouettes from each camera, and the carving they imply
  function computeMasks() {
    const rt = new THREE.WebGLRenderTarget(MASK, MASK);
    const ms = new THREE.Scene(), black = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const g = gt.clone();
    g.traverse((o) => { if (o.isMesh) o.material = black; });
    ms.add(g);
    ms.background = new THREE.Color(0xffffff);
    const c = new THREE.PerspectiveCamera(FOV, 1, 0.05, 50), buf = new Uint8Array(MASK * MASK * 4);
    const keep = new Uint8Array(N).fill(1), v = new THREE.Vector3();
    dirs.forEach((d, k) => {
      c.position.copy(d).multiplyScalar(CAM_DIST); c.up.set(0, 0, 1); c.lookAt(0, 0, 0); c.updateMatrixWorld(); c.updateProjectionMatrix();
      renderer.setRenderTarget(rt); renderer.render(ms, c); renderer.readRenderTargetPixels(rt, 0, 0, MASK, MASK, buf);
      renderer.setRenderTarget(null);
      // thumbnail (WebGL rows are bottom-up)
      const tc = h('canvas', 'if-thumb', strip);
      tc.width = tc.height = MASK;
      const ctx = tc.getContext('2d'), img = ctx.createImageData(MASK, MASK);
      for (let y = 0; y < MASK; y++) for (let x = 0; x < MASK; x++) {
        const s = ((MASK - 1 - y) * MASK + x) * 4, t = (y * MASK + x) * 4, inside = buf[s] < 128;
        img.data[t] = img.data[t + 1] = img.data[t + 2] = inside ? 30 : 250; img.data[t + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      thumbs.push(tc);
      // carve: a voxel survives if its center projects inside the silhouette (dilated by one pixel)
      for (let i = 0; i < N; i++) {
        if (!keep[i]) continue;
        v.set(...center(i)).project(c);
        const px = Math.floor((v.x * 0.5 + 0.5) * MASK), py = Math.floor((v.y * 0.5 + 0.5) * MASK);
        let inside = false;
        for (let dy = -1; dy <= 1 && !inside; dy++) for (let dx = -1; dx <= 1 && !inside; dx++) {
          const x = px + dx, y = py + dy;
          if (x >= 0 && y >= 0 && x < MASK && y < MASK && buf[(y * MASK + x) * 4] < 128) inside = true;
        }
        if (!inside) keep[i] = 0;
      }
      keepAfter.push(keep.slice());
    });
    rt.dispose();
  }

  // draw surface voxels of `set`; voxels in `fading` shrink by f
  const m4 = new THREE.Matrix4(), col = new THREE.Color();
  function showVoxels(set, fading = null, f = 0, color = BLUE) {
    const any = (i) => set[i] || (fading && fading[i]);
    let n = 0;
    for (let i = 0; i < N; i++) {
      if (!any(i)) continue;
      const x = i % RES, y = Math.floor(i / RES) % RES, z = Math.floor(i / RES / RES);
      const surface = x === 0 || y === 0 || z === 0 || x === RES - 1 || y === RES - 1 || z === RES - 1 ||
        !any(i - 1) || !any(i + 1) || !any(i - RES) || !any(i + RES) || !any(i - RES * RES) || !any(i + RES * RES);
      if (!surface) continue;
      const s = set[i] ? 1 : 1 - f;
      if (s <= 0.02) continue;
      m4.makeScale(s, s, s).setPosition(...center(i));
      vox.setMatrixAt(n, m4);
      vox.setColorAt(n, col.setHex(set[i] ? color : ORANGE));
      n++;
    }
    vox.count = n;
    vox.instanceMatrix.needsUpdate = true;
    if (vox.instanceColor) vox.instanceColor.needsUpdate = true;
  }

  // (a new geometry each time: three.js keeps the old buffers when the segment count changes)
  function setLines(obj, cp, curves = null) {
    if (!cp) { obj.visible = false; return; }
    let src = cp;
    if (curves) { src = new Float32Array(curves.length * 12); curves.forEach((c, j) => src.set(cp.subarray(c * 12, c * 12 + 12), j * 12)); }
    obj.visible = src.length > 0;
    if (!src.length) return;
    obj.geometry.dispose();
    obj.geometry = new LineSegmentsGeometry();
    obj.geometry.setPositions(curvesToSegments(src));
  }

  const midpoint = (cp, c) => {
    const o = c * 12;
    return [0, 1, 2].map((k) => 0.125 * cp[o + k] + 0.375 * cp[o + 3 + k] + 0.375 * cp[o + 6 + k] + 0.125 * cp[o + 9 + k]);
  };

  function goto(i) {
    st.stage = i; st.time = 0; st.snapT = 0; st.seeded = 0; st.lastSnap = -1;
    rings.splice(0).forEach((r) => scene.remove(r.m));
  }

  // ------------------------------------------------------------------ per-frame stage logic
  function update(dt, clock) {
    const s = STAGES[st.stage].key, f = st.time;
    chips.forEach((c, i) => c.classList.toggle('is-active', i === st.stage));
    text.innerHTML = ''; rich(text, STAGES[st.stage].text);
    const active = s === 'masks' ? Math.min(Math.floor(f * 12 * 1.15), 11) : s === 'carve' ? Math.min(Math.floor(f * 12), 11) : -1;
    thumbs.forEach((t, k) => {
      t.classList.toggle('is-on', k === active);
      t.classList.toggle('is-seen', s === 'masks' ? k < active : s === 'carve' ? k < active : true);
    });
    frusta.forEach((fr, k) => {
      const on = k === active, seen = s === 'masks' ? k <= active : s === 'carve' ? k <= active : false;
      fr.visible = s === 'masks' || s === 'carve';
      fr.material.color.setHex(on ? ORANGE : seen ? 0x8d96a8 : GREY);
      fr.material.opacity = on ? 1 : 0.5;
    });
    // ground truth ghost
    gt.visible = s === 'masks' || (s === 'carve' && f < 0.15);
    const gto = s === 'masks' ? 0.85 : 0.85 * (1 - f / 0.15);
    gt.traverse((o) => { if (o.isMesh) o.material.opacity = gto; });

    if (s === 'masks') {
      vox.visible = false; lines.visible = ghosts.visible = born.visible = false;
      counter.textContent = `${active + 1} / 12 masks`;
    } else if (s === 'carve') {
      vox.visible = true; vox.material.opacity = 1; vox.material.depthWrite = true;
      lines.visible = ghosts.visible = born.visible = false;
      const g = f * 12 - active;                         // progress within the current camera
      showVoxels(keepAfter[active], active ? keepAfter[active - 1] : alive, ease(clamp01(g / 0.8)));
      let n = 0;
      for (let i = 0; i < N; i++) n += keepAfter[active][i];
      counter.textContent = `camera ${active + 1} / 12 · ${n} voxels left`;
    } else if (s === 'seed') {
      const g0 = clamp01(f / 0.3);
      vox.visible = true; vox.material.opacity = 1 - 0.8 * g0; vox.material.depthWrite = g0 < 0.5;
      showVoxels(keepAfter[11], null, 0, col.setHex(BLUE).lerp(new THREE.Color(PALE), g0).getHex());
      const cp0 = snapCP(0), n = Math.min(C, Math.floor(clamp01((f - 0.1) / 0.75) * C + 1e-6));
      for (let c = st.seeded || 0; c < n; c++) ring(midpoint(cp0, c), ORANGE, clock);
      st.seeded = n;
      setLines(lines, cp0, [...Array(n).keys()]);
      ghosts.visible = born.visible = false;
      counter.textContent = `${n} / ${C} lines seeded`;
    } else {
      vox.visible = true; vox.material.opacity = 0.2 - 0.1 * clamp01((st.snapT - 2) / 6); vox.material.depthWrite = false;
      showVoxels(keepAfter[11], null, 0, PALE);
      const i = Math.min(Math.floor(st.snapT), batches.length - 1), g = st.snapT - i;
      const a = snapCP(i), b = snapCP(Math.min(i + 1, batches.length - 1)), ev = events[i] || [];
      const cp = new Float32Array(a.length);
      for (let k = 0; k < a.length; k++) cp[k] = a[k] + ease(g) * (b[k] - a[k]);
      // respawned curves do not travel: the old one turns red and fades, the new one appears
      for (const c of ev) cp.set((g < 0.5 ? a : b).subarray(c * 12, c * 12 + 12), c * 12);
      setLines(lines, cp, [...Array(C).keys()].filter((c) => !ev.includes(c)));
      setLines(ghosts, a, ev);
      ghostMat.opacity = 1 - clamp01(g / 0.55);
      setLines(born, b, ev);
      bornMat.opacity = clamp01((g - 0.4) / 0.3);
      if (ev.length && st.lastSnap !== i) {
        ev.forEach((c) => ring(midpoint(b, c), ORANGE, clock + 0.45 * EVENT_DUR));
        st.lastSnap = i;
      }
      counter.textContent = `training step ${batches[g > 0.5 ? Math.min(i + 1, batches.length - 1) : i]} / ${batches[batches.length - 1]}` +
        (ev.length ? ` · respawning ${ev.length} curve${ev.length > 1 ? 's' : ''}` : '');
    }
    // pulse rings face the camera, grow and fade
    for (let k = rings.length - 1; k >= 0; k--) {
      const r = rings[k], a = clock - r.t0;
      if (a > 1.1) { scene.remove(r.m); rings.splice(k, 1); continue; }
      r.m.visible = a >= 0;
      r.m.quaternion.copy(cam.quaternion);
      r.m.scale.setScalar(1 + 1.3 * clamp01(a));
      r.m.material.opacity = 0.9 * (1 - clamp01(a / 1.1));
    }
  }

  const SNAP_DUR = 0.32, EVENT_DUR = 1.6;
  function advance(dt) {
    if (!st.playing) return;
    const s = STAGES[st.stage];
    if (s.key === 'train') {
      const i = Math.min(Math.floor(st.snapT), batches.length - 1);
      st.snapT += dt / ((events[i] || []).length ? EVENT_DUR : SNAP_DUR);
      if (st.snapT >= batches.length + 2) goto(0);     // hold the final sketch for a moment, then loop
      return;
    }
    st.time += dt / s.dur;
    if (st.time >= 1) goto(st.stage + 1);
  }

  // ------------------------------------------------------------------ loop
  let visible = false, last = performance.now(), clock = 0;
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { rootMargin: '100px' }).observe(root);
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1); last = now;
    requestAnimationFrame(frame);
    if (!visible || !st.ready) return;
    clock += dt;
    if (!drag) az += dt * 0.15;
    advance(dt);
    update(dt, clock);
    const w = stage3d.clientWidth, hgt = stage3d.clientHeight, pr = renderer.getPixelRatio();
    if (canvas.width !== Math.round(w * pr) || canvas.height !== Math.round(hgt * pr)) renderer.setSize(w, hgt, false);
    const d = 4.7;
    cam.position.set(d * Math.cos(el) * Math.cos(az), d * Math.cos(el) * Math.sin(az), d * Math.sin(el));
    cam.up.set(0, 0, 1); cam.lookAt(0, 0, 0); cam.aspect = w / hgt; cam.updateProjectionMatrix();
    [lineMat, ghostMat, bornMat].forEach((m) => m.resolution.set(w, hgt));
    renderer.render(scene, cam);
  }
  requestAnimationFrame(frame);
}
