/* 3D 高斯查看器：基于 antimatter15/Spark（WebGL2 高斯泼溅渲染）
   数据为 .splat 二进制（32B/点），经 Spark SplatMesh 加载渲染 */
import * as THREE from "../vendor/three/three.module.js";
import { SparkRenderer, SplatMesh, SplatFileType } from "../vendor/Spark/spark.module.js";

let renderer = null, scene = null, camera = null, spark = null, grid = null;
let splat = null;
let container = null, rafId = 0;

// 轨道相机参数
const cam = { yaw: 0.7, pitch: 0.35, dist: 2.4, tx: 0, ty: 0, tz: 0 };

function loop() {
  rafId = requestAnimationFrame(loop);
  if (!renderer) return;
  try {
    const cy = Math.cos(cam.pitch), sy = Math.sin(cam.pitch);
    camera.position.set(
      cam.tx + cam.dist * cy * Math.sin(cam.yaw),
      cam.ty + cam.dist * sy,
      cam.tz + cam.dist * cy * Math.cos(cam.yaw));
    camera.lookAt(cam.tx, cam.ty, cam.tz);
    if (spark) spark.update({ scene, camera }).catch((e) => console.error("Spark update error", e));
    renderer.render(scene, camera);
  } catch (e) {
    console.error("渲染循环错误", e);
  }
}

function onResize() {
  if (!container || !renderer) return;
  const w = container.clientWidth, h = container.clientHeight;
  if (w < 2 || h < 2) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function attachInput() {
  const el = renderer.domElement;
  let lastX = 0, lastY = 0, dragging = false;
  el.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX; lastY = e.clientY;
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (e.buttons === 1 || e.buttons === 2) {
      if (e.buttons === 1) {
        cam.yaw -= dx * 0.01;
        cam.pitch += dy * 0.01;
        cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
      } else {
        const s = cam.dist * 0.0012;
        cam.tx -= dx * s;
        cam.ty += dy * s;
      }
    }
  });
  el.addEventListener("pointerup", () => { dragging = false; });
  el.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.dist *= Math.pow(1.0015, e.deltaY);
    cam.dist = Math.max(0.4, Math.min(30, cam.dist));
  }, { passive: false });
}

function init(el) {
  container = el;
  console.log("[viewer] init start, three r" + THREE.REVISION);
  // WebGL2 检测（Spark 必需）
  const probe = document.createElement("canvas");
  const gl2 = probe.getContext("webgl2");
  console.log("[viewer] WebGL2 supported: " + !!gl2);
  if (!gl2) {
    console.error("WebGL2 不可用，Spark 无法渲染。请检查显卡驱动/硬件加速。");
  }

  const w = el.clientWidth || 800, h = el.clientHeight || 600;
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2.5));
  el.appendChild(renderer.domElement);
  console.log("[viewer] WebGLRenderer created");

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);

  // 质量参数：更高像素比超采样 + 更宽的高斯衰减（减少硬边/颗粒感）
  spark = new SparkRenderer({
    renderer,
    maxStdDev: 4.0,      // 高斯衰减半径（默认 √8≈2.83，调大更平滑）
    minPixelRadius: 0,
    maxPixelRadius: 1024,
  });
  scene.add(spark);
  console.log("[viewer] SparkRenderer created");

  grid = new THREE.GridHelper(2.4, 24, 0x26374B, 0x151D2B);
  grid.position.y = -0.5;
  scene.add(grid);

  if (window.ResizeObserver) {
    new ResizeObserver(onResize).observe(el);
  } else {
    window.addEventListener("resize", onResize);
  }
  attachInput();
  rafId = requestAnimationFrame(loop);
  console.log("[viewer] init done, loop started");
}

// 加入场景并自动取景
function addMesh(mesh) {
  if (splat) {
    scene.remove(splat);
    splat.dispose();
  }
  splat = mesh;
  scene.add(splat);
  try {
    const box = mesh.getBoundingBox();
    if (box && !box.isEmpty()) {
      const center = new THREE.Vector3();
      box.getCenter(center);
      const size = new THREE.Vector3();
      box.getSize(size);
      const radius = Math.max(size.x, size.y, size.z) * 0.5 + 1e-3;
      cam.tx = center.x; cam.ty = center.y; cam.tz = center.z;
      cam.dist = Math.max(0.6, radius * 3.2);
      grid.position.y = center.y - radius;
    }
  } catch (e) { /* 取景失败忽略 */ }
}

// fileBytes: .splat 二进制（32B/点，坐标已正立）
async function setModel(fileBytes, count) {
  if (!renderer || !fileBytes || count <= 0) return;
  let mesh;
  try {
    mesh = new SplatMesh({ fileBytes, fileType: SplatFileType.SPLAT });
    await mesh.initialized;
  } catch (e) {
    console.error("Spark load failed", e);
    return;
  }
  addMesh(mesh);
}

// 导入用户 PLY，原样渲染、不做坐标变换
async function loadPly(fileBytes) {
  if (!renderer || !fileBytes) return;
  let mesh;
  try {
    mesh = new SplatMesh({ fileBytes, fileType: SplatFileType.PLY });
    await mesh.initialized;
  } catch (e) {
    console.error("Spark PLY load failed", e);
    return;
  }
  addMesh(mesh);
}

// 视图切换（front / left / top / persp）
function setView(name) {
  if (name === "front") { cam.yaw = 0; cam.pitch = 0; }
  else if (name === "left") { cam.yaw = -Math.PI / 2; cam.pitch = 0; }
  else if (name === "top") { cam.yaw = 0; cam.pitch = Math.PI / 2 - 0.01; }
  else if (name === "persp") { cam.yaw = 0.7; cam.pitch = 0.35; }
  cam.dist = Math.max(0.6, cam.dist);
}

function toggleGrid() {
  if (grid) grid.visible = !grid.visible;
}

// 归位：重置默认视角
function resetView() {
  cam.yaw = 0.7; cam.pitch = 0.35; cam.dist = 2.4;
}

// 上下翻转（部分外部 PLY 是 -Y up，导入后倒置，手动纠正）
let flippedY = false;
function flipY() {
  if (!splat) return;
  flippedY = !flippedY;
  splat.scale.y = flippedY ? -1 : 1;
  // 网格跟随模型底部
  if (grid) {
    try {
      const box = splat.getBoundingBox();
      if (box && !box.isEmpty()) {
        const center = new THREE.Vector3();
        box.getCenter(center);
        const size = new THREE.Vector3();
        box.getSize(size);
        grid.position.y = center.y - size.y * 0.5;
      }
    } catch (e) { /* 忽略 */ }
  }
}

function clearModel() {
  if (splat) {
    scene.remove(splat);
    splat.dispose();
    splat = null;
  }
}

// 运行时调整渲染质量（Spark 参数每帧同步到 uniform，可即时生效）
function setQuality(opts) {
  if (spark) {
    if (opts.maxStdDev != null) spark.maxStdDev = opts.maxStdDev;
    if (opts.minPixelRadius != null) spark.minPixelRadius = opts.minPixelRadius;
    if (opts.maxPixelRadius != null) spark.maxPixelRadius = opts.maxPixelRadius;
  }
  if (renderer && opts.pixelRatio != null) {
    renderer.setPixelRatio(Math.max(0.5, Math.min(opts.pixelRatio, 3)));
  }
}

window.OSTViewer = {
  init, setModel, loadPly, clearModel, setView, toggleGrid, resetView, flipY, setQuality,
};
