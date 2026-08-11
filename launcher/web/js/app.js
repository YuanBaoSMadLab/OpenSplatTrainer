/* OpenSplat3DTrainer_Launcher 前端逻辑：任务队列 + 每任务独立参数 + 3D 渲染 */
(function () {
  "use strict";

  const W2 = window.chrome && window.chrome.webview;
  // 直接传对象：JSON 字符串会被再包一层引号，C++ 收到的就是字符串
  function send(obj) { if (W2) W2.postMessage(obj); }

  const state = {
    tasks: new Map(),   // tag → {tag, path, thumb, st, params, error, count, seconds}
    seq: 0,             // 任务序号
    selected: null,     // 当前选中任务 tag（null = 编辑默认参数）
    defaults: { steps: 20, guidance: 3.0, shift: 3.0, num: 262144, seed: 42, rmbg_res: 1024, img_res: 1024 },
    outDir: "output",
    savePly: true,
    saveSplat: false,
    cuda: false,
    vram: { free: 0, total: 0, auto: 262144 },
    vramMode: "high",
  };

  const $ = (id) => document.getElementById(id);
  const ST_LABEL = { wait: "待生成", run: "生成中", ok: "完成", err: "失败" };

  /* ---------- 工具 ---------- */
  function basename(p) {
    const s = p.replace(/\\/g, "/");
    return s.split("/").pop();
  }
  function fmtCount(n) { return Number(n).toLocaleString("en-US"); }
  function now() {
    const d = new Date();
    return String(d.getHours()).padStart(2, "0") + ":" +
           String(d.getMinutes()).padStart(2, "0") + ":" +
           String(d.getSeconds()).padStart(2, "0");
  }
  function toast(text, isErr) {
    const t = $("toast");
    t.textContent = text;
    t.className = "toast show" + (isErr ? " err" : "");
    clearTimeout(toast._h);
    toast._h = setTimeout(() => (t.className = "toast"), 2600);
  }
  function b64ToBytes(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  /* ---------- 日志 ---------- */
  const LOG_MAX = 800;
  function addLog(text, isErr) {
    const list = $("logList");
    const line = document.createElement("div");
    line.className = "log-line" + (isErr ? " err" : "");
    line.innerHTML = "<span class='t'>[" + now() + "]</span> " + escapeHtml(text);
    list.appendChild(line);
    while (list.childNodes.length > LOG_MAX) list.removeChild(list.firstChild);
    if ($("logAuto").checked) list.scrollTop = list.scrollHeight;
    if (isErr && state.cuda) toast(text, true);
  }

  // console 输出转发到 UI 日志（诊断）
  const _origLog = console.log, _origErr = console.error;
  console.log = function () {
    const parts = Array.from(arguments).map((a) =>
      typeof a === "string" ? a : (a && a.message) ? a.message : String(a));
    if (parts.join(" ").indexOf("[viewer]") >= 0) addLog(parts.join(" "), false);
    _origLog.apply(console, arguments);
  };
  console.error = function () {
    const parts = Array.from(arguments).map((a) =>
      typeof a === "string" ? a : (a && a.message) ? a.message : String(a));
    addLog("渲染错误: " + parts.join(" "), true);
    _origErr.apply(console, arguments);
  };
  function applyLogFilter() {
    const f = $("logFilter").value.toLowerCase();
    document.querySelectorAll(".log-line").forEach((el) => {
      el.style.display = f && el.textContent.toLowerCase().indexOf(f) < 0 ? "none" : "";
    });
  }

  /* ---------- 任务队列 ---------- */
  function addTasks(paths, thumbs) {
    if (!paths || !paths.length) return;
    const added = [];
    (paths || []).forEach((p, i) => {
      if (!p) return;
      const tag = "t" + (++state.seq) + "_" + Date.now().toString(36);
      const task = {
        tag, path: p, thumb: (thumbs && thumbs[i]) || "",
        st: "wait", error: "", count: 0, seconds: 0,
        params: Object.assign({}, state.defaults),
      };
      state.tasks.set(tag, task);
      added.push(tag);
    });
    if (!added.length) return;
    renderQueue();
    selectTask(added[added.length - 1]);
    addLog("已添加 " + added.length + " 个任务", false);
    toast("已添加 " + added.length + " 个任务");
  }

  function renderQueue() {
    const box = $("taskList");
    box.innerHTML = "";
    if (!state.tasks.size) {
      box.innerHTML = "<div class='empty'>（空 &middot; 添加图片或拖入窗口）</div>";
      return;
    }
    state.tasks.forEach((t) => {
      const it = document.createElement("div");
      it.className = "task-item" + (t.tag === state.selected ? " sel" : "");
      const img = t.thumb
        ? "<img class='thumb' src='data:image/png;base64," + t.thumb + "'>"
        : "<span class='thumb thumb-ph'></span>";
      const p = t.params;
      const stText = ST_LABEL[t.st] || t.st;
      const meta = "<span class='st-" + t.st + "'>" + stText + "</span> " +
        p.steps + "步 &middot; " + p.guidance.toFixed(1) + " &middot; " + fmtCount(p.num) +
        (t.st === "ok" ? " &middot; " + fmtCount(t.count) + "高斯" : "");
      // 运行中显示阶段进度条
      const progPct = (t.progress && t.progress.pct) || 0;
      const progHtml = t.st === "run"
        ? "<div class='task-prog'><div class='fill' style='width:" + progPct + "%'></div></div>" +
          "<span class='task-plabel'>" + escapeHtml((t.progress && t.progress.label) || "准备中…") + " " + progPct + "%</span>"
        : "";
      it.innerHTML = img +
        "<div class='task-info'>" +
        "<span class='task-name' title='" + escapeHtml(t.path) + "'>" + escapeHtml(basename(t.path)) + "</span>" +
        "<span class='task-meta'>" + meta + "</span>" + progHtml +
        "</div>" +
        "<span class='task-dot st-" + t.st + "'></span>";
      t._el = it;
      it.onclick = () => selectTask(t.tag);
      if (t.st === "ok") {
        it.title = "点击选中 · 双击在 3D 视口显示";
        it.ondblclick = () => send({ cmd: "getModel", path: t.path });
      } else {
        it.title = "点击选中该任务，在右侧调整它自己的参数";
      }
      box.appendChild(it);
    });
    $("btnStart").disabled = !state.cuda || ![...state.tasks.values()].some((t) => t.st === "wait");
  }

  function selectTask(tag) {
    state.selected = tag;
    renderQueue();
    renderParamScope();
  }

  /* ---------- 参数 ---------- */
  function getTarget() {
    if (state.selected) {
      const t = state.tasks.get(state.selected);
      if (t) return t.params;
    }
    return state.defaults;
  }
  function getTargetTask() {
    return state.selected ? state.tasks.get(state.selected) : null;
  }
  function readCtrl() {
    return {
      img_res: +$("imgRes").value,
      rmbg_res: +$("rmbgRes").value,
      steps: +$("steps").value,
      guidance: (+$("guidance").value) / 10,
      shift: (+$("shift").value) / 10,
      seed: +$("seed").value,
      num: +$("num").value,
    };
  }
  function writeCtrl(p) {
    $("imgRes").value = String(p.img_res || 1024);
    $("rmbgRes").value = String(p.rmbg_res || 1024);
    $("steps").value = String(p.steps || 20);
    $("guidance").value = String(Math.round((p.guidance || 3) * 10));
    $("shift").value = String(Math.round((p.shift || 3) * 10));
    $("seed").value = String(p.seed || 42);
    $("num").value = String(p.num || 262144);
    syncOutputs(p);
  }
  function syncOutputs(p) {
    $("imgResOut").textContent = p.img_res;
    $("rmbgOut").textContent = p.rmbg_res;
    $("stepsOut").textContent = p.steps;
    $("guidanceOut").textContent = p.guidance.toFixed(1);
    $("shiftOut").textContent = p.shift.toFixed(1);
    $("seedOut").textContent = p.seed;
    $("numOut").textContent = fmtCount(p.num);
  }
  function renderParamScope() {
    const t = getTargetTask();
    $("paramScope").textContent = t ? basename(t.path) : "默认";
    $("paramHint").style.display = t ? "none" : "";
    $("btnDup").disabled = !t;
    $("btnRemove").disabled = !t;
    writeCtrl(getTarget());
  }
  function onParamInput() {
    Object.assign(getTarget(), readCtrl());
    syncOutputs(getTarget());
  }
  function onParamChange() {
    Object.assign(getTarget(), readCtrl());
    syncOutputs(getTarget());
    renderQueue();
  }
  function randomSeed() {
    const v = Math.floor(Math.random() * 100000);
    Object.assign(getTarget(), { seed: v });
    $("seed").value = String(v);
    syncOutputs(getTarget());
  }

  /* ---------- 显存 ---------- */
  function renderVram(v) {
    state.vram = v;
    const used = v.total > 0 ? v.total - v.free : 0;
    const pct = v.total > 0 ? (used / v.total) * 100 : 0;
    $("vramBar").style.width = pct + "%";
    $("vramText").textContent = v.cuda
      ? (used / 1024).toFixed(1) + " / " + (v.total / 1024).toFixed(1) + " GB" +
        "  · 建议 " + fmtCount(v.auto) + " 高斯 · " + (v.rmbg || 1024) + "px" +
        (v.lazy ? " · 按需加载" : " · 模型常驻")
      : "无 CUDA";
    $("vramStatus").textContent = v.cuda ? (v.total / 1024).toFixed(0) + "GB VRAM" : "NO CUDA";
    $("vramStatus").className = "chip" + (v.cuda ? " ok" : " bad");
  }

  // 首次启动选显存模式：C++ 保存配置后关闭覆盖层
  function chooseVramMode(mode) {
    state.vramMode = mode;
    $("setupStatus").textContent = "正在应用…";
    send({ cmd: "setVramMode", mode });
  }

  /* ---------- 生成 ---------- */
  function startGenerate() {
    if (!state.cuda) return toast("未检测到 CUDA，无法生成", true);
    const waiting = [...state.tasks.values()].filter((t) => t.st === "wait");
    if (!waiting.length) return toast("没有待生成的任务（可复制已完成任务再次生成）");
    const tasks = waiting.map((t) => {
      const p = t.params;
      return {
        tag: t.tag,
        path: t.path,
        steps: p.steps,
        guidance: p.guidance,
        shift: p.shift,
        num: p.num,
        seed: p.seed,
        rmbg_res: p.rmbg_res,
        img_res: p.img_res,
        out_ply: state.savePly ? state.outDir.replace(/\\$/, "") + "\\" + outName(t, waiting) : "",
        out_splat: state.saveSplat ? state.outDir.replace(/\\$/, "") + "\\" + outName(t, waiting, ".splat") : "",
      };
    });
    waiting.forEach((t) => { t.st = "run"; });
    renderQueue();
    send({ cmd: "start", tasks });
    toast("已提交 " + tasks.length + " 个任务");
  }

  // 同一图片多次生成自动加序号，避免输出互相覆盖
  function outName(task, batch, ext) {
    ext = ext || ".ply";
    const base = basename(task.path).replace(/\.[^.]+$/, "");
    const all = [...state.tasks.values()].concat(batch);
    const before = all.filter((t) => t.path === task.path && t !== task).length;
    return before === 0 ? base + ext : base + " (" + (before + 1) + ")" + ext;
  }

  /* ---------- 事件处理 ---------- */
  function onMessage(e) {
    // WebView2 的 e.data 已是解析后的对象；兼容字符串形式
    let j = e.data;
    if (typeof j === "string") {
      try { j = JSON.parse(j); } catch (_) { return; }
    }
    if (!j || typeof j !== "object") return;
    switch (j.event) {
      case "init":
        state.cuda = !!j.cuda;
        $("cudaStatus").textContent = j.cuda ? "CUDA OK" : "NO CUDA";
        $("cudaStatus").className = "chip" + (j.cuda ? " ok" : " bad");
        if (!state.cuda) {
          $("setupOverlay").classList.add("hidden");
        } else if (!j.hasConfig) {
          $("setupOverlay").classList.remove("hidden");
        } else {
          $("setupOverlay").classList.add("hidden");
          $("vramMode").value = j.vramMode === "low" ? "low" : "high";
          state.vramMode = $("vramMode").value;
        }
        if (!j.cuda) toast("未检测到 NVIDIA 显卡，生成功能不可用", true);
        addLog("模型目录: " + j.modelsDir);
        renderQueue();
        send({ cmd: "vram" });
        break;
      case "picked":
        addTasks(j.paths, j.thumbs);
        break;
      case "log":
        addLog(j.text, !!j.err);
        break;
      case "task": {
        const t = state.tasks.get(j.tag);
        if (t) {
          t.st = j.ok ? "ok" : (j.error ? "err" : "run");
          t.error = j.error || "";
          t.count = j.count;
          t.seconds = j.seconds;
          renderQueue();
        }
        break;
      }
      case "progress": {
        const t = state.tasks.get(j.tag);
        if (!t) break;
        t.progress = { pct: j.pct, label: j.label };
        // 局部更新进度条，避免整列重绘抖动
        const el = t._el;
        if (el && el.isConnected && t.st === "run") {
          const fill = el.querySelector(".task-prog .fill");
          const pl = el.querySelector(".task-plabel");
          if (fill) fill.style.width = j.pct + "%";
          if (pl) pl.textContent = j.label + " " + j.pct + "%";
        } else {
          renderQueue();
        }
        break;
      }
      case "model":
        $("hudModel").textContent = (j.name || "").toUpperCase() + " · " + fmtCount(j.count);
        $("hudState").className = "hud-dot ok";
        $("hudStateText").textContent = "MODEL READY";
        send({ cmd: "getModel" });
        break;
      case "modelData":
        if (j.ok && window.OSTViewer) {
          const bytes = b64ToBytes(j.data);
          const n = Math.floor(bytes.length / 32);
          if (n > 0) {
            window.OSTViewer.setModel(bytes, n);
            $("hudCount").textContent = fmtCount(n) + " GAUSSIANS";
            toast("已加载：" + j.name + " · " + fmtCount(n) + " 高斯");
          }
        }
        break;
      case "plyLoaded":
        if (window.OSTViewer) {
          const bytes = b64ToBytes(j.data);
          window.OSTViewer.loadPly(bytes);
          $("hudModel").textContent = String(j.name || "PLY").toUpperCase();
          $("hudState").className = "hud-dot ok";
          $("hudStateText").textContent = "PLY LOADED";
          toast("已导入：" + (j.name || "PLY"));
        }
        break;
      case "vramModeSaved":
        $("vramMode").value = j.mode === "low" ? "low" : "high";
        state.vramMode = $("vramMode").value;
        if (j.restart) {
          $("setupOverlay").classList.remove("hidden");
          $("setupStatus").textContent = "已保存，正在重启…";
          break;
        }
        $("setupOverlay").classList.add("hidden");
        toast("已保存：" + (j.mode === "low" ? "低显存（按需加载）" : "高显存（模型常驻）"));
        break;
      case "vram":
        renderVram(j);
        break;
    }
  }

  /* ---------- 初始化 ---------- */
  function init() {
    // viewer.js（module script）就绪后再初始化 3D 渲染器
    const startViewer = () => {
      try {
        window.OSTViewer.init($("viewer"));
      } catch (e) {
        addLog("3D 渲染器初始化失败: " + (e && e.message), true);
      }
    };
    if (window.OSTViewer) {
      startViewer();
    } else {
      addLog("3D 渲染器加载中，重试中…", false);
      let tries = 0;
      const timer = setInterval(() => {
        tries++;
        if (window.OSTViewer) {
          clearInterval(timer);
          addLog("3D 渲染器就绪（重试 " + tries + " 次）", false);
          startViewer();
        } else if (tries >= 20) {
          clearInterval(timer);
          addLog("3D 渲染器 10 秒内未就绪，请检查日志中的 JS错误", true);
        }
      }, 500);
    }

    // 任务队列
    $("btnPick").onclick = () => send({ cmd: "pickImages", multi: true });
    $("btnStart").onclick = startGenerate;
    $("btnClear").onclick = () => {
      [...state.tasks.keys()].forEach((k) => {
        const t = state.tasks.get(k);
        if (t.st === "ok" || t.st === "err") state.tasks.delete(k);
      });
      if (state.selected && !state.tasks.has(state.selected)) state.selected = null;
      renderQueue();
      renderParamScope();
    };

    // 参数（编辑默认 / 选中任务）
    ["imgRes", "rmbgRes", "num"].forEach((id) => $(id).addEventListener("change", onParamChange));
    ["steps", "guidance", "shift", "seed"].forEach((id) => $(id).addEventListener("input", onParamInput));
    $("btnSeedRand").onclick = randomSeed;
    $("btnDup").onclick = () => {
      const t = getTargetTask();
      if (!t) return;
      const tag = "t" + (++state.seq) + "_" + Date.now().toString(36);
      const copy = Object.assign({}, t, {
        tag, st: "wait", error: "", count: 0, seconds: 0,
        params: Object.assign({}, t.params),
      });
      state.tasks.set(tag, copy);
      renderQueue();
      selectTask(tag);
      toast("已复制任务，可修改参数后再次生成");
    };
    $("btnRemove").onclick = () => {
      if (!state.selected) return;
      state.tasks.delete(state.selected);
      state.selected = null;
      renderQueue();
      renderParamScope();
    };

    // 输出 / 显存
    $("outDir").onchange = (e) => { state.outDir = e.target.value.trim() || "output"; };
    $("savePly").onchange = (e) => { state.savePly = e.target.checked; };
    $("saveSplat").onchange = (e) => { state.saveSplat = e.target.checked; };
    $("vramMode").onchange = (e) => {
      state.vramMode = e.target.value;
      send({ cmd: "setVramMode", mode: state.vramMode });
      toast("已保存：" + (state.vramMode === "low" ? "低显存（按需加载）" : "高显存（模型常驻）"));
    };
    $("setupHigh").onclick = () => chooseVramMode("high");
    $("setupLow").onclick = () => chooseVramMode("low");

    // 视口工具：三视图 / 网格 / 归位 / 翻转 / 导入PLY
    document.querySelectorAll("[data-view]").forEach((b) => {
      b.onclick = () => { if (window.OSTViewer) window.OSTViewer.setView(b.dataset.view); };
    });
    $("btnGrid").onclick = () => { if (window.OSTViewer) window.OSTViewer.toggleGrid(); };
    $("btnReset").onclick = () => { if (window.OSTViewer) window.OSTViewer.resetView(); };
    $("btnFlip").onclick = () => { if (window.OSTViewer) window.OSTViewer.flipY(); };
    $("btnOpenPly").onclick = () => send({ cmd: "openPly" });

    // 渲染质量（实时生效）
    $("rStd").addEventListener("input", () => {
      const v = (+$("rStd").value) / 10;
      $("rStdOut").textContent = v.toFixed(1);
      if (window.OSTViewer) window.OSTViewer.setQuality({ maxStdDev: v });
    });
    $("rPx").addEventListener("change", () => {
      const v = +$("rPx").value;
      $("rPxOut").textContent = v + "x";
      if (window.OSTViewer) window.OSTViewer.setQuality({ pixelRatio: v });
    });

    // 日志
    $("logFilter").oninput = applyLogFilter;

    // 拖放：优先 File.path（WebView2 扩展），不可用时读文件内容走 base64 交 C++ 落盘
    const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };
    window.addEventListener("dragover", prevent);
    window.addEventListener("dragenter", prevent);
    window.addEventListener("drop", (e) => {
      prevent(e);
      const files = e.dataTransfer && e.dataTransfer.files ? Array.from(e.dataTransfer.files) : [];
      if (!files.length) { addLog("拖放未包含文件（请从资源管理器拖入图片）", true); return; }
      const paths = files.map((f) => (f && f.path) || "").filter(Boolean);
      if (paths.length) {
        addLog("拖入 " + paths.length + " 张图片（直接路径）", false);
        send({ cmd: "addImages", paths });
        return;
      }
      // File.path 不可用 → base64 交宿主落盘（imported/ 目录）
      addLog("读取拖入文件内容（" + files.length + " 个）…", false);
      const items = [];
      let done = 0;
      files.forEach((f) => {
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const u8 = new Uint8Array(reader.result);
            let bin = "";
            const CH = 0x8000;
            for (let i = 0; i < u8.length; i += CH) {
              bin += String.fromCharCode.apply(null, u8.subarray(i, Math.min(i + CH, u8.length)));
            }
            items.push({ name: f.name || ("drag_" + (items.length + 1) + ".img"), data: btoa(bin) });
          } catch (err) { addLog("读取文件失败: " + (err && err.message), true); }
          done++;
          if (done === files.length) {
            if (items.length) {
              addLog("拖入 " + items.length + " 张图片（内容模式）", false);
              send({ cmd: "addImagesData", files: items });
            } else {
              addLog("拖放内容为空", true);
            }
          }
        };
        reader.onerror = () => {
          done++;
          if (done === files.length && items.length) send({ cmd: "addImagesData", files: items });
        };
        reader.readAsArrayBuffer(f);
      });
    });

    renderQueue();
    renderParamScope();
    addLog("UI 初始化完成", false);

    if (W2) {
      W2.addEventListener("message", onMessage);
      send({ cmd: "ready" });
      setInterval(() => send({ cmd: "vram" }), 2500);
    } else {
      addLog("浏览器预览模式（无宿主）", true);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
