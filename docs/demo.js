(function () {
  "use strict";
  var N = 8;
  var CELL = 48;
  var DPR = Math.min(window.devicePixelRatio || 1, 2);
  var C = {
    bg: "#0c281d", grid: "#2c5a46", text: "#F4F8DC", neutral: "#1B4A38",
    prev: "rgba(191,216,203,0.45)", cur: "#F2A33A", machine: "#F4F8DC"
  };
  var data = null, seeds = {}, sol = null, step = 0, timer = null;
  var el = {};

  function $(id) { return document.getElementById(id); }

  function showError(msg) {
    var box = $("demo-app");
    box.innerHTML = '<div class="err">' + msg + "</div>";
  }

  function load() {
    fetch("demo_data.json")
      .then(function (r) {
        if (!r.ok) { throw new Error("HTTP " + r.status); }
        return r.json();
      })
      .then(init)
      .catch(function (e) {
        showError("No se pudieron cargar los datos de la demo (" + e.message + "). " +
          "Si abriste el archivo con doble clic, servilo con <code>python -m http.server</code> y abrí <code>http://localhost:8000</code>.");
      });
  }

  function init(d) {
    data = d;
    d.seeds.forEach(function (s) { seeds[s.seed] = s; });
    el.seed = $("seed");
    el.slider = $("paso");
    el.play = $("play");
    el.speed = $("vel");
    el.cA = $("canvas-agente");
    el.cG = $("canvas-greedy");
    el.cChart = $("canvas-curva");
    el.info = $("info-paso");
    el.kpi = $("kpi");
    el.resumen = $("resumen-global");

    d.seeds.forEach(function (s) {
      var o = document.createElement("option");
      o.value = s.seed; o.textContent = "Seed " + s.seed;
      el.seed.appendChild(o);
    });

    el.seed.addEventListener("change", function () { setSeed(parseInt(el.seed.value, 10)); });
    $("prev").addEventListener("click", function () { moveSeed(-1); });
    $("next").addEventListener("click", function () { moveSeed(1); });
    $("azar").addEventListener("click", function () {
      var arr = data.seeds, pick = arr[Math.floor(Math.random() * arr.length)].seed;
      setSeed(pick);
    });
    el.slider.addEventListener("input", function () { stop(); setStep(parseInt(el.slider.value, 10)); });
    el.play.addEventListener("click", togglePlay);
    $("reiniciar").addEventListener("click", function () { stop(); setStep(0); });
    $("final").addEventListener("click", function () { stop(); setStep(sol.agente.orden.length); });

    resumenGlobal();
    setSeed(data.seeds[0].seed);
  }

  function moveSeed(delta) {
    var list = data.seeds.map(function (s) { return s.seed; });
    var i = list.indexOf(sol.seed) + delta;
    if (i < 0) { i = list.length - 1; }
    if (i >= list.length) { i = 0; }
    setSeed(list[i]);
  }

  function setSeed(seed) {
    stop();
    sol = seeds[seed];
    el.seed.value = seed;
    el.slider.max = sol.agente.orden.length;
    setStep(0);
    kpis();
  }

  function setStep(k) {
    step = k;
    el.slider.value = k;
    draw();
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
    if (el.play) { el.play.textContent = "Reproducir"; }
  }

  function togglePlay() {
    if (timer) { stop(); return; }
    if (step >= sol.agente.orden.length) { setStep(0); }
    el.play.textContent = "Pausar";
    var ms = parseInt(el.speed.value, 10);
    timer = setInterval(function () {
      if (step >= sol.agente.orden.length) { stop(); return; }
      setStep(step + 1);
    }, ms);
  }

  function fmt(x) { return x.toFixed(2).replace(/\.00$/, ""); }

  function kpis() {
    var a = sol.agente.costo, g = sol.greedy.costo;
    var dif = 100 * (g - a) / g;
    var cls = dif > 0.0005 ? "good" : (dif < -0.0005 ? "bad" : "");
    var txt = dif > 0.0005 ? "el agente ahorra " + dif.toFixed(2) + "%" :
      (dif < -0.0005 ? "el agente gasta " + (-dif).toFixed(2) + "% más" : "empate");
    el.kpi.innerHTML =
      "<span>Agente <b>" + fmt(a) + "</b></span>" +
      "<span>Greedy <b>" + fmt(g) + "</b></span>" +
      '<span>Resultado <b class="' + cls + '">' + txt + "</b></span>" +
      "<span>Viajes <b>" + sol.agente.orden.length + "</b></span>";
  }

  function resumenGlobal() {
    var n = data.seeds.length, sa = 0, sg = 0, w = 0, t = 0, l = 0;
    data.seeds.forEach(function (s) {
      var a = s.agente.costo, g = s.greedy.costo;
      sa += a; sg += g;
      if (a < g - 1e-9) { w++; } else if (Math.abs(a - g) <= 1e-9) { t++; } else { l++; }
    });
    el.resumen.innerHTML =
      "En los " + n + " terrenos de test, este modelo tiene un costo medio de <b>" + (sa / n).toFixed(2) +
      "</b> contra <b>" + (sg / n).toFixed(2) + "</b> del greedy. Es mejor en " + w + " terrenos, empata en " + t +
      " y es peor en " + l + ".";
  }

  /* ---------- dibujo ---------- */

  function prepCanvas(c, w, h) {
    c.width = Math.round(w * DPR);
    c.height = Math.round(h * DPR);
    var ctx = c.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    return ctx;
  }

  function diffGrid(k, orden) {
    var g = [], i, j;
    for (i = 0; i < N; i++) {
      g.push([]);
      for (j = 0; j < N; j++) { g[i].push(sol.alturas[i][j] - sol.objetivo[i][j]); }
    }
    for (i = 0; i < k; i++) {
      var v = orden[i];
      g[v[0][0]][v[0][1]] -= v[2];
      g[v[1][0]][v[1][1]] += v[2];
    }
    return g;
  }

  function cellColor(v, maxAbs) {
    if (v === 0) { return C.neutral; }
    var t = Math.min(1, Math.abs(v) / maxAbs);
    var a = 0.25 + 0.65 * t;
    return v > 0 ? "rgba(232,121,90," + a + ")" : "rgba(95,199,168," + a + ")";
  }

  function center(p) { return [(p[1] + 0.5) * CELL, (p[0] + 0.5) * CELL]; }

  function arrow(ctx, from, to, color, width, dash) {
    ctx.save();
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.setLineDash(dash || []);
    ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke();
    ctx.setLineDash([]);
    var dx = to[0] - from[0], dy = to[1] - from[1], len = Math.hypot(dx, dy);
    if (len > 6) {
      var ux = dx / len, uy = dy / len, s = 7;
      ctx.beginPath();
      ctx.moveTo(to[0], to[1]);
      ctx.lineTo(to[0] - ux * s + uy * s * 0.55, to[1] - uy * s - ux * s * 0.55);
      ctx.lineTo(to[0] - ux * s - uy * s * 0.55, to[1] - uy * s + ux * s * 0.55);
      ctx.closePath(); ctx.fill();
    }
    ctx.restore();
  }

  function drawPanel(canvas, orden, maxAbs) {
    var size = N * CELL;
    var ctx = prepCanvas(canvas, size, size);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, size, size);
    var g = diffGrid(step, orden), i, j;
    ctx.font = "13px Inter, system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        ctx.fillStyle = cellColor(g[i][j], maxAbs);
        ctx.fillRect(j * CELL + 1, i * CELL + 1, CELL - 2, CELL - 2);
        if (g[i][j] !== 0) {
          ctx.fillStyle = C.text;
          ctx.fillText(String(g[i][j]), (j + 0.5) * CELL, (i + 0.5) * CELL);
        }
      }
    }
    var k, pos = [0, 0];
    for (k = 0; k < step - 1; k++) {
      arrow(ctx, center(orden[k][0]), center(orden[k][1]), C.prev, 1.5);
    }
    if (step > 0) {
      var cur = orden[step - 1];
      var prevPos = step >= 2 ? orden[step - 2][1] : [0, 0];
      var a = center(prevPos), o = center(cur[0]), d = center(cur[1]);
      if (a[0] !== o[0] || a[1] !== o[1]) { arrow(ctx, a, o, C.cur, 2, [5, 4]); }
      arrow(ctx, o, d, C.cur, 3.5);
      pos = cur[1];
    }
    var m = center(pos);
    ctx.beginPath(); ctx.arc(m[0], m[1], 7, 0, Math.PI * 2);
    ctx.fillStyle = C.machine; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = C.bg; ctx.stroke();
  }

  function cumulative(orden) {
    var out = [0], s = 0;
    orden.forEach(function (v) { s += v[3]; out.push(s); });
    return out;
  }

  function drawChart() {
    var W = 720, H = 220, padL = 46, padR = 14, padT = 14, padB = 30;
    var ctx = prepCanvas(el.cChart, W, H);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    var ca = cumulative(sol.agente.orden), cg = cumulative(sol.greedy.orden);
    var n = ca.length - 1;
    var ymax = Math.max(ca[n], cg[n]) * 1.05;
    function X(i) { return padL + (W - padL - padR) * (i / n); }
    function Y(v) { return H - padB - (H - padT - padB) * (v / ymax); }
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    ctx.fillStyle = "#BFD8CB"; ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    var t, ticks = 4;
    for (t = 0; t <= ticks; t++) {
      var v = ymax * t / ticks, y = Y(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillText(Math.round(v), padL - 6, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("viaje", (padL + W - padR) / 2, H - 16);
    function line(arr, color, upto) {
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.beginPath();
      for (var i = 0; i <= n; i++) {
        if (i === 0) { ctx.moveTo(X(i), Y(arr[i])); } else { ctx.lineTo(X(i), Y(arr[i])); }
      }
      ctx.globalAlpha = 0.28; ctx.stroke(); ctx.globalAlpha = 1;
      ctx.beginPath();
      for (i = 0; i <= upto; i++) {
        if (i === 0) { ctx.moveTo(X(i), Y(arr[i])); } else { ctx.lineTo(X(i), Y(arr[i])); }
      }
      ctx.stroke();
      ctx.beginPath(); ctx.arc(X(upto), Y(arr[upto]), 4, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
    }
    line(cg, "#9DB4A9", step);
    line(ca, "#F4F8DC", step);
    ctx.strokeStyle = "rgba(244,248,220,0.25)"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(X(step), padT); ctx.lineTo(X(step), H - padB); ctx.stroke(); ctx.setLineDash([]);
  }

  function draw() {
    var maxAbs = 1, i, j;
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        maxAbs = Math.max(maxAbs, Math.abs(sol.alturas[i][j] - sol.objetivo[i][j]));
      }
    }
    drawPanel(el.cA, sol.agente.orden, maxAbs);
    drawPanel(el.cG, sol.greedy.orden, maxAbs);
    drawChart();
    var ca = cumulative(sol.agente.orden), cg = cumulative(sol.greedy.orden), n = sol.agente.orden.length;
    var txt = "Viaje " + step + " de " + n + " · costo acumulado: agente " + ca[step] + ", greedy " + cg[step];
    if (step === n) { txt += " · terreno nivelado"; }
    el.info.textContent = txt;
  }

  document.addEventListener("DOMContentLoaded", load);
})();
