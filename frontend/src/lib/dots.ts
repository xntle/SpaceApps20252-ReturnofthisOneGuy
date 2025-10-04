export type Dot = {
  x: number;
  y: number;
  size: number;
  alpha: number;
  tw?: number;
  tws?: number;
};
export type PlanetDot = {
  x0: number;
  y0: number;
  size: number;
  ring: number;
  phase: number;
};
export type PlanetCfg = {
  radius: number;
  rings: number;
  eccentricity: number;
  speed: number;
};
export type Planet = {
  cx: number;
  cy: number;
  dots: PlanetDot[];
  cfg: PlanetCfg;
};

export type SceneConfig = {
  DPR: number;
  width: number;
  height: number;
  center: { x: number; y: number };
  mouse: { x: number; y: number };
  stars: Dot[];
  planets: Planet[];
  settings: {
    starDensity: number;
    glow: boolean;
    swirlStrength: number;
    maxShadowBlur: number;
  };
};

export function buildScene(scene: SceneConfig) {
  const { width: w, height: h, stars, planets, center } = scene;
  stars.length = 0;
  planets.length = 0;

  // Stars — density based on area
  const starCount = Math.floor(w * h * scene.settings.starDensity);
  for (let i = 0; i < starCount; i++) {
    stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      size: Math.random() * 1.4 + 0.2,
      alpha: Math.random() * 0.5 + 0.3,
      tw: Math.random() * 2 * Math.PI,
      tws: Math.random() * 0.004 + 0.001,
    });
  }

  // Planets
  const base = Math.min(w, h);
  const configs: PlanetCfg[] = [
    { radius: base * 0.18, rings: 3, eccentricity: 0.06, speed: 0.15 },
    { radius: base * 0.28, rings: 4, eccentricity: 0.12, speed: 0.08 },
    { radius: base * 0.42, rings: 5, eccentricity: 0.18, speed: 0.05 },
  ];
  for (const cfg of configs) planets.push(makePlanet(center.x, center.y, cfg));
}

export function drawFrame(
  ctx: CanvasRenderingContext2D,
  scene: SceneConfig,
  t: number
) {
  const { width: w, height: h, center } = scene;

  ctx.clearRect(0, 0, w, h);

  // Soft vignette
  const g = ctx.createRadialGradient(
    center.x,
    center.y,
    0,
    center.x,
    center.y,
    Math.max(w, h) * 0.7
  );
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, "rgba(0,0,0,0.5)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  // Stars
  for (let i = 0; i < scene.stars.length; i++) drawStar(ctx, scene.stars[i]);

  // Planets
  ctx.save();
  for (const p of scene.planets) drawPlanet(ctx, p, t, scene);
  ctx.restore();
}

function drawStar(ctx: CanvasRenderingContext2D, s: Dot) {
  // advance phase safely without illegal assignment targets
  s.tw = (s.tw ?? 0) + (s.tws ?? 0);
  const twinkle = 0.6 + Math.sin(s.tw) * 0.4;
  ctx.globalAlpha = s.alpha * twinkle;
  ctx.beginPath();
  ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.globalAlpha = 1;
}

function swirl(scene: SceneConfig, x: number, y: number) {
  // Predictive swirl field — nudges points toward a clean orbit
  const { center, mouse, settings } = scene;
  const dx = x - center.x;
  const dy = y - center.y;
  const rotX = -dy;
  const rotY = dx;
  return {
    x: x + rotX * settings.swirlStrength + mouse.x * 0.0006,
    y: y + rotY * settings.swirlStrength + mouse.y * 0.0006,
  };
}

function makePlanet(cx: number, cy: number, cfg: PlanetCfg): Planet {
  const dots: PlanetDot[] = [];
  const total = cfg.rings;

  for (let ring = 0; ring < total; ring++) {
    const n = Math.floor(120 + ring * 40); // dots per ring
    const R = cfg.radius * (0.45 + (ring / (total - 1 || 1)) * 0.7);
    for (let i = 0; i < n; i++) {
      const theta = (i / n) * Math.PI * 2;
      const e = cfg.eccentricity * (0.6 + ring / total);
      const x0 = cx + Math.cos(theta) * R * (1 + e);
      const y0 = cy + Math.sin(theta) * R * (1 - e);
      dots.push({
        x0,
        y0,
        size: 1.1 + Math.random() * 1.4,
        ring,
        phase: Math.random() * Math.PI * 2,
      });
    }
  }

  // Central dot-sphere (Fibonacci lattice projection)
  const sphereR = cfg.radius * 0.42;
  const sphereN = 800;
  for (let i = 0; i < sphereN; i++) {
    const k = i + 0.5;
    const phi = Math.acos(1 - (2 * k) / sphereN);
    const theta = Math.PI * (1 + Math.sqrt(5)) * k;
    const x = Math.cos(theta) * Math.sin(phi);
    const y = Math.sin(theta) * Math.sin(phi);
    dots.push({
      x0: cx + x * sphereR,
      y0: cy + y * sphereR,
      size: 1.2,
      ring: -1,
      phase: Math.random() * Math.PI * 2,
    });
  }

  return { cx, cy, dots, cfg };
}

function drawPlanet(
  ctx: CanvasRenderingContext2D,
  p: Planet,
  t: number,
  scene: SceneConfig
) {
  for (const d of p.dots) {
    // time-based angular rotation per ring + per-dot phase
    const base = p.cfg.speed; // planet base speed
    const ringFactor = (d.ring + 1) * 0.06; // outer rings move a bit faster
    const angle = t * (base + ringFactor) + d.phase * 0.2;

    // rotate original point around planet center by angle
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    const rx = (d.x0 - p.cx) * cosA - (d.y0 - p.cy) * sinA + p.cx;
    const ry = (d.x0 - p.cx) * sinA + (d.y0 - p.cy) * cosA + p.cy;

    // apply predictive swirl + pointer parallax
    const s = swirl(scene, rx, ry);

    ctx.beginPath();
    ctx.arc(s.x, s.y, d.size, 0, Math.PI * 2);

    // subtle depth/alpha falloff from center
    const dist = Math.hypot(s.x - p.cx, s.y - p.cy);
    const inner = p.cfg.radius * 0.6;

    let alpha = 0.6;
    if (d.ring === -1) {
      alpha = 0.75 * Math.max(0.25, 1 - dist / (inner * 1.2));
    } else {
      alpha = 0.45 * Math.max(0.15, 1 - dist / (p.cfg.radius * 1.3));
    }

    ctx.globalAlpha = alpha;
    ctx.fillStyle = "#ffffff";
    ctx.shadowColor = scene.settings.glow
      ? "rgba(255,255,255,0.35)"
      : "transparent";
    ctx.shadowBlur = scene.settings.glow ? scene.settings.maxShadowBlur : 0;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
  }
}
