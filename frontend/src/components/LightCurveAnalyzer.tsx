// components/LightCurveAnalyzer.tsx
"use client";

import React, { useMemo, useRef, useState } from "react";

type LCPoint = { t: number; f: number; e?: number };

function parseCSV(text: string): LCPoint[] {
  // expects header row with "time,flux" (flux_err optional)
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) return [];
  const headers = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const ti = headers.indexOf("time");
  const fi = headers.indexOf("flux");
  const ei = headers.indexOf("flux_err");
  if (ti === -1 || fi === -1) return [];

  const pts: LCPoint[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    const t = Number(cols[ti]);
    const f = Number(cols[fi]);
    const e = ei >= 0 ? Number(cols[ei]) : undefined;
    if (Number.isFinite(t) && Number.isFinite(f)) pts.push({ t, f, e });
  }
  return pts.sort((a, b) => a.t - b.t);
}

function median(arr: number[]): number {
  if (!arr.length) return NaN;
  const a = arr.slice().sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

function rollingMedian(y: number[], win: number): number[] {
  const n = y.length;
  const w = Math.max(3, win | 0);
  const half = Math.floor(w / 2);
  const out = new Array(n).fill(0);
  // simple O(n*w) version (fine for a few 10k points)
  for (let i = 0; i < n; i++) {
    const s = Math.max(0, i - half);
    const e = Math.min(n, i + half + 1);
    out[i] = median(y.slice(s, e));
  }
  return out;
}

function detrendHighpass(f: number[], kernel: number): number[] {
  const trend = rollingMedian(f, kernel);
  const out = f.map((v, i) => v - trend[i] + 1); // keep mean ~1
  return out;
}

function robustStd(y: number[]): number {
  // 1.4826 * MAD as a robust sigma
  const med = median(y);
  const absDev = y.map((v) => Math.abs(v - med));
  return 1.4826 * median(absDev);
}

function linspace(a: number, b: number, n: number): number[] {
  if (n <= 1) return [a];
  const step = (b - a) / (n - 1);
  return Array.from({ length: n }, (_, i) => a + i * step);
}

function phaseFold(t: number[], P: number, t0: number): number[] {
  // phase in [0,1)
  return t.map((ti) => {
    const ph = (ti - t0) / P;
    return ph - Math.floor(ph);
  });
}

function binPhase(phase: number[], flux: number[], nbins = 50) {
  const bins = new Array(nbins).fill(0).map(() => ({ s: 0, n: 0 }));
  for (let i = 0; i < phase.length; i++) {
    const p = Math.min(nbins - 1, Math.max(0, Math.floor(phase[i] * nbins)));
    const v = flux[i];
    if (Number.isFinite(v)) {
      bins[p].s += v;
      bins[p].n += 1;
    }
  }
  const x: number[] = [];
  const y: number[] = [];
  for (let b = 0; b < nbins; b++) {
    x.push((b + 0.5) / nbins);
    y.push(bins[b].n ? bins[b].s / bins[b].n : NaN);
  }
  return { x, y };
}

type SearchResult = {
  period: number;
  depth_ppm: number;
  duration_hr: number;
  snr: number;
  phaseSeries: { x: number[]; y: number[] };
  t0: number;
  oddEvenDeltaPpm: number;
  secondaryDepthPpm: number;
};

function coarsePeriodSearch(
  t: number[],
  f: number[],
  t0: number
): SearchResult | null {
  if (t.length < 20) return null;
  const span = t[t.length - 1] - t[0];
  const Pmin = Math.max(0.3, span / 200); // sensible lower bound
  const Pmax = Math.min(30, span / 2); // don’t exceed half the timespan
  const trials = linspace(Math.log(Pmin), Math.log(Pmax), 220).map(Math.exp); // log-space

  const all = f.filter((v) => Number.isFinite(v));
  const med = median(all);
  const sigma = robustStd(all) || 1e-4;

  let best = {
    score: -Infinity,
    P: trials[0],
    depthPpm: 0,
    durHr: 0,
    phaseXY: { x: [], y: [] as number[] },
  };

  for (const P of trials) {
    const ph = phaseFold(t, P, t0);
    const nb = 60;
    const { x, y } = binPhase(ph, f, nb);
    // find min bin
    let minV = Infinity,
      minI = 0;
    for (let i = 0; i < nb; i++) {
      const v = y[i];
      if (Number.isFinite(v) && v < minV) {
        minV = v;
        minI = i;
      }
    }
    if (!Number.isFinite(minV)) continue;

    const depth = Math.max(0, med - minV);
    const depthPpm = (depth / med) * 1e6;

    // estimate duration = contiguous bins within threshold near min
    const thr = minV + 0.25 * (med - minV); // 25% up from bottom
    let widthBins = 1;
    // expand left
    let j = (minI - 1 + nb) % nb;
    while (Number.isFinite(y[j]) && y[j] <= thr && j !== minI) {
      widthBins++;
      j = (j - 1 + nb) % nb;
    }
    // expand right
    j = (minI + 1) % nb;
    while (Number.isFinite(y[j]) && y[j] <= thr && j !== minI) {
      widthBins++;
      j = (j + 1) % nb;
    }
    const dur = (widthBins / nb) * P; // days
    const durHr = dur * 24;

    const snr = depth / sigma; // crude SNR
    const score = depthPpm * Math.sqrt(Math.max(1, snr)); // heuristic

    if (score > best.score) {
      best = { score, P, depthPpm, durHr, phaseXY: { x, y } };
    }
  }

  // odd-even & secondary checks at best P
  const bestP = best.P;
  const ph = phaseFold(t, bestP, t0);
  const { x, y } = binPhase(ph, f, 80);
  const yMed = median(y.filter(Number.isFinite));
  // locate primary min
  let minI = 0,
    minV = Infinity;
  for (let i = 0; i < y.length; i++) {
    const v = y[i];
    if (Number.isFinite(v) && v < minV) {
      minV = v;
      minI = i;
    }
  }

  // Odd-even depths: estimate by stacking individual transits
  // Windows around each transit center (kP near the minimum phase)
  const pCenter = x[minI];
  const centers: number[] = [];
  // pick k so that center ~ phase pCenter
  const kStart = Math.floor((t[0] - t0) / bestP) - 1;
  const kEnd = Math.ceil((t[t.length - 1] - t0) / bestP) + 1;
  for (let k = kStart; k <= kEnd; k++) {
    const c = t0 + k * bestP;
    // only centers inside span +/- half P
    if (c >= t[0] - bestP && c <= t[t.length - 1] + bestP) centers.push(c);
  }
  const window = Math.max(0.06 * bestP, (best.durHr / 24) * 1.5); // generous
  const depths: number[] = [];
  for (const c of centers) {
    const seg: number[] = [];
    for (let i = 0; i < t.length; i++) {
      const dt = Math.abs(t[i] - c);
      if (dt <= window) seg.push(f[i]);
    }
    if (seg.length > 4) depths.push(median(seg));
  }
  const odd = depths.filter((_, i) => i % 2 === 0);
  const even = depths.filter((_, i) => i % 2 === 1);
  const oddMed = odd.length ? median(odd) : NaN;
  const evenMed = even.length ? median(even) : NaN;
  const oddEvenDeltaPpm =
    Number.isFinite(oddMed) && Number.isFinite(evenMed)
      ? (Math.abs(oddMed - evenMed) / yMed) * 1e6
      : NaN;

  // Secondary: bin near phase 0.5 from primary
  const secBin = (minI + Math.floor(y.length / 2)) % y.length;
  const secDepthPpm = Number.isFinite(y[secBin])
    ? Math.max(0, ((yMed - y[secBin]) / yMed) * 1e6)
    : NaN;

  return {
    period: bestP,
    depth_ppm: best.depthPpm,
    duration_hr: best.durHr,
    snr: Math.max(
      0,
      Math.round(
        (best.depthPpm / 1e6) * (yMed / (robustStd(all) || 1e-4)) * 100
      ) / 100
    ), // display-ish
    phaseSeries: best.phaseXY,
    t0,
    oddEvenDeltaPpm,
    secondaryDepthPpm: secDepthPpm,
  };
}

function useLinePath(
  xs: number[],
  ys: number[],
  w: number,
  h: number,
  pad = 8
) {
  if (!xs.length || xs.length !== ys.length) return "";
  const xmin = Math.min(...xs),
    xmax = Math.max(...xs);
  const ymin = Math.min(...ys.filter(Number.isFinite)),
    ymax = Math.max(...ys.filter(Number.isFinite));
  const X = (x: number) =>
    pad + ((x - xmin) / (xmax - xmin || 1)) * (w - 2 * pad);
  const Y = (y: number) =>
    h - pad - ((y - ymin) / (ymax - ymin || 1)) * (h - 2 * pad);

  let d = "";
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (!Number.isFinite(y)) continue;
    const cmd = d ? "L" : "M";
    d += `${cmd}${X(xs[i]).toFixed(2)},${Y(y).toFixed(2)} `;
  }
  return d;
}

export function LightCurveAnalyzer() {
  const [raw, setRaw] = useState<LCPoint[]>([]);
  const [detrended, setDetrended] = useState<LCPoint[]>([]);
  const [res, setRes] = useState<SearchResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [kernel, setKernel] = useState<number>(101); // rolling-median window (points)
  const [starR, setStarR] = useState<number | "">(""); // R_sun
  const [starM, setStarM] = useState<number | "">(""); // M_sun

  const handleFile = async (file: File) => {
    setErr(null);
    const text = await file.text();
    const pts = parseCSV(text);
    if (!pts.length) {
      setErr('CSV must have header "time,flux[,flux_err]" and numeric rows.');
      return;
    }
    // normalize flux around ~1 (if not already)
    const fMed = median(pts.map((p) => p.f));
    const norm = pts.map((p) => ({ ...p, f: p.f / (fMed || 1) }));

    // detrend high-pass with rolling median
    const f = norm.map((p) => p.f);
    const hp = detrendHighpass(f, Math.max(11, kernel | 0));
    const det = norm.map((p, i) => ({ ...p, f: hp[i] }));
    setRaw(norm);
    setDetrended(det);
    setRes(null);
  };

  const runSearch = () => {
    if (!detrended.length) return;
    const t = detrended.map((p) => p.t);
    const f = detrended.map((p) => p.f);
    const t0 = t[0];
    const out = coarsePeriodSearch(t, f, t0);
    if (!out) {
      setErr("Not enough points or span to search.");
      return;
    }
    setRes(out);
  };

  // plots
  const timePath = useMemo(() => {
    if (!raw.length) return "";
    const xs = raw.map((p) => p.t);
    const ys = raw.map((p) => p.f);
    return useLinePath(xs, ys, 640, 180);
  }, [raw]);

  const detPath = useMemo(() => {
    if (!detrended.length) return "";
    const xs = detrended.map((p) => p.t);
    const ys = detrended.map((p) => p.f);
    return useLinePath(xs, ys, 640, 180);
  }, [detrended]);

  const foldPath = useMemo(() => {
    if (!res || !detrended.length) return "";
    const { x, y } = res.phaseSeries;
    // duplicate phase (0..2) for nicer wrap
    const xs = x.concat(x.map((v) => v + 1));
    const ys = y.concat(y);
    return useLinePath(xs, ys, 640, 180);
  }, [res]);

  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.02] backdrop-blur p-6 space-y-6">
      <div className="flex items-center justify-between">
        {/* header toolbar */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="text-sm font-medium text-white/70">
            Light curve analyzer
          </div>

          <div className="flex items-center gap-6 text-sm">
            {/* existing detrend control */}
            <div className="flex items-center gap-3">
              <label className="text-white/70">detrend window</label>
              <input
                type="number"
                min={11}
                step={2}
                value={kernel}
                onChange={(e) => setKernel(Number(e.target.value))}
                className="w-24 rounded-lg bg-black/40 border border-white/10 px-2 py-1"
                title="rolling median kernel (points)"
              />
            </div>

            {/* NEW: stellar params inputs */}
            <div className="flex items-center gap-3">
              <label className="text-white/70">R★ (R☉)</label>
              <input
                value={starR}
                onChange={(e) => setStarR(Number(e.target.value) || "")}
                className="w-20 rounded-lg bg-black/40 border border-white/10 px-2 py-1"
              />
              <label className="text-white/70">M★ (M☉)</label>
              <input
                value={starM}
                onChange={(e) => setStarM(Number(e.target.value) || "")}
                className="w-20 rounded-lg bg-black/40 border border-white/10 px-2 py-1"
              />
            </div>
          </div>
        </div>
      </div>

      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files?.[0];
          if (f) handleFile(f);
        }}
        className="border border-dashed border-white/20 rounded-2xl p-8 text-center hover:border-white/40 transition"
      >
        <p className="text-white/70">
          Drag & drop a CSV with columns:{" "}
          <code className="text-white">time,flux[,flux_err]</code>
        </p>
        <div className="mt-4">
          <label className="inline-block px-4 py-2 rounded-xl bg-white text-black font-medium cursor-pointer hover:opacity-90">
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />
            Choose file
          </label>
        </div>
        <p className="mt-3 text-xs text-white/50">
          Tip: start with Kepler/TESS light curves exported as time–flux.
        </p>
        {err && <p className="mt-3 text-red-300 text-sm">{err}</p>}
      </div>

      {/* raw plot */}
      {raw.length > 0 && (
        <div>
          <div className="text-xs text-white/60 uppercase tracking-wide mb-2">
            raw flux
          </div>
          <div className="relative h-[200px] w-full overflow-hidden rounded-xl border border-white/10 bg-black/40">
            <svg viewBox="0 0 640 200" className="w-full h-full">
              <path
                d={timePath}
                stroke="white"
                strokeOpacity="0.6"
                strokeWidth="1.2"
                fill="none"
              />
            </svg>
          </div>
        </div>
      )}

      {/* detrended plot */}
      {detrended.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-xs text-white/60 uppercase tracking-wide">
              detrended flux
            </div>
            <button
              onClick={runSearch}
              className="px-4 py-2 rounded-xl bg-white text-black font-medium hover:opacity-90"
            >
              Find period
            </button>
          </div>
          <div className="relative h-[200px] w-full overflow-hidden rounded-xl border border-white/10 bg-black/40">
            <svg viewBox="0 0 640 200" className="w-full h-full">
              <path
                d={detPath}
                stroke="white"
                strokeOpacity="0.9"
                strokeWidth="1.2"
                fill="none"
              />
            </svg>
          </div>
        </div>
      )}

      {/* results */}
      {res && (
        <>
          {/* existing plots... */}
          <div className="grid md:grid-cols-3 gap-3 mt-4">
            {/* Existing KPIs ... */}

            {Number(starR) && (
              <KPI
                label="Estimated Rₚ"
                value={(() => {
                  const depthFrac = (res.depth_ppm || 0) / 1e6;
                  const RpRs = Math.sqrt(Math.max(0, depthFrac));
                  const RpRsun = (Number(starR) as number) * RpRs;
                  const earthR = RpRsun * 109.2; // 1 R_sun ≈ 109.2 R_earth
                  return `${earthR.toFixed(2)} R⊕`;
                })()}
              />
            )}

            {Number(starM) && (
              <KPI
                label="Semi-major axis"
                value={(() => {
                  const P_yr = res.period / 365.25;
                  const a_AU =
                    Math.cbrt(Number(starM) as number) * Math.pow(P_yr, 2 / 3);
                  return `${a_AU.toFixed(3)} AU`;
                })()}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}

function KPI({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
      <div className="text-xs text-white/60 uppercase tracking-wide">
        {label}
      </div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
  );
}
