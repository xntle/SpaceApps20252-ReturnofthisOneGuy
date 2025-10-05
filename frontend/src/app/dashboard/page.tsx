"use client";

import React, { useMemo, useRef, useState } from "react";
import { Header } from "@/components/Header";
import { ExoDotsCanvas } from "@/components/ExoDotsCanvas";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
} from "recharts";
import Link from "next/link";

type Row = { time: number; flux: number; flux_err?: number };
type Features = {
  // stats
  flux_mean: number;
  flux_std: number;
  flux_median: number;
  flux_mad: number;
  flux_skew: number;
  flux_kurtosis: number;
  flux_range: number;
  flux_iqr: number;
  flux_rms: number;
  // period-ish
  best_period: number;
  bls_power: number;
  period_error: number;
  // data quality
  n_points: number;
  time_span: number;
  cadence_median: number;
  cadence_std: number;
  // transit
  transit_depth: number;
  transit_significance: number;
};
type Output = { predicted_label: 0 | 1; predicted_proba: number };

const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const std = (a: number[]) => {
  const m = mean(a);
  return Math.sqrt(mean(a.map((v) => (v - m) ** 2)));
};
const mad = (a: number[]) => median(a.map((v) => Math.abs(v - median(a))));
const iqr = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  const q1 = s[Math.floor(0.25 * (s.length - 1))];
  const q3 = s[Math.floor(0.75 * (s.length - 1))];
  return q3 - q1;
};
const skew = (a: number[]) => {
  const m = mean(a),
    sd = std(a) || 1;
  return mean(a.map((v) => ((v - m) / sd) ** 3));
};
const kurtosis = (a: number[]) => {
  const m = mean(a),
    sd = std(a) || 1;
  return mean(a.map((v) => ((v - m) / sd) ** 4));
};
const clamp = (x: number, min: number, max: number) =>
  Math.max(min, Math.min(max, x));

function parseCSV(text: string): Row[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const tIdx = headers.indexOf("time");
  const fIdx = headers.indexOf("flux");
  const eIdx = headers.indexOf("flux_err");
  if (tIdx < 0 || fIdx < 0)
    throw new Error("CSV must include headers: time, flux[, flux_err]");
  const rows: Row[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    const time = parseFloat(cols[tIdx]);
    const flux = parseFloat(cols[fIdx]);
    const flux_err = eIdx >= 0 ? parseFloat(cols[eIdx]) : undefined;
    if (!Number.isFinite(time) || !Number.isFinite(flux)) continue;
    rows.push({ time, flux, flux_err });
  }
  return rows;
}

/* ============================
   Detrend (simple rolling median)
============================= */
function rollingMedianDetrend(rows: Row[], win = 51): Row[] {
  if (rows.length === 0) return [];
  const y = rows.map((r) => r.flux);
  const half = Math.max(1, Math.floor(win / 2));
  const trend: number[] = y.map((_, i) => {
    const s = Math.max(0, i - half),
      e = Math.min(y.length - 1, i + half);
    return median(y.slice(s, e + 1));
  });
  return rows.map((r, i) => ({
    ...r,
    flux_detrended: r.flux / (trend[i] || 1),
  }));
}

function estimatePeriodViaACF(rows: Row[]): {
  best_period: number;
  power: number;
  width: number;
} {
  if (rows.length < 50) return { best_period: NaN, power: 0, width: 0 };
  const times = rows.map((r) => r.time);
  const y = rows.map((r) => r.flux);
  const dt = median(times.slice(1).map((t, i) => t - times[i]));
  const N = Math.min(5000, y.length);
  const y0 = y.slice(0, N);
  const m = mean(y0);
  const sd = std(y0) || 1;
  const norm = y0.map((v) => (v - m) / sd);
  const maxLag = Math.min(Math.floor(N / 3), 2000);
  let bestLag = 0,
    best = -Infinity,
    bestW = 0;
  for (let lag = 5; lag < maxLag; lag++) {
    let s = 0;
    let c = 0;
    for (let i = lag; i < norm.length; i++) {
      s += norm[i] * norm[i - lag];
      c++;
    }
    const corr = s / (c || 1);
    if (corr > best) {
      best = corr;
      bestLag = lag;
      bestW = 5;
    }
  }
  const period = bestLag * dt;
  const power = clamp((best + 1) / 2, 0, 1); // normalize -inf..inf -> 0..1-ish
  const width = bestW * dt * 0.1;
  return { best_period: period, power, width };
}

/* ============================
   Features (18)
============================= */
function computeFeatures(rows: Row[]): Features {
  const t = rows.map((r) => r.time);
  const y = rows.map((r) => r.flux);
  const yMed = median(y);
  const yCentered = y.map((v) => v - yMed);

  // Stats
  const flux_mean = mean(y);
  const flux_std = std(y);
  const flux_median = yMed;
  const flux_mad = mad(y);
  const flux_skew = skew(y);
  const flux_kurtosis = kurtosis(y);
  const flux_range = Math.max(...y) - Math.min(...y);
  const flux_iqr = iqr(y);
  const flux_rms = Math.sqrt(mean(yCentered.map((v) => v * v)));

  // Data quality
  const n_points = rows.length;
  const time_span = Math.max(...t) - Math.min(...t);
  const cadences = t.slice(1).map((ti, i) => ti - t[i]);
  const cadence_median = cadences.length ? median(cadences) : 0;
  const cadence_std = cadences.length ? std(cadences) : 0;

  // Transit-ish
  const depth = yMed - median(y.filter((v) => v < yMed)); // crude
  const transit_depth = depth;
  const transit_significance = flux_mad > 0 ? depth / flux_mad : 0;

  // Period proxy
  const { best_period, power: bls_power, width } = estimatePeriodViaACF(rows);
  const period_error = width || cadence_median * 2;

  return {
    flux_mean,
    flux_std,
    flux_median,
    flux_mad,
    flux_skew,
    flux_kurtosis,
    flux_range,
    flux_iqr,
    flux_rms,
    best_period,
    bls_power,
    period_error,
    n_points,
    time_span,
    cadence_median,
    cadence_std,
    transit_depth,
    transit_significance,
  };
}

/* ============================
   Heuristic fallback "model"
============================= */
function fallbackPredict(f: Features): Output {
  const z1 = clamp(f.transit_significance / 10, 0, 2); // ~0..2
  const z2 = clamp(f.bls_power, 0, 1); // 0..1
  const z = 0.9 * z1 + 0.8 * z2 - 0.6;
  const proba = 1 / (1 + Math.exp(-3 * z)); // sigmoid
  return { predicted_label: proba >= 0.7 ? 1 : 0, predicted_proba: proba };
}

/* ============================
   Radial confidence
============================= */
function Radial({ p }: { p: number }) {
  const pct = clamp(p, 0, 1);
  const R = 38,
    C = 2 * Math.PI * R,
    off = C * (1 - pct);
  const band = pct < 0.3 ? "red" : pct < 0.7 ? "yellow" : "emerald";
  return (
    <div className="relative w-24 h-24">
      <svg viewBox="0 0 100 100" className="w-24 h-24">
        <circle
          cx="50"
          cy="50"
          r={R}
          className="fill-none stroke-[10] stroke-white/10"
        />
        <circle
          cx="50"
          cy="50"
          r={R}
          className={`fill-none stroke-[10] -rotate-90 origin-center stroke-${band}-400`}
          strokeDasharray={C}
          strokeDashoffset={off}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center">
        <div className="text-center">
          <div className="text-lg font-semibold">{Math.round(pct * 100)}%</div>
          <div className="text-[10px] uppercase tracking-wide text-white/60">
            confidence
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================
   Main page
============================= */
export default function DashboardPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [detrendOn, setDetrendOn] = useState(true);
  const [sigma, setSigma] = useState(4);
  const [features, setFeatures] = useState<Features | null>(null);
  const [output, setOutput] = useState<Output | null>(null);
  const [error, setError] = useState<string | null>(null);

  const det = useMemo(
    () => (detrendOn ? rollingMedianDetrend(rows) : []),
    [rows, detrendOn]
  );

  const onFile = async (file: File) => {
    setError(null);
    try {
      const text = await file.text();
      const parsed = parseCSV(text);
      if (!parsed.length) throw new Error("No valid rows parsed.");
      // (optional) sigma-clip
      const med = median(parsed.map((r) => r.flux));
      const m = mad(parsed.map((r) => r.flux)) || 1e-9;
      const lo = med - sigma * m * 1.4826,
        hi = med + sigma * m * 1.4826;
      const clipped = parsed.filter((r) => r.flux >= lo && r.flux <= hi);
      setRows(clipped);
      setFeatures(null);
      setOutput(null);
    } catch (e: any) {
      setError(e?.message || "Failed to parse CSV.");
    }
  };

  // post route after fastapi(todo)
  const analyze = async () => {
    if (!rows.length) return;
    const f = computeFeatures(rows);
    setFeatures(f);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: f }),
      });
      if (!res.ok) throw new Error("stub");
      const data = (await res.json()) as Output;
      setOutput(data);
    } catch {
      setOutput(fallbackPredict(f));
    }
  };

  const verdict = output
    ? output.predicted_proba < 0.3
      ? "Very likely non-planet"
      : output.predicted_proba < 0.7
      ? "Uncertain (needs follow-up)"
      : "Very likely planet"
    : "—";

  return (
    <main className="relative min-h-screen bg-black text-white overflow-hidden">
      <Header />
      <div className="fixed inset-0 z-0 opacity-35 pointer-events-none">
        <ExoDotsCanvas />
      </div>

      <section className="relative z-10 px-6 pt-20 pb-28 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8 flex items-center justify-between gap-4"
        >
          <h1 className="text-2xl md:text-4xl font-semibold tracking-tight">
            lol
          </h1>
          <Link href="/" className="text-white/60 hover:text-white text-sm">
            ← back
          </Link>
        </motion.div>

        {/* Step 1: Upload */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05, duration: 0.6 }}
          className="grid md:grid-cols-3 gap-6"
        >
          <div className="md:col-span-2 rounded-3xl border border-white/10 bg-white/[0.02] backdrop-blur p-6">
            <div className="mb-3 text-sm font-medium text-white/70">
              1) Upload light curve (CSV)
            </div>

            <div
              onDragOver={(e) => {
                e.preventDefault();
              }}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files?.[0];
                if (f) onFile(f);
              }}
              className="border border-dashed border-white/20 rounded-2xl p-8 text-center hover:border-white/40 transition"
            >
              <p className="text-white/70">
                Drag & drop a CSV with{" "}
                <code className="text-white">time, flux[, flux_err]</code>
              </p>
              <div className="mt-4">
                <label className="inline-block px-4 py-2 rounded-xl bg-white text-black font-medium cursor-pointer hover:opacity-90">
                  <input
                    type="file"
                    accept=".csv"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) onFile(f);
                    }}
                  />
                  Choose file
                </label>
              </div>
              {error && <p className="mt-3 text-red-300 text-sm">{error}</p>}
            </div>

            {rows.length > 0 && (
              <div className="mt-5 text-xs text-white/70">
                Loaded <b>{rows.length}</b> points • First 5 rows:
                <div className="mt-2 overflow-x-auto rounded-xl border border-white/10">
                  <table className="w-full text-left whitespace-nowrap">
                    <thead className="bg-white/5">
                      <tr>
                        <th className="px-3 py-2">time</th>
                        <th className="px-3 py-2">flux</th>
                        <th className="px-3 py-2">flux_err</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.slice(0, 5).map((r, i) => (
                        <tr key={i} className="border-t border-white/5">
                          <td className="px-3 py-2">{r.time}</td>
                          <td className="px-3 py-2">{r.flux}</td>
                          <td className="px-3 py-2">{r.flux_err ?? ""}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* KPIs */}
        {rows.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.4 }}
            transition={{ duration: 0.6 }}
            className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4"
          >
            <KPI label="Points" value={rows.length.toLocaleString()} />
            <KPI
              label="Span (d)"
              value={fmtNum(
                Math.max(...rows.map((r) => r.time)) -
                  Math.min(...rows.map((r) => r.time))
              )}
            />
            <KPI
              label="Median cadence (d)"
              value={fmtNum(
                median(rows.slice(1).map((r, i) => r.time - rows[i].time)) || 0
              )}
            />
            <KPI
              label="RMS"
              value={fmtNum(
                Math.sqrt(
                  mean(
                    rows
                      .map((r) => r.flux)
                      .map((v) => (v - median(rows.map((r) => r.flux))) ** 2)
                  )
                )
              )}
            />
          </motion.div>
        )}

        {/* Charts */}
        {rows.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6 }}
            className="mt-8 grid lg:grid-cols-2 gap-6"
          >
            <Card title="Raw light curve">
              <ChartLC
                data={rows.map((r) => ({ time: r.time, flux: r.flux }))}
              />
            </Card>
            <Card
              title={detrendOn ? "Detrended (flux/trend)" : "Detrended (off)"}
            >
              <ChartDetrended
                data={(det.length ? det : rows).map((r: any) => ({
                  time: r.time,
                  flux: r.flux_detrended ?? r.flux,
                }))}
              />
            </Card>
          </motion.div>
        )}

        {/* Output & Features */}
        {(features || output) && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6 }}
            className="mt-8 grid md:grid-cols-5 gap-6"
          >
            <Card className="md:col-span-2" title="Model output">
              {output ? (
                <div className="flex items-center gap-5">
                  <Radial p={output.predicted_proba} />
                  <div>
                    <div className="text-sm uppercase tracking-wide text-white/60">
                      classification
                    </div>
                    <div className="mt-1 text-2xl font-semibold">
                      {output.predicted_label === 1 ? "Planet" : "Non-planet"}
                    </div>
                    <div className="mt-1 text-sm text-white/70">{verdict}</div>
                  </div>
                </div>
              ) : (
                <div className="text-white/60 text-sm">
                  Run “Analyze” to see predictions.
                </div>
              )}
            </Card>

            <Card className="md:col-span-3" title="Features (18)">
              {features ? (
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
                  {Object.entries(features).map(([k, v]) => (
                    <div
                      key={k}
                      className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2"
                    >
                      <div className="text-white/50 text-[11px] uppercase tracking-wide">
                        {k}
                      </div>
                      <div className="mt-1 font-semibold">{fmtNum(v)}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-white/60 text-sm">
                  Features will appear after analysis.
                </div>
              )}
            </Card>
          </motion.div>
        )}
      </section>
    </main>
  );
}

/* ========== Small subcomponents ========== */
function KPI({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-4">
      <div className="text-xs text-white/60 uppercase tracking-wide">
        {label}
      </div>
      <div className="mt-1 text-3xl font-semibold">{value}</div>
    </div>
  );
}

function Card({
  title,
  className,
  children,
}: {
  title: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`rounded-3xl border border-white/10 bg-white/[0.02] backdrop-blur p-5 md:p-7 ${
        className || ""
      }`}
    >
      <div className="mb-3 text-sm font-medium text-white/70">{title}</div>
      {children}
    </div>
  );
}

function ChartLC({ data }: { data: { time: number; flux: number }[] }) {
  return (
    <div className="h-64 md:h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" />
          <XAxis dataKey="time" stroke="#aaa" />
          <YAxis stroke="#aaa" />
          <Tooltip
            contentStyle={{
              background: "rgba(0,0,0,0.8)",
              border: "1px solid rgba(255,255,255,0.1)",
            }}
          />
          <Line
            type="monotone"
            dataKey="flux"
            dot={false}
            stroke="#fff"
            strokeWidth={1.2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
function ChartDetrended({ data }: { data: { time: number; flux: number }[] }) {
  return (
    <div className="h-64 md:h-72">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" />
          <XAxis dataKey="time" stroke="#aaa" />
          <YAxis stroke="#aaa" />
          <Tooltip
            contentStyle={{
              background: "rgba(0,0,0,0.8)",
              border: "1px solid rgba(255,255,255,0.1)",
            }}
          />
          <Area dataKey="flux" stroke="#fff" fill="rgba(255,255,255,0.12)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function fmtNum(n: number) {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  return Number.isInteger(n) ? n.toString() : n.toFixed(4);
}
