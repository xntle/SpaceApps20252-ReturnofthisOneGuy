// app/api/predict/tabular/route.ts
import { NextResponse } from "next/server";

export const runtime = "nodejs";

// safe number
const f = (v: unknown): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
};

function to39Features(row: Record<string, string>) {
  // --- transit (15)
  const period = f(row.koi_period);
  const period_e1 = f(row.koi_period_err1);
  const period_e2 = f(row.koi_period_err2);
  const epoch = f(row.koi_time0bk);
  const epoch_e1 = f(row.koi_time0bk_err1);
  const epoch_e2 = f(row.koi_time0bk_err2);
  const depth_ppm = f(row.koi_depth);
  const depth_e1 = f(row.koi_depth_err1);
  const depth_e2 = f(row.koi_depth_err2);
  const dur_hr = f(row.koi_duration);
  const dur_e1 = f(row.koi_duration_err1);
  const dur_e2 = f(row.koi_duration_err2);
  const impact = f(row.koi_impact);
  const impact_e1 = f(row.koi_impact_err1);
  const impact_e2 = f(row.koi_impact_err2);

  // --- stellar (15)
  const teff = f(row.koi_steff);
  const teff_e1 = f(row.koi_steff_err1);
  const teff_e2 = f(row.koi_steff_err2);

  const srad = f(row.koi_srad);
  const srad_e1 = f(row.koi_srad_err1);
  const srad_e2 = f(row.koi_srad_err2);

  const smass = f(row.koi_smass);
  const smass_e1 = f(row.koi_smass_err1);
  const smass_e2 = f(row.koi_smass_err2);

  const slogg = f(row.koi_slogg);
  const slogg_e1 = f(row.koi_slogg_err1);
  const slogg_e2 = f(row.koi_slogg_err2);

  const smet = f(row.koi_smet);
  const smet_e1 = f(row.koi_smet_err1);
  const smet_e2 = f(row.koi_smet_err2);

  // --- engineered (9)
  const quarters = (row.koi_quarters ?? "").trim();
  const n_quarters = quarters ? quarters.match(/1/g)?.length ?? 0 : 0;

  const duty_cycle =
    Number.isFinite(period) && period > 0 && Number.isFinite(dur_hr)
      ? dur_hr / (24 * period)
      : 0;

  const log_period =
    Number.isFinite(period) && period > 0 ? Math.log10(period) : 0;
  const log_duration =
    Number.isFinite(dur_hr) && dur_hr > 0 ? Math.log10(dur_hr) : 0;

  const period_err_rel =
    Number.isFinite(period) &&
    period !== 0 &&
    Number.isFinite(period_e1) &&
    Number.isFinite(period_e2)
      ? (Math.abs(period_e1) + Math.abs(period_e2)) / (2 * period)
      : 0;

  const duration_err_rel =
    Number.isFinite(dur_hr) &&
    dur_hr !== 0 &&
    Number.isFinite(dur_e1) &&
    Number.isFinite(dur_e2)
      ? (Math.abs(dur_e1) + Math.abs(dur_e2)) / (2 * dur_hr)
      : 0;

  const err_asym_period =
    (Number.isFinite(period_e1) ? Math.abs(period_e1) : 0) -
    (Number.isFinite(period_e2) ? Math.abs(period_e2) : 0);

  const err_asym_duration =
    (Number.isFinite(dur_e1) ? Math.abs(dur_e1) : 0) -
    (Number.isFinite(dur_e2) ? Math.abs(dur_e2) : 0);

  const epoch_err_span =
    (Number.isFinite(epoch_e1) ? Math.abs(epoch_e1) : 0) +
    (Number.isFinite(epoch_e2) ? Math.abs(epoch_e2) : 0);

  // Keep a single, explicit order (39 total)
  const order = [
    // 15 transit
    "period",
    "period_e1",
    "period_e2",
    "epoch",
    "epoch_e1",
    "epoch_e2",
    "depth_ppm",
    "depth_e1",
    "depth_e2",
    "dur_hr",
    "dur_e1",
    "dur_e2",
    "impact",
    "impact_e1",
    "impact_e2",
    // 15 stellar
    "teff",
    "teff_e1",
    "teff_e2",
    "srad",
    "srad_e1",
    "srad_e2",
    "smass",
    "smass_e1",
    "smass_e2",
    "slogg",
    "slogg_e1",
    "slogg_e2",
    "smet",
    "smet_e1",
    "smet_e2",
    // 9 engineered
    "duty_cycle",
    "log_period",
    "log_duration",
    "period_err_rel",
    "duration_err_rel",
    "err_asym_period",
    "err_asym_duration",
    "epoch_err_span",
    "n_quarters",
  ] as const;

  const features: Record<(typeof order)[number], number> = {
    period,
    period_e1,
    period_e2,
    epoch,
    epoch_e1,
    epoch_e2,
    depth_ppm,
    depth_e1,
    depth_e2,
    dur_hr,
    dur_e1,
    dur_e2,
    impact,
    impact_e1,
    impact_e2,
    teff,
    teff_e1,
    teff_e2,
    srad,
    srad_e1,
    srad_e2,
    smass,
    smass_e1,
    smass_e2,
    slogg,
    slogg_e1,
    slogg_e2,
    smet,
    smet_e1,
    smet_e2,
    duty_cycle,
    log_period,
    log_duration,
    period_err_rel,
    duration_err_rel,
    err_asym_period,
    err_asym_duration,
    epoch_err_span,
    n_quarters,
  };

  return { features, order };
}

// Placeholder scaler (replace with your saved scaler params)
const MEAN = new Array(39).fill(0);
const STD = new Array(39).fill(1);

function standardize(vec: number[], mean: number[], std: number[]) {
  return vec.map((v, i) => {
    const m = mean[i] ?? 0;
    const s = std[i] ?? 1;
    if (!Number.isFinite(v)) return 0; // simple impute
    return s ? (v - m) / s : v - m;
  });
}

// Feature indices (avoid off-by-one)
const IDX = {
  depth_ppm: 6,
  dur_hr: 9,
  impact: 12,
  log_period: 31, // <-- correct index
} as const;

// Tiny scoring stub — replace with your real model inference
function score(z: number[]) {
  const depth = z[IDX.depth_ppm] ?? 0; // depth_ppm (standardized)
  const logP = z[IDX.log_period] ?? 0; // log_period (standardized)
  const dur = z[IDX.dur_hr] ?? 0; // dur_hr (standardized)
  const imp = z[IDX.impact] ?? 0; // impact (standardized)
  const lin = 0.8 * depth + 0.2 * logP + 0.15 * dur - 0.3 * Math.abs(imp);
  const proba = 1 / (1 + Math.exp(-lin));
  const label: 0 | 1 = proba >= 0.7 ? 1 : 0;
  return { predicted_label: label, predicted_proba: proba };
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const row = (body?.row ?? null) as Record<string, string> | null;
    if (!row) {
      return NextResponse.json({ error: "Missing row" }, { status: 400 });
    }

    const { features, order } = to39Features(row);
    const vec = order.map((k) => features[k]);
    const z = standardize(vec, MEAN, STD);

    const { predicted_label, predicted_proba } = score(z);

    // helpers
    const clamp = (x: number, a = 1e-9, b = 1 - 1e-9) =>
      Math.min(b, Math.max(a, x));
    const p = clamp(predicted_proba);
    const logit = Math.log(p / (1 - p));
    const entropy = -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
    const thresholds = { green: 0.7, yellow: 0.3 };
    const decision_band =
      p >= thresholds.green
        ? "green"
        : p >= thresholds.yellow
        ? "yellow"
        : "red";

    const debug = Object.fromEntries(order.map((k, i) => [k, z[i]]));
    const top_features = Object.entries(debug)
      .map(([name, v]) => ({ name, z: Number(v), a: Math.abs(Number(v)) }))
      .sort((a, b) => b.a - a.a)
      .slice(0, 8)
      .map(({ name, z }) => ({ name, z }));

    return NextResponse.json({
      predicted_label,
      predicted_proba: p,
      debug_features: debug,
      extras: {
        decision_band,
        thresholds,
        logit,
        entropy,
        used_cnn: false,
        top_features,
      },
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : "Server error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
