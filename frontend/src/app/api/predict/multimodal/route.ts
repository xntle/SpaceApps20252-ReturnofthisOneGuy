import { NextResponse } from "next/server";

export const runtime = "nodejs";

// ---- Types ----------------------------------------------------
type DecisionBand = "green" | "yellow" | "red";
type ZVec = Record<string, number>;
interface TopFeature {
  name: string;
  z: number;
}
interface Extras {
  logit: number;
  entropy: number;
  thresholds: { green: number; yellow: number };
  decision_band: DecisionBand;
  used_cnn: boolean;
  top_features: TopFeature[];
}
interface PredictResponse {
  predicted_label: 0 | 1;
  predicted_proba: number;
  debug_features: ZVec;
  extras: Extras;
}

// ---- Utils ----------------------------------------------------
const num = (x: unknown) => {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
};

function scoreTabularFallback(row: Record<string, unknown>) {
  const depth = num(row.koi_depth);
  const dur = num(row.koi_duration);
  const period = num(row.koi_period);
  const imp = Math.abs(num(row.koi_impact));

  const lin = 0.002 * depth + 0.1 * dur + 0.05 * period - 0.2 * imp;
  const proba = 1 / (1 + Math.exp(-lin));
  const predicted_label: 0 | 1 = proba >= 0.7 ? 1 : 0;

  const raw = [
    { name: "koi_depth", v: 0.002 * depth },
    { name: "koi_duration", v: 0.1 * dur },
    { name: "koi_period", v: 0.05 * period },
    { name: "koi_impact", v: -0.2 * imp },
  ];
  const maxAbs = Math.max(1e-9, ...raw.map((r) => Math.abs(r.v)));
  const top_features: TopFeature[] = raw
    .map((r) => ({ name: r.name, z: Number(((r.v / maxAbs) * 3).toFixed(3)) }))
    .sort((a, b) => Math.abs(b.z) - Math.abs(a.z));

  const debug_features: ZVec = Object.fromEntries(
    top_features.map((f) => [f.name, f.z])
  );

  return {
    predicted_label,
    predicted_proba: proba,
    top_features,
    debug_features,
  };
}

// ---- Route ----------------------------------------------------
export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const rowRaw = form.get("row");
    if (typeof rowRaw !== "string") {
      return NextResponse.json({ error: "Missing row JSON" }, { status: 400 });
    }
    const row = JSON.parse(rowRaw) as Record<string, unknown>;

    // Type-safe file extraction
    const resEntry = form.get("residual_windows");
    const pixEntry = form.get("pixel_diffs");
    const hasFileCtor = typeof File !== "undefined";
    const residual = hasFileCtor && resEntry instanceof File ? resEntry : null;
    const pixels = hasFileCtor && pixEntry instanceof File ? pixEntry : null;

    // Fallback scoring
    const out = scoreTabularFallback(row);

    // Stable numerics for logit/entropy
    const clamp = (p: number) => Math.min(0.999999999, Math.max(1e-9, p));
    const p = clamp(out.predicted_proba);
    const logit = Math.log(p / (1 - p));
    const entropy = -(p * Math.log(p) + (1 - p) * Math.log(1 - p));

    const thresholds = { green: 0.7, yellow: 0.3 } as const;
    const decision_band: DecisionBand =
      p >= thresholds.green
        ? "green"
        : p >= thresholds.yellow
        ? "yellow"
        : "red";

    const used_cnn = Boolean(residual && pixels);

    const payload: PredictResponse = {
      predicted_label: out.predicted_label,
      predicted_proba: p,
      debug_features: out.debug_features,
      extras: {
        logit,
        entropy,
        thresholds: { green: thresholds.green, yellow: thresholds.yellow },
        decision_band,
        used_cnn,
        top_features: out.top_features,
      },
    };

    return NextResponse.json(payload);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : "Server error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
