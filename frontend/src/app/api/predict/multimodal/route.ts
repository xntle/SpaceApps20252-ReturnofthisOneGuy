import { NextResponse } from "next/server";

export const runtime = "nodejs"; // ensure we're on node runtime

// Safe number parser
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
  const predicted_label = proba >= 0.7 ? 1 : 0;

  // ---- NEW: pseudo feature "z-scores" for UI visuals ----
  // Use the same weights as the fallback model to build contributions,
  // then scale to ±3 so the radar/feature bars look nice and stable.
  const raw = [
    { name: "koi_depth", v: 0.002 * depth },
    { name: "koi_duration", v: 0.1 * dur },
    { name: "koi_period", v: 0.05 * period },
    { name: "koi_impact", v: -0.2 * imp },
  ];
  const maxAbs = Math.max(1e-9, ...raw.map((r) => Math.abs(r.v)));
  const top_features = raw
    .map((r) => ({ name: r.name, z: Number(((r.v / maxAbs) * 3).toFixed(3)) })) // scale to ±3
    .sort((a, b) => Math.abs(b.z) - Math.abs(a.z));

  // Also provide a debug_features map (what FeatureBars prefers when available)
  const debug_features = Object.fromEntries(
    top_features.map((f) => [f.name, f.z])
  );

  return {
    predicted_label,
    predicted_proba: proba,
    top_features,
    debug_features,
  };
}

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const rowRaw = form.get("row");

    if (typeof rowRaw !== "string") {
      return NextResponse.json({ error: "Missing row JSON" }, { status: 400 });
    }

    const row = JSON.parse(rowRaw) as Record<string, unknown>;

    const resEntry = form.get("residual_windows");
    const pixEntry = form.get("pixel_diffs");
    const residual = resEntry instanceof File ? resEntry : null;
    const pixels = pixEntry instanceof File ? pixEntry : null;

    // TODO: forward residual/pixels to Python CNN when ready.
    // const resBytes = residual ? await residual.arrayBuffer() : null;
    // const pixBytes = pixels ? await pixels.arrayBuffer() : null;

    // For now, run tabular fallback so UI flows and visuals render.
    const out = scoreTabularFallback(row);

    const clamp = (p: number) => Math.min(0.999999999, Math.max(1e-9, p));
    const p = clamp(out.predicted_proba);
    const logit = Math.log(p / (1 - p));
    const entropy = -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
    const thresholds = { green: 0.7, yellow: 0.3 };
    const decision_band =
      p >= thresholds.green
        ? "green"
        : p >= thresholds.yellow
        ? "yellow"
        : "red";
    const used_cnn = Boolean(residual && pixels);

    return NextResponse.json({
      predicted_label: out.predicted_label,
      predicted_proba: p,
      debug_features: out.debug_features,
      extras: {
        logit,
        entropy,
        thresholds,
        decision_band,
        used_cnn,
        top_features: out.top_features,
      },
    });
  } catch (e: any) {
    return NextResponse.json(
      { error: e?.message ?? "Server error" },
      { status: 500 }
    );
  }
}
