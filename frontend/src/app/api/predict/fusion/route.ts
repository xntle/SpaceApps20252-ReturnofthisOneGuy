// src/app/api/predict/fusion/route.ts
import { NextResponse } from "next/server";

export const runtime = "nodejs";

// Set where your FastAPI is running
const FUSION_URL =
  process.env.FUSION_URL || "http://127.0.0.1:8000/predict_exoplanet";

// Helper: forward multipart form-data (keeps files streaming)
async function forwardMultipart(request: Request) {
  const inForm = await request.formData();
  const out = new FormData();

  // passthrough known fields (strings)
  for (const key of [
    "kepid",
    "features",
    "residual_window_path",
    "pixel_image_path",
  ]) {
    const v = inForm.get(key);
    if (typeof v === "string") out.append(key, v);
  }

  // files from your UI (rename keys to match FastAPI endpoint above)
  const residual = inForm.get("residual_windows");
  const pixels = inForm.get("pixel_diffs");
  if (residual instanceof File)
    out.append("residual_window", residual, residual.name);
  if (pixels instanceof File) out.append("pixel_image", pixels, pixels.name);

  const res = await fetch(FUSION_URL, { method: "POST", body: out });
  const json = await res.json().catch(() => ({}));
  return NextResponse.json(json, { status: res.status });
}

// Helper: forward JSON by converting to form-data for FastAPI
async function forwardJSON(request: Request) {
  const body = await request
    .json()
    .catch(() => ({} as Record<string, unknown>));
  const fd = new FormData();

  // Expecting: { kepid?: number, features?: object, residual_window_path?: string, pixel_image_path?: string }
  const kepid = body.kepid;
  if (typeof kepid === "number" || typeof kepid === "string")
    fd.append("kepid", String(kepid));

  const features = body.features ?? body.row ?? {}; // allow `{row}` from your tabular flow
  fd.append("features", JSON.stringify(features));

  if (typeof body.residual_window_path === "string") {
    fd.append("residual_window_path", body.residual_window_path);
  }
  if (typeof body.pixel_image_path === "string") {
    fd.append("pixel_image_path", body.pixel_image_path);
  }

  const res = await fetch(FUSION_URL, { method: "POST", body: fd });
  const json = await res.json().catch(() => ({}));
  return NextResponse.json(json, { status: res.status });
}

export async function POST(req: Request) {
  const ct = req.headers.get("content-type") || "";
  if (ct.includes("multipart/form-data")) {
    return forwardMultipart(req);
  }
  return forwardJSON(req);
}

export async function GET() {
  // proxy FastAPI health for convenience
  const url =
    (process.env.FUSION_URL || "http://127.0.0.1:8000").replace(/\/+$/, "") +
    "/health";
  try {
    const res = await fetch(url);
    const json = await res.json();
    return NextResponse.json(json, { status: res.status });
  } catch (e) {
    return NextResponse.json({ status: "down" }, { status: 502 });
  }
}
