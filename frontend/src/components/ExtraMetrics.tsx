"use client";
import React from "react";

export function ExtrasMetrics({
  logit,
  entropy,
  used_cnn,
}: {
  logit?: number;
  entropy?: number;
  used_cnn?: boolean;
}) {
  const round = (x?: number) =>
    Number.isFinite(x!) ? (x as number).toFixed(2) : "—";
  return (
    <div className="grid grid-cols-3 gap-3">
      <Card label="Logit" value={round(logit)} />
      <Card label="Entropy" value={round(entropy)} />
      <Card label="CNN data" value={used_cnn ? "yes" : "no"} />
    </div>
  );
}
function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
      <div className="text-xs text-white/60 uppercase tracking-wide">
        {label}
      </div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
  );
}
