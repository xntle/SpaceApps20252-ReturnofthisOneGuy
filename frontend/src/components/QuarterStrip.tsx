"use client";
import React from "react";

export function QuartersStrip({ quarters }: { quarters?: string }) {
  if (!quarters) return null;
  const cells = quarters.trim().split("").slice(0, 18); // cap to 18 for Kepler
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
      <div className="text-xs text-white/60 uppercase tracking-wide mb-2">
        observation quarters
      </div>
      <div className="flex gap-1.5">
        {cells.map((c, i) => (
          <div
            key={i}
            className={`h-4 w-6 rounded ${
              c === "1"
                ? "bg-white/80 shadow-[0_0_10px_rgba(255,255,255,0.45)]"
                : "bg-white/10"
            }`}
            title={`Q${i + 1}: ${c === "1" ? "observed" : "gap"}`}
          />
        ))}
      </div>
    </div>
  );
}
