"use client";
import React from "react";

export function DecisionBands({
  p,
  thresholds = { green: 0.7, yellow: 0.3 },
}: {
  p: number;
  thresholds?: { green: number; yellow: number };
}) {
  const pct = Math.max(0, Math.min(1, p)) * 100;
  return (
    <div>
      <div className="h-3 w-full rounded bg-gradient-to-r from-rose-500 via-amber-400 to-emerald-400" />
      <div className="relative h-6">
        <div className="absolute left-0 top-0 text-xs text-white/60">0</div>
        <div className="absolute right-0 top-0 text-xs text-white/60">1</div>
        <div
          className="absolute top-0"
          style={{ left: `${pct}%`, transform: "translateX(-50%)" }}
        >
          <div className="w-0 h-0 border-l-4 border-r-4 border-b-8 border-b-white mx-auto" />
        </div>
        {/* thresholds */}
        <div
          className="absolute top-0"
          style={{
            left: `${thresholds.yellow * 100}%`,
            transform: "translateX(-50%)",
          }}
        >
          <div className="w-px h-6 bg-white/40" title="yellow threshold" />
        </div>
        <div
          className="absolute top-0"
          style={{
            left: `${thresholds.green * 100}%`,
            transform: "translateX(-50%)",
          }}
        >
          <div className="w-px h-6 bg-white/60" title="green threshold" />
        </div>
      </div>
    </div>
  );
}
