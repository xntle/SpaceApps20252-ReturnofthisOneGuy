"use client";
import React from "react";

export function ConfidenceGauge({ p }: { p: number }) {
  const r = 44,
    c = 2 * Math.PI * r,
    v = Math.max(0, Math.min(1, p));
  const stroke = v >= 0.7 ? "#34d399" : v >= 0.3 ? "#fbbf24" : "#f87171";
  const off = c * (1 - v);
  return (
    <div className="inline-flex items-center gap-3">
      <svg width="110" height="110" viewBox="0 0 110 110">
        <circle
          cx="55"
          cy="55"
          r={r}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth="10"
          fill="none"
        />
        <circle
          cx="55"
          cy="55"
          r={r}
          stroke={stroke}
          strokeWidth="10"
          fill="none"
          strokeDasharray={`${c} ${c}`}
          strokeDashoffset={off}
          transform="rotate(-90 55 55)"
        />
        <text
          x="55"
          y="60"
          textAnchor="middle"
          fontSize="18"
          fill="white"
          fontWeight={600}
        >
          {(v * 100).toFixed(0)}%
        </text>
      </svg>
      <div className="text-sm text-white/70">
        <div className="font-medium text-white">Confidence</div>
        <div>
          {v >= 0.7
            ? "Very likely planet"
            : v >= 0.3
            ? "Uncertain"
            : "Likely non-planet"}
        </div>
      </div>
    </div>
  );
}
