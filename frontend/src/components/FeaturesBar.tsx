// components/FeatureBars.tsx
"use client";
import React, { useMemo } from "react";

type Dict = Record<string, number>;

export function FeatureBars({
  zvec,
  limit = 8,
  pretty = (k: string) => k.replaceAll("_", " "),
}: {
  zvec?: Dict | null;
  limit?: number;
  pretty?: (k: string) => string;
}) {
  const top = useMemo(() => {
    if (!zvec) return [];
    return Object.entries(zvec)
      .map(([k, v]) => ({ k, v, a: Math.abs(v ?? 0) }))
      .sort((a, b) => b.a - a.a)
      .slice(0, limit);
  }, [zvec, limit]);

  if (!top.length) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-sm text-white/60">
        Run analysis to view feature contributions.
      </div>
    );
  }

  const maxA = top[0].a || 1;

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
      <div className="text-xs text-white/60 uppercase tracking-wide mb-3">
        top feature drivers (|z|)
      </div>
      <div className="space-y-2">
        {top.map(({ k, v, a }) => (
          <div key={k}>
            <div className="flex items-center justify-between text-xs text-white/70">
              <span className="truncate pr-2">{pretty(k)}</span>
              <span className="tabular-nums">{v.toFixed(2)}</span>
            </div>
            <div className="h-2 w-full bg-white/10 rounded">
              <div
                className={`h-2 rounded ${
                  v >= 0 ? "bg-emerald-400" : "bg-red-400"
                }`}
                style={{ width: `${Math.max(6, (a / maxA) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
