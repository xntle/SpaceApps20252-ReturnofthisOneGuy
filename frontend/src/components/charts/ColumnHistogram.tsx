"use client";
import React, { useMemo } from "react";

export function ColumnHistogram({
  rows,
  col,
  bins = 24,
  title,
  unit,
}: {
  rows: Record<string, string>[];
  col: string;
  bins?: number;
  title?: string;
  unit?: string;
}) {
  const { xs, ys, min, max } = useMemo(() => {
    const vals = rows.map((r) => Number(r[col])).filter(Number.isFinite);
    const mn = Math.min(...vals),
      mx = Math.max(...vals);
    const n = Math.max(1, bins);
    const counts = new Array(n).fill(0);
    for (const v of vals) {
      const i = Math.min(
        n - 1,
        Math.max(0, Math.floor(((v - mn) / (mx - mn || 1)) * n))
      );
      counts[i]++;
    }
    return {
      xs: counts.map((_, i) => mn + (i + 0.5) * ((mx - mn) / n || 1)),
      ys: counts,
      min: mn,
      max: mx,
    };
  }, [rows, col, bins]);

  const ymax = Math.max(1, ...ys);
  return (
    <div>
      <div className="text-xs text-white/60 uppercase tracking-wide mb-2">
        {title || col}
      </div>
      <div className="relative h-[120px] w-full overflow-hidden rounded-xl border border-white/10 bg-black/40 p-2">
        <svg viewBox="0 0 640 120" className="w-full h-full">
          {ys.map((y, i) => {
            const w = 640 / ys.length;
            const h = (y / ymax) * 100;
            return (
              <rect
                key={i}
                x={i * w}
                y={110 - h}
                width={Math.max(1, w - 2)}
                height={h}
                fill="white"
                opacity="0.75"
              />
            );
          })}
        </svg>
        <div className="absolute bottom-1 left-2 right-2 flex justify-between text-[10px] text-white/50">
          <span>
            {Number.isFinite(min) ? min.toFixed(2) : "—"}
            {unit}
          </span>
          <span>
            {Number.isFinite(max) ? max.toFixed(2) : "—"}
            {unit}
          </span>
        </div>
      </div>
    </div>
  );
}
