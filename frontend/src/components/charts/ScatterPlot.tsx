"use client";
import React, { useMemo } from "react";

export function ScatterPlot({
  rows,
  xKey,
  yKey,
  selected,
  title,
  unitX,
  unitY,
}: {
  rows: Record<string, string>[];
  xKey: string;
  yKey: string;
  selected?: number | null;
  title?: string;
  unitX?: string;
  unitY?: string;
}) {
  const pts = useMemo(
    () =>
      rows
        .map((r, i) => ({
          i,
          x: Number(r[xKey]),
          y: Number(r[yKey]),
        }))
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)),
    [rows, xKey, yKey]
  );

  const xs = pts.map((p) => p.x),
    ys = pts.map((p) => p.y);
  const xmin = Math.min(...xs),
    xmax = Math.max(...xs);
  const ymin = Math.min(...ys),
    ymax = Math.max(...ys);
  const X = (x: number) => 10 + ((x - xmin) / (xmax - xmin || 1)) * 620;
  const Y = (y: number) => 110 - ((y - ymin) / (ymax - ymin || 1)) * 100;

  return (
    <div>
      <div className="text-xs text-white/60 uppercase tracking-wide mb-2">
        {title || `${xKey} vs ${yKey}`}
      </div>
      <div className="relative h-[120px] w-full overflow-hidden rounded-xl border border-white/10 bg-black/40 p-2">
        <svg viewBox="0 0 640 120" className="w-full h-full">
          {/* axes */}
          <line
            x1="10"
            y1="110"
            x2="630"
            y2="110"
            stroke="white"
            strokeOpacity=".15"
          />
          <line
            x1="10"
            y1="10"
            x2="10"
            y2="110"
            stroke="white"
            strokeOpacity=".15"
          />
          {/* points */}
          {pts.map((p) => (
            <circle
              key={p.i}
              cx={X(p.x)}
              cy={Y(p.y)}
              r={selected === p.i ? 4.5 : 3}
              fill={selected === p.i ? "white" : "rgba(255,255,255,0.7)"}
            />
          ))}
        </svg>
        <div className="absolute bottom-1 left-2 right-2 flex justify-between text-[10px] text-white/50">
          <span>
            {xKey}
            {unitX ? ` (${unitX})` : ""}
          </span>
          <span>
            {yKey}
            {unitY ? ` (${unitY})` : ""}
          </span>
        </div>
      </div>
    </div>
  );
}
