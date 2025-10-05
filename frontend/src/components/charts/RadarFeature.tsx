// components/charts/RadarFeatures.tsx
"use client";
import React, { useMemo } from "react";

type Item = { name: string; z: number };

export function RadarFeatures({
  items,
  title = "top feature z-scores",
}: {
  items: Item[] | undefined | null;
  title?: string;
}) {
  // sanitize inputs
  const clean: Item[] = Array.isArray(items)
    ? items.filter((d) => d && Number.isFinite(d.z))
    : [];

  const data = clean.slice(0, 6); // show up to 6
  const clampZ = (z: number) => Math.max(-6, Math.min(6, z)); // tame outliers

  // If we don't have enough points to make a polygon, show a placeholder card
  if (data.length < 3) {
    return (
      <div>
        <div className="text-xs text-white/60 uppercase tracking-wide mb-2">
          {title}
        </div>
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-6 text-white/60 text-sm">
          Run analysis to view feature z-scores.
        </div>
      </div>
    );
  }

  const R = 70,
    cx = 90,
    cy = 90;
  const maxA = Math.max(1e-6, ...data.map((d) => Math.abs(clampZ(d.z))));

  const pts = useMemo(() => {
    return data.map((d, i) => {
      const a = (i / data.length) * 2 * Math.PI - Math.PI / 2; // start at top
      const r = (Math.abs(clampZ(d.z)) / maxA) * R;
      // Put sign into “direction” for contrast (optional)
      const dir = clampZ(d.z) >= 0 ? 1 : -1;
      const x = cx + r * Math.cos(a) * dir;
      const y = cy + r * Math.sin(a);
      return { x, y };
    });
  }, [data, maxA]);

  // Build a valid path only when we actually have points
  const pathD =
    pts.length >= 3
      ? pts
          .map(
            (p, i) =>
              `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`
          )
          .join(" ") + " Z"
      : "";

  return (
    <div>
      <div className="text-xs text-white/60 uppercase tracking-wide mb-2">
        {title}
      </div>
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black/40 p-3">
        <svg width="180" height="180" className="block mx-auto">
          {/* rings */}
          {[0.33, 0.66, 1].map((k, i) => (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={R * k}
              fill="none"
              stroke="white"
              strokeOpacity={0.08 + i * 0.05}
            />
          ))}
          {/* axes */}
          {data.map((_, i) => {
            const a = (i / data.length) * 2 * Math.PI - Math.PI / 2;
            const x2 = cx + R * Math.cos(a);
            const y2 = cy + R * Math.sin(a);
            return (
              <line
                key={i}
                x1={cx}
                y1={cy}
                x2={x2}
                y2={y2}
                stroke="white"
                strokeOpacity="0.08"
              />
            );
          })}
          {/* polygon (only if valid) */}
          {pathD && (
            <path
              d={pathD}
              fill="rgba(255,255,255,0.15)"
              stroke="white"
              strokeOpacity="0.7"
            />
          )}
        </svg>

        <div className="grid grid-cols-2 gap-2 mt-3 text-[11px] text-white/70">
          {data.map((d, i) => (
            <div key={i} className="truncate">
              {d.name.replaceAll("_", " ")}: {clampZ(d.z).toFixed(2)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
