// components/RunHistory.tsx
"use client";
import React, { useEffect, useMemo, useState } from "react";

type Item = {
  id: string;
  label: "Planet" | "Non-planet";
  proba: number;
  ts: number;
};

export function addRunToHistory(i: Item) {
  try {
    const k = "exoruns";
    const arr: Item[] = JSON.parse(localStorage.getItem(k) || "[]");
    const next = [i, ...arr].slice(0, 12);
    localStorage.setItem(k, JSON.stringify(next));
  } catch {}
}

export function RunHistory() {
  const [items, setItems] = useState<Item[]>([]);
  useEffect(() => {
    try {
      const k = "exoruns";
      setItems(JSON.parse(localStorage.getItem(k) || "[]"));
    } catch {}
  }, []);

  const points = useMemo(
    () =>
      items
        .slice()
        .reverse()
        .map((d, i) => ({ x: i, y: d.proba })),
    [items]
  );

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
      <div className="text-xs text-white/60 uppercase tracking-wide mb-3">
        recent runs
      </div>

      {/* Sparkline */}
      <div className="h-1 w-full relative mb-4">
        <svg className="absolute inset-0 w-full h-full">
          <polyline
            fill="none"
            stroke="white"
            strokeOpacity="0.8"
            strokeWidth="2"
            points={points
              .map((p, i, a) => {
                const w = a.length > 1 ? i / (a.length - 1) : 0.5;
                const x = w * 100;
                const y = (1 - p.y) * 100;
                return `${x},${y}`;
              })
              .join(" ")}
          />
        </svg>
        <div className="absolute inset-0 border-t border-white/10" />
      </div>

      {/* Last few runs */}
      <div className="grid grid-cols-2 gap-2">
        {items.slice(0, 4).map((r, i) => (
          <div
            key={i}
            className="rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-xs flex items-center justify-between"
          >
            <span className="truncate">{r.id}</span>
            <span
              className={`font-semibold ${
                r.label === "Planet" ? "text-emerald-300" : "text-red-300"
              }`}
            >
              {(r.proba * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
