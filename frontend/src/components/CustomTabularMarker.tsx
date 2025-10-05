// components/CustomTabularMapper.tsx
"use client";
import React, { useMemo, useState } from "react";

const TARGET_KEYS = [
  "kepid",
  "koi_period",
  "koi_depth",
  "koi_duration",
  "koi_steff",
  "koi_srad",
  "koi_smass",
  "koi_slogg",
  "koi_smet",
  "koi_impact",
] as const;

export function CustomTabularMapper({
  headers,
  rows,
  onPredict,
}: {
  headers: string[];
  rows: Record<string, string>[];
  onPredict: (mappedRow: Record<string, string>) => void;
}) {
  const [mapping, setMapping] = useState<Record<string, string>>(
    Object.fromEntries(TARGET_KEYS.map((k) => [k, ""])) as Record<
      string,
      string
    >
  );
  const canRun = useMemo(() => TARGET_KEYS.every((k) => mapping[k]), [mapping]);
  const [rowIndex, setRowIndex] = useState(0);

  const sample = rows[rowIndex] ?? {};

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 space-y-4">
      <div className="text-sm text-white/70 font-medium">
        Custom CSV → Tabular (map columns)
      </div>

      <div className="grid md:grid-cols-2 gap-3">
        {TARGET_KEYS.map((key) => (
          <label key={key} className="text-sm">
            <div className="text-white/60 mb-1">{key}</div>
            <select
              value={mapping[key]}
              onChange={(e) =>
                setMapping((m) => ({ ...m, [key]: e.target.value }))
              }
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2"
            >
              <option value="">— select a column —</option>
              {headers.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>

      {/* pick which row to run */}
      <div className="flex items-center gap-3 text-sm">
        <span className="text-white/60">Row</span>
        <input
          type="number"
          min={0}
          max={Math.max(0, rows.length - 1)}
          value={rowIndex}
          onChange={(e) =>
            setRowIndex(
              Math.max(
                0,
                Math.min(rows.length - 1, Number(e.target.value) || 0)
              )
            )
          }
          className="w-20 bg-black/40 border border-white/10 rounded-lg px-2 py-1"
        />
        <span className="text-white/40">/ {Math.max(0, rows.length - 1)}</span>
      </div>

      {/* preview mapped values */}
      <div className="text-xs text-white/60">
        Preview →
        <pre className="mt-2 bg-black/40 border border-white/10 rounded-lg p-3 whitespace-pre-wrap">
          {JSON.stringify(
            Object.fromEntries(
              TARGET_KEYS.map((k) => [k, sample[mapping[k] || ""] ?? ""])
            ),
            null,
            2
          )}
        </pre>
      </div>

      <button
        disabled={!canRun || !rows.length}
        onClick={() => {
          const mapped = Object.fromEntries(
            TARGET_KEYS.map((k) => [k, sample[mapping[k] || ""] ?? ""])
          ) as Record<string, string>;
          onPredict(mapped);
        }}
        className="px-5 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 disabled:opacity-40 transition"
      >
        Analyze (use mapped row)
      </button>
    </div>
  );
}
