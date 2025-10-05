"use client";

import React from "react";

export type ModelOutput = {
  predicted_label: 0 | 1;
  predicted_proba: number;
};

export function ModelIO({ output }: { output: ModelOutput }) {
  const prob = Math.round(output.predicted_proba * 100);
  const verdict =
    output.predicted_proba < 0.3
      ? "Very likely non-planet"
      : output.predicted_proba < 0.7
      ? "Uncertain (needs follow-up)"
      : "Very likely planet";

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
      <div className="text-sm text-white/60 uppercase tracking-wide">
        Model output
      </div>
      <div className="mt-2 text-2xl font-semibold">
        {output.predicted_label === 1 ? "Planet" : "Non-planet"}
      </div>
      <div className="mt-1 text-white/70">
        {verdict} · {prob}%
      </div>
    </div>
  );
}
