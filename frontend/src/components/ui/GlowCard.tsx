"use client";
import React from "react";

export function GlowCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`relative rounded-3xl border border-white/10 bg-white/[0.03] backdrop-blur p-6 ${className}`}
    >
      <div className="pointer-events-none absolute -inset-px rounded-3xl bg-[radial-gradient(60%_60%_at_50%_-10%,rgba(120,150,255,0.14),transparent_60%)]" />
      {children}
    </div>
  );
}
