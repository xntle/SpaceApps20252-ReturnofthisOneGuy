// components/AnalysisSkeleton.tsx
"use client";
export function AnalysisSkeleton() {
  return (
    <div className="animate-pulse space-y-3">
      <div className="h-6 w-40 bg-white/10 rounded" />
      <div className="h-28 w-full bg-white/10 rounded-xl" />
      <div className="h-4 w-2/3 bg-white/10 rounded" />
      <div className="h-4 w-1/2 bg-white/10 rounded" />
    </div>
  );
}
