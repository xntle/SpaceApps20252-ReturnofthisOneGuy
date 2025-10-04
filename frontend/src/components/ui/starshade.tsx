// components/StarshadeDotMark.tsx
"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

type Props = {
  size?: number; // rendered px size
  petals?: number; // number of lobes
  outer?: number; // outer radius (svg units)
  inner?: number; // center hole radius
  spacing?: number; // grid spacing (smaller = more dots)
  dotR?: number; // final dot radius
  hex?: boolean; // hex vs square grid
  rotate?: boolean; // slow rotate after forming
  intro?: boolean; // play the "form together" effect on mount
  introDuration?: number; // base duration of fly-out
  className?: string;
};

export default function StarshadeDotMark({
  size = 160,
  petals = 12,
  outer = 56,
  inner = 18,
  spacing = 4.5,
  dotR = 1,
  hex = true,
  rotate = false,
  intro = true,
  introDuration = 1.1,
  className,
}: Props) {
  const amp = outer * 0.22; // scallop amplitude

  const rBoundary = (theta: number) => outer - amp * Math.cos(petals * theta);

  // generate dots and keep metadata for nice intro staggering
  const dots = useMemo(() => {
    const pts: Array<{
      x: number;
      y: number;
      r: number;
      dly: number;
      key: string;
    }> = [];
    const stepY = hex ? spacing * 0.8660254038 : spacing; // sin(60°)
    const stepX = spacing;

    for (let y = -outer; y <= outer; y += stepY) {
      const rowOffset = hex ? (Math.round(y / stepY) & 1 ? stepX / 2 : 0) : 0;
      for (let x = -outer - stepX; x <= outer + stepX; x += stepX) {
        const X = x + rowOffset;
        const r = Math.hypot(X, y);
        if (r < inner || r > outer + 1) continue;

        let theta = Math.atan2(y, X);
        if (theta < 0) theta += Math.PI * 2;

        if (r <= rBoundary(theta) && r >= inner) {
          // delay: ripple from center outward + tiny randomness
          const dly =
            0.05 + (r / outer) * 0.45 + Math.abs(Math.sin(theta * 3)) * 0.06;
          pts.push({ x: X, y, r, dly, key: `${X.toFixed(2)}_${y.toFixed(2)}` });
        }
      }
    }
    return pts.sort((a, b) => a.r - b.r); // inner ➜ outer looks smoother
  }, [outer, inner, spacing, hex, petals, amp]);

  return (
    <svg
      viewBox={`${-outer} ${-outer} ${outer * 2} ${outer * 2}`}
      width={size}
      height={size}
      className={cn(
        "text-white",
        rotate && "motion-safe:animate-[spin_60s_linear_infinite]",
        className
      )}
      aria-label="Starshade-inspired dotted mark (original artwork)"
    >
      {/* dots only */}
      <g fill="currentColor">
        {dots.map(({ x, y, dly, key }) => (
          <motion.circle
            key={key}
            // animate cx/cy/r/opacity for SVG
            initial={
              intro
                ? { cx: 0, cy: 0, r: 0.1, opacity: 0 }
                : { cx: x, cy: y, r: dotR, opacity: 1 }
            }
            animate={{ cx: x, cy: y, r: dotR, opacity: 1 }}
            transition={{
              type: "spring",
              stiffness: 140,
              damping: 18,
              mass: 0.6,
              duration: introDuration,
              delay: intro ? dly : 0,
            }}
          />
        ))}
      </g>

      {/* punch out the aperture (assumes black bg underneath) */}
      <circle r={inner} fill="black" />
    </svg>
  );
}
