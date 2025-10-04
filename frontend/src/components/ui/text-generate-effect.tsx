// components/ui/text-generate-effect.tsx
"use client";

import { useEffect, useMemo } from "react";
import { motion, stagger, useAnimate } from "motion/react";
import { cn } from "@/lib/utils";

export const TextGenerateEffect = ({
  words,
  className,
  filter = true,
  duration = 0.5,
  staggerDelay = 0.2,
  onComplete,
}: {
  words: string;
  className?: string;
  filter?: boolean;
  duration?: number;
  staggerDelay?: number;
  onComplete?: () => void;
}) => {
  const [scope, animate] = useAnimate();
  const wordsArray = useMemo(() => words.split(" "), [words]);

  useEffect(() => {
    // animate all spans
    animate(
      "span",
      {
        opacity: 1,
        filter: filter ? "blur(0px)" : "none",
      },
      {
        duration: duration ?? 1,
        delay: stagger(staggerDelay),
      }
    );

    // call onComplete after the last word’s delay + duration
    const totalMs =
      ((duration ?? 1) + (wordsArray.length - 1) * staggerDelay) * 1000;
    const t = setTimeout(() => onComplete?.(), totalMs + 40);
    return () => clearTimeout(t);
  }, [animate, filter, duration, staggerDelay, wordsArray.length, onComplete]);

  return (
    <div className={cn("font-bold", className)}>
      <div className="mt-4">
        <div className="text-white text-2xl md:text-4xl leading-snug tracking-wide">
          <motion.div ref={scope}>
            {wordsArray.map((word, idx) => (
              <motion.span
                key={word + idx}
                className="text-white opacity-0"
                style={{ filter: filter ? "blur(10px)" : "none" }}
              >
                {word}{" "}
              </motion.span>
            ))}
          </motion.div>
        </div>
      </div>
    </div>
  );
};
