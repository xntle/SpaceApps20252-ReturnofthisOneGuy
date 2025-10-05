"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { motion, animate, useMotionValue } from "framer-motion";
import StarshadeDotMark from "./ui/starshade";

const PLACEHOLDER_ACTIVE = 128;
const PLACEHOLDER_DATA = 12457890;

function CounterNumber({
  to,
  duration = 1.2,
}: {
  to: number;
  duration?: number;
}) {
  const mv = useMotionValue(0);
  const [val, setVal] = useState(0);

  useEffect(() => {
    const controls = animate(mv, to, { duration, ease: "easeOut" });
    const unsub = mv.on("change", (v) => setVal(v));
    return () => {
      controls.stop();
      unsub();
    };
  }, [mv, to, duration]);

  return <span>{new Intl.NumberFormat().format(Math.round(val))}</span>;
}

export function Hero() {
  return (
    <section className="relative z-10 flex min-h-[100svh] items-center justify-center">
      <div className="text-center px-6">
        {/* centered starshade */}
        <StarshadeDotMark
          size={84}
          petals={18}
          spacing={4.5}
          dotR={1.1}
          intro
          rotate
          className="mx-auto mb-8"
        />

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          className="text-[8vw] leading-[0.95] font-semibold tracking-tight select-none"
        >
          returnofthisoneguy
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 1 }}
          className="mt-6 max-w-2xl mx-auto text-white/70"
        >
          Somejsfdjnfdksnfksd fsdj fjhksd f sd fjs fj sdj fjs jsdf jhs fjds
          fsdjh
        </motion.p>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.6, ease: "easeOut" }}
          className="mt-10 flex items-center justify-center gap-4"
        >
          <Link
            href="/dashboard"
            className="px-6 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 transition focus:outline-none focus:ring-2 focus:ring-white/40"
          >
            Start
          </Link>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0, duration: 0.6, ease: "easeOut" }}
          className="mt-12 grid grid-cols-2 gap-4 max-w-xl mx-auto"
        >
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-4">
            <div className="text-sm text-white/60">Active clients</div>
            <div className="mt-1 text-3xl font-semibold">
              <CounterNumber to={PLACEHOLDER_ACTIVE} />
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-4">
            <div className="text-sm text-white/60">Data processed</div>
            <div className="mt-1 text-3xl font-semibold">
              <CounterNumber to={PLACEHOLDER_DATA} />
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
