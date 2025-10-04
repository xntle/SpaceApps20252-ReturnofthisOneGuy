"use client";

import Link from "next/link";
import { motion } from "framer-motion";

export function Hero() {
  return (
    <section className="relative z-10 flex min-h-[100svh] items-center justify-center">
      <div className="text-center px-6">
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
            href="/login"
            className="px-6 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 transition focus:outline-none focus:ring-2 focus:ring-white/40"
          >
            Start
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
