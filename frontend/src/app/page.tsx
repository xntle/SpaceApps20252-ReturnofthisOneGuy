// app/page.tsx
"use client";

import React, { useRef, useState } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { Hero } from "@/components/Hero";
import { ExoDotsCanvas } from "@/components/ExoDotsCanvas";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { Header } from "@/components/Header";

const INTRO_TEXT =
  "Finding real exoplanets is hard. Signals are scattered across archives and notebooks, and manual vetting slows everything down. What if there were one place that pulled official NASA Kepler/K2/TESS light curves, ran trustworthy AI checks, and let the community review together? It would be comprehensive and easy to use, helping researchers surface the best planet candidates—fast, reproducibly, with full provenance. Wouldn’t that be amazing?";

export default function Page() {
  const introRef = useRef<HTMLDivElement>(null);
  const heroRef = useRef<HTMLDivElement>(null);
  const [showCTA, setShowCTA] = useState(false);

  const { scrollYProgress } = useScroll({
    target: introRef,
    offset: ["end 0.95", "end 0.2"],
  });
  const dotsOpacity = useTransform(scrollYProgress, [0, 1], [0, 1]);

  const goToHero = () => {
    heroRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <main className="relative min-h-screen w-full bg-black text-white overflow-hidden">
      <Header />

      <section
        ref={introRef}
        className="relative z-30 flex min-h-screen items-center justify-center px-6"
      >
        <div className="max-w-5xl text-center [&_*]:text-white">
          <TextGenerateEffect
            words={INTRO_TEXT}
            staggerDelay={0.2}
            duration={0.5}
            onComplete={() => setShowCTA(true)}
          />

          {showCTA && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="mt-8"
            >
              <button
                onClick={goToHero}
                className="px-6 py-3 rounded-2xl font-medium hover:opacity-90 transition focus:outline-none focus:ring-2 focus:ring-white"
              >
                Yes, it would be
              </button>
            </motion.div>
          )}
        </div>
      </section>

      <motion.div
        className="fixed inset-0 z-10 pointer-events-none"
        style={{ opacity: dotsOpacity }}
      >
        <ExoDotsCanvas />
      </motion.div>

      <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <div ref={heroRef}>
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.5 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <Hero />
        </motion.div>
      </div>
    </main>
  );
}
