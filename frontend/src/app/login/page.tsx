// app/login/page.tsx
"use client";

import React, { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Header } from "@/components/Header";
import { ExoDotsCanvas } from "@/components/ExoDotsCanvas";
import StarshadeDotMark from "@/components/ui/starshade";

function isValidEmail(v: string) {
  return /\S+@\S+\.\S+/.test(v);
}

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [remember, setRemember] = useState(true);

  const emailOk = isValidEmail(email);
  const canSubmit = emailOk && pw.length >= 1;

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    // TODO: AUTH HERE
    console.log({ email, pw, remember });
  };

  return (
    <main className="relative min-h-screen w-full bg-black text-white overflow-hidden">
      <Header />

      <div className="fixed inset-0 z-0 opacity-35 pointer-events-none">
        <ExoDotsCanvas />
      </div>

      <div className="pointer-events-none fixed inset-x-0 -bottom-40 h-[40vh] bg-[radial-gradient(60%_60%_at_50%_0%,rgba(75,119,255,0.10),transparent_60%)]" />

      <section className="relative z-10 flex min-h-[100svh] items-center justify-center px-6">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="w-full max-w-md"
        >
          <div className="mb-8 text-center">
            <StarshadeDotMark size={96} className="block mx-auto mb-6" />
            <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">
              welcome back
            </h1>
            <p className="mt-2 text-white/70">
              sign in to continue your planet hunt
            </p>
          </div>

          <form
            onSubmit={onSubmit}
            className="rounded-3xl border border-white/10 bg-white/[0.02] backdrop-blur p-6 md:p-8"
          >
            <div className="space-y-6">
              <div>
                <label htmlFor="email" className="sr-only">
                  email
                </label>
                <input
                  id="email"
                  type="email"
                  placeholder="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  aria-invalid={email.length > 0 && !emailOk}
                  className="w-full bg-transparent border-b border-white/20 focus:border-white/60 outline-none py-3 placeholder-white/40 text-base"
                />
                {email.length > 0 && !emailOk && (
                  <p className="mt-1 text-xs text-red-300">
                    enter a valid email
                  </p>
                )}
              </div>

              <div>
                <label htmlFor="password" className="sr-only">
                  password
                </label>
                <div className="flex items-center gap-2">
                  <input
                    id="password"
                    type={showPw ? "text" : "password"}
                    placeholder="password"
                    autoComplete="current-password"
                    value={pw}
                    onChange={(e) => setPw(e.target.value)}
                    className="w-full bg-transparent border-b border-white/20 focus:border-white/60 outline-none py-3 placeholder-white/40 text-base"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw((s) => !s)}
                    className="text-xs text-white/60 hover:text-white transition px-2 py-1"
                    aria-pressed={showPw}
                    aria-label="toggle password visibility"
                  >
                    {showPw ? "hide" : "show"}
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between text-sm">
                <label className="flex items-center gap-2 select-none text-white/70">
                  <input
                    type="checkbox"
                    checked={remember}
                    onChange={(e) => setRemember(e.target.checked)}
                    className="h-4 w-4 accent-white"
                  />
                  remember me
                </label>
                <Link
                  href="/reset"
                  className="text-white/60 hover:text-white transition"
                >
                  forgot password?
                </Link>
              </div>

              <button
                type="submit"
                disabled={!canSubmit}
                className="w-full mt-2 px-6 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 disabled:opacity-40 disabled:pointer-events-none transition"
              >
                continue
              </button>

              <div className="relative my-6">
                <div className="h-px w-full bg-white/10" />
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 bg-black px-3 text-xs text-white/50">
                  or
                </span>
              </div>

              <div className="flex items-center justify-center gap-3">
                <button
                  type="button"
                  className="px-4 py-2 rounded-xl border border-white/15 text-white/80 hover:bg-white/10 transition"
                  onClick={() => console.log("oauth: google")}
                >
                  continue with google
                </button>
                <button
                  type="button"
                  className="px-4 py-2 rounded-xl border border-white/15 text-white/80 hover:bg-white/10 transition"
                  onClick={() => console.log("oauth: github")}
                >
                  github
                </button>
              </div>
            </div>
          </form>

          <div className="mt-6 text-center text-sm text-white/60">
            new here?{" "}
            <Link href="/signup" className="underline hover:text-white">
              create an account
            </Link>
          </div>
        </motion.div>
      </section>
    </main>
  );
}
