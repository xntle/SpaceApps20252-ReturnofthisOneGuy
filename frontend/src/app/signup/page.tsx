// app/signup/page.tsx
"use client";

import React, { useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Header } from "@/components/Header";
import { ExoDotsCanvas } from "@/components/ExoDotsCanvas";
// import StarshadeDotMark from "@/components/ui/starshade-logo"; // no space for this rn :*()

function isValidEmail(v: string) {
  return /\S+@\S+\.\S+/.test(v);
}
function passwordScore(pw: string) {
  let s = 0;
  if (pw.length >= 8) s++;
  if (/[A-Z]/.test(pw)) s++;
  if (/[a-z]/.test(pw)) s++;
  if (/\d/.test(pw)) s++;
  if (/[^A-Za-z0-9]/.test(pw)) s++;
  return Math.min(s, 5);
}

export default function SignupPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [pw2, setPw2] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [agree, setAgree] = useState(false);

  const score = passwordScore(pw);
  const strength = useMemo(
    () =>
      ["too weak", "weak", "ok", "good", "strong"][Math.max(0, score - 1)] ??
      "too weak",
    [score]
  );

  const emailOk = isValidEmail(email);
  const pwMatch = pw.length > 0 && pw === pw2;
  const pwOk = score >= 3;
  const canSubmit =
    name.trim().length > 1 && emailOk && pwOk && pwMatch && agree;

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    // TODO: API SIGNUP
    console.log({ name, email });
  };

  return (
    <main className="relative min-h-dvh w-full bg-black text-white overflow-x-hidden">
      <Header />

      <div className="fixed inset-0 z-0 opacity-35 pointer-events-none">
        <ExoDotsCanvas />
      </div>

      <div className="pointer-events-none fixed inset-x-0 bottom-0 h-[32dvh] bg-[radial-gradient(60%_60%_at_50%_0%,rgba(75,119,255,0.10),transparent_60%)]" />

      <section className="relative z-10 grid min-h-dvh place-items-center px-6 pt-16">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="w-full max-w-md"
        >
          <div className="mb-6 text-center">
            {/* <StarshadeDotMark size={96} className="block mx-auto mb-5" /> */}
            <h1 className="text-[clamp(24px,3.2vw,32px)] font-semibold tracking-tight">
              create your account
            </h1>
            <p className="mt-2 text-white/70">join the hunt for new worlds</p>
          </div>

          <form
            onSubmit={onSubmit}
            className="rounded-3xl border border-white/10 bg-white/[0.02] backdrop-blur p-6 md:p-8
                       max-h-[min(84dvh,720px)] overflow-auto overscroll-contain"
          >
            <div className="space-y-6">
              <div>
                <label htmlFor="name" className="sr-only">
                  name
                </label>
                <input
                  id="name"
                  type="text"
                  placeholder="name"
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-transparent border-b border-white/20 focus:border-white/60 outline-none py-3 placeholder-white/40 text-base"
                />
              </div>

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
                {!emailOk && email.length > 0 && (
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
                    placeholder="password (8+ chars)"
                    autoComplete="new-password"
                    value={pw}
                    onChange={(e) => setPw(e.target.value)}
                    aria-invalid={pw.length > 0 && !pwOk}
                    className="w-full bg-transparent border-b border-white/20 focus:border-white/60 outline-none py-3 placeholder-white/40 text-base"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw((s) => !s)}
                    className="text-xs text-white/60 hover:text-white transition px-2 py-1"
                    aria-pressed={showPw}
                  >
                    {showPw ? "hide" : "show"}
                  </button>
                </div>
                <div className="mt-2">
                  <div className="h-1 w-full bg-white/10 rounded">
                    <div
                      className="h-1 bg-white rounded transition-all"
                      style={{ width: `${(score / 5) * 100}%` }}
                    />
                  </div>
                  <p className="mt-1 text-xs text-white/60">
                    strength: {strength}
                  </p>
                </div>
              </div>

              <div>
                <label htmlFor="password2" className="sr-only">
                  confirm password
                </label>
                <input
                  id="password2"
                  type={showPw ? "text" : "password"}
                  placeholder="confirm password"
                  autoComplete="new-password"
                  value={pw2}
                  onChange={(e) => setPw2(e.target.value)}
                  aria-invalid={pw2.length > 0 && !pwMatch}
                  className="w-full bg-transparent border-b border-white/20 focus:border-white/60 outline-none py-3 placeholder-white/40 text-base"
                />
                {pw2.length > 0 && !pwMatch && (
                  <p className="mt-1 text-xs text-red-300">
                    passwords don’t match
                  </p>
                )}
              </div>

              <label className="flex items-start gap-3 text-sm text-white/70 select-none">
                <input
                  type="checkbox"
                  checked={agree}
                  onChange={(e) => setAgree(e.target.checked)}
                  className="mt-1 h-4 w-4 accent-white"
                />
                <span>
                  I agree to the{" "}
                  <Link href="/terms" className="underline hover:text-white">
                    Terms
                  </Link>{" "}
                  and{" "}
                  <Link href="/privacy" className="underline hover:text-white">
                    Privacy Policy
                  </Link>
                  .
                </span>
              </label>

              <button
                type="submit"
                disabled={!canSubmit}
                className="w-full mt-2 px-6 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 disabled:opacity-40 disabled:pointer-events-none transition"
              >
                create account
              </button>

              {/* Optional social providers */}
              <div className="relative">
                <div className="my-6 h-px w-full bg-white/10" />
                <div className="flex items-center justify-center gap-3">
                  <button
                    type="button"
                    className="px-4 py-2 rounded-xl border border-white/15 text-white/80 hover:bg-white/10 transition"
                  >
                    continue with google
                  </button>
                  <button
                    type="button"
                    className="px-4 py-2 rounded-xl border border-white/15 text-white/80 hover:bg-white/10 transition"
                  >
                    github
                  </button>
                </div>
              </div>
            </div>
          </form>

          <div className="mt-6 text-center text-sm text-white/60">
            already have an account?{" "}
            <Link href="/login" className="underline hover:text-white">
              log in
            </Link>
          </div>
        </motion.div>
      </section>
    </main>
  );
}
