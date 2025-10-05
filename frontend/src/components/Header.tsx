"use client";

import Link from "next/link";

export function Header() {
  return (
    <header className="fixed top-0 inset-x-0 z-50 bg-gradient-to-b from-black/70 to-transparent">
      <div className="h-16 flex items-center justify-center">
        <Link
          href="/"
          className="tracking-[0.2em] uppercase text-xs text-white/70 hover:text-white transition"
        >
          nasa-exo-iden-v1
        </Link>
      </div>
    </header>
  );
}
