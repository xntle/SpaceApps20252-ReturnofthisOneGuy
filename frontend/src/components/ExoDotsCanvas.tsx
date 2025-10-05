"use client";

import React, { useEffect, useRef } from "react";
import { buildScene, drawFrame, type SceneConfig } from "@/lib/dots";

export function ExoDotsCanvas({ className = "" }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const lastRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const roRef = useRef<ResizeObserver | null>(null);

  useEffect(() => {
    const cnv = canvasRef.current;
    if (!cnv) return;

    const ctx = cnv.getContext("2d", { alpha: true });
    if (!ctx) return;
    ctxRef.current = ctx;

    const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    let t = 0;

    const mouse = { x: 0, y: 0 };
    const center = { x: 0, y: 0 };

    const scene: SceneConfig = {
      DPR,
      width: 0,
      height: 0,
      center,
      mouse,
      stars: [],
      planets: [],
      settings: {
        starDensity: 1 / 6500,
        glow: true,
        swirlStrength: 0.0006,
        maxShadowBlur: 6,
      },
    };

    const resize = () => {
      const c = canvasRef.current;
      const cctx = ctxRef.current;
      if (!c || !cctx) return;

      const w = c.clientWidth;
      const h = c.clientHeight;

      scene.width = w;
      scene.height = h;

      // set backing store size
      c.width = Math.floor(w * DPR);
      c.height = Math.floor(h * DPR);
      cctx.setTransform(DPR, 0, 0, DPR, 0, 0);

      center.x = w / 2;
      center.y = h / 2;

      buildScene(scene);
    };

    const onPointer = (e: PointerEvent) => {
      const c = canvasRef.current;
      if (!c) return;
      const rect = c.getBoundingClientRect();
      const w = c.clientWidth;
      const h = c.clientHeight;
      mouse.x = e.clientX - rect.left - w / 2;
      mouse.y = e.clientY - rect.top - h / 2;
    };

    const loop = () => {
      const cctx = ctxRef.current;
      const c = canvasRef.current;
      if (!c || !cctx) return; // stop if unmounted

      const now = performance.now();
      const last = lastRef.current ?? now;
      const dt = (now - last) / 1000;
      lastRef.current = now;

      t += dt;
      drawFrame(cctx, scene, t);
      rafRef.current = requestAnimationFrame(loop);
    };

    // Observe size
    if (typeof ResizeObserver !== "undefined") {
      roRef.current = new ResizeObserver(resize);
      roRef.current.observe(cnv);
    }

    window.addEventListener("pointermove", onPointer, { passive: true });
    resize();
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      roRef.current?.disconnect();
      window.removeEventListener("pointermove", onPointer);
      ctxRef.current = null;
    };
  }, []);

  return <canvas ref={canvasRef} className={`${className} h-full w-full`} />;
}
