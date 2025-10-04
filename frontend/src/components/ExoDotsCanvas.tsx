"use client";

import React, { useEffect, useRef } from "react";
import { buildScene, drawFrame, type SceneConfig } from "@/lib/dots";

export function ExoDotsCanvas({ className = "" }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d", { alpha: true })!;

    // Device pixel ratio capped for perf
    const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));

    let w = 0,
      h = 0,
      t = 0,
      raf = 0;

    const mouse = { x: 0, y: 0 };
    const center = { x: 0, y: 0 };

    // Build the scene containers
    const scene: SceneConfig = {
      DPR,
      width: 0,
      height: 0,
      center,
      mouse,
      stars: [],
      planets: [],
      settings: {
        starDensity: 1 / 6500, // stars per px^2
        glow: true,
        swirlStrength: 0.0006,
        maxShadowBlur: 6,
      },
    };

    function resize() {
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      scene.width = w;
      scene.height = h;
      canvas.width = Math.floor(w * DPR);
      canvas.height = Math.floor(h * DPR);
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      center.x = w / 2;
      center.y = h / 2;
      buildScene(scene);
    }

    function onPointer(e: PointerEvent) {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left - w / 2;
      mouse.y = e.clientY - rect.top - h / 2;
    }

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    window.addEventListener("pointermove", onPointer);
    resize();

    function loop() {
      const now = performance.now();
      // keep last in closure via property on loop
      const last = (loop as any)._last ?? now;
      const dt = (now - last) / 1000; // seconds
      (loop as any)._last = now;

      t += dt; // real time delta for smooth animation
      drawFrame(ctx, scene, t);
      raf = requestAnimationFrame(loop);
    }
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("pointermove", onPointer);
    };
  }, []);

  return <canvas ref={canvasRef} className={className + " h-full w-full"} />;
}
