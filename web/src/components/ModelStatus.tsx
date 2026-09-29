"use client";

import { useEffect, useState } from "react";
import { health, type Health } from "@/lib/api";

export function ModelStatus() {
  const [state, setState] = useState<Health | null | "loading">("loading");

  useEffect(() => {
    let alive = true;
    health().then((value) => alive && setState(value));
    return () => {
      alive = false;
    };
  }, []);

  const online = state !== "loading" && state !== null;
  return (
    <p className="eyebrow flex items-center gap-2" role="status">
      <span
        aria-hidden
        className={online ? "h-1.5 w-1.5 rounded-full bg-lime" : "h-1.5 w-1.5 rounded-full bg-white/30"}
      />
      {state === "loading" && "Checking model server"}
      {state === null && "Model server offline"}
      {online && `Model server on ${state.device.toUpperCase()} · ${state.separator}`}
    </p>
  );
}
