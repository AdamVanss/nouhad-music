"use client";

import clsx from "clsx";
import { ScanFace } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/Button";
import { Dropzone } from "@/components/Dropzone";
import { classify, type ClassifyResult } from "@/lib/api";

type State =
  | { phase: "idle" }
  | { phase: "working"; preview: string }
  | { phase: "done"; preview: string; result: ClassifyResult }
  | { phase: "error"; preview: string | null; message: string };

const CLASSES = ["Boy", "Girl"] as const;

export function ClassifyClient() {
  const [state, setState] = useState<State>({ phase: "idle" });
  const previewRef = useRef<string | null>(null);
  const run = useRef(0);

  useEffect(() => () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
  }, []);

  const start = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setState({ phase: "error", preview: null, message: "Choose a JPG, PNG or WebP photo." });
      return;
    }
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const preview = URL.createObjectURL(file);
    previewRef.current = preview;
    const id = ++run.current;
    setState({ phase: "working", preview });
    try {
      const result = await classify(file);
      if (id === run.current) setState({ phase: "done", preview, result });
    } catch (error) {
      if (id === run.current) {
        setState({ phase: "error", preview, message: error instanceof Error ? error.message : "Classification failed." });
      }
    }
  };

  const preview = state.phase === "idle" ? null : state.preview;
  const result = state.phase === "done" ? state.result : null;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
      <div className="flex flex-col gap-4">
        <Dropzone
          accept="image/*"
          title="Drop a face photo"
          hint="A clear, front-facing photo works best. JPG, PNG or WebP."
          disabled={state.phase === "working"}
          onFile={start}
        >
          {preview ? (
            <span className="relative inline-block max-h-[26rem]">
              {/* eslint-disable-next-line @next/next/no-img-element -- local object URL */}
              <img src={preview} alt="Uploaded photo" className="max-h-[26rem] w-auto rounded-[var(--radius-input)] object-contain" />
              {result && (
                <span
                  aria-hidden
                  className="rise-in absolute border-2 border-lime"
                  style={{
                    left: `${(result.box[0] / result.image_size[0]) * 100}%`,
                    top: `${(result.box[1] / result.image_size[1]) * 100}%`,
                    width: `${(result.box[2] / result.image_size[0]) * 100}%`,
                    height: `${(result.box[3] / result.image_size[1]) * 100}%`,
                  }}
                />
              )}
              {state.phase === "working" && (
                <span className="absolute inset-x-0 bottom-0 h-1 overflow-hidden bg-black/40">
                  <span className="scan-bar absolute inset-y-0 left-0 w-2/5 bg-lime" />
                </span>
              )}
            </span>
          ) : undefined}
        </Dropzone>
        {preview && state.phase !== "working" && (
          <p className="text-sm text-muted-foreground">Drop or click the photo to try another one.</p>
        )}
      </div>

      <section
        aria-label="Prediction"
        aria-live="polite"
        className="flex flex-col rounded-[var(--radius-panel)] border border-border bg-surface-2 p-5 sm:p-6"
      >
        <p className="eyebrow">Prediction</p>
        {state.phase === "idle" && (
          <div className="flex flex-1 flex-col justify-center gap-3 py-8">
            <ScanFace aria-hidden className="size-6 text-white/40" />
            <p className="text-sm leading-relaxed text-muted-foreground">
              The server finds the largest face, crops it the way the training photos were cropped, and the CNN scores it.
            </p>
          </div>
        )}
        {state.phase === "working" && <p className="mt-6 text-sm text-muted-foreground">Finding a face…</p>}
        {state.phase === "error" && (
          <p role="alert" className="mt-6 rounded-[var(--radius-input)] border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-red-200">
            {state.message}
          </p>
        )}
        {result && (
          <div className="rise-in mt-5 flex flex-col gap-6">
            <div className="flex items-center gap-5">
              {result.crop && (
                // eslint-disable-next-line @next/next/no-img-element -- base64 crop from the API
                <img src={result.crop} alt="Face crop the model saw" className="h-24 w-24 rounded-[var(--radius-input)] object-cover" />
              )}
              <div>
                <p className="font-display text-5xl text-white">{result.label}</p>
                <p className="mt-1 font-mono text-sm tabular-nums text-lime">
                  {(result.probabilities[result.label] * 100).toFixed(1)}% confident
                </p>
              </div>
            </div>
            <ul className="flex flex-col gap-4">
              {CLASSES.map((name) => {
                const p = result.probabilities[name];
                return (
                  <li key={name}>
                    <div className="flex items-baseline justify-between text-sm">
                      <span className={clsx(name === result.label ? "text-white" : "text-muted-foreground")}>{name}</span>
                      <span className="font-mono tabular-nums text-white/70">{(p * 100).toFixed(1)}%</span>
                    </div>
                    <div className="mt-2 h-2 bg-white/10">
                      <div
                        className={clsx("h-full origin-left transition-transform duration-500", name === result.label ? "bg-lime" : "bg-white/35")}
                        style={{ transform: `scaleX(${p})` }}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
            <p className="text-xs leading-relaxed text-white/45">
              A guess from appearance only, learned from UTKFace labels
              {result.val_accuracy ? ` (${(result.val_accuracy * 100).toFixed(1)}% right on held-out faces)` : ""}. It
              says nothing about who someone is.
            </p>
            <Button variant="outline" className="self-start" onClick={() => setState({ phase: "idle" })}>
              Clear
            </Button>
          </div>
        )}
      </section>
    </div>
  );
}
