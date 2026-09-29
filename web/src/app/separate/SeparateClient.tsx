"use client";

import { AudioLines } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/Button";
import { Dropzone } from "@/components/Dropzone";
import { STEM_ORDER, StemMixer } from "@/components/StemMixer";
import { apiUrl, separate, type SeparateResult } from "@/lib/api";

type State =
  | { phase: "idle" }
  | { phase: "working"; file: File; startedAt: number }
  | { phase: "done"; file: File; result: SeparateResult; seconds: number }
  | { phase: "error"; message: string };

const ACCEPT = ".wav,.mp3,.flac,.ogg,.m4a,.aac,.opus,.webm,.aif,.aiff,audio/*";

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(since);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(timer);
  }, []);
  return <span className="font-mono tabular-nums">{((now - since) / 1000).toFixed(1)}s</span>;
}

export function SeparateClient() {
  const [state, setState] = useState<State>({ phase: "idle" });
  const run = useRef(0);

  const start = async (file: File) => {
    const id = ++run.current;
    const startedAt = Date.now();
    setState({ phase: "working", file, startedAt });
    try {
      const result = await separate(file);
      if (id !== run.current) return;
      const stems = Object.fromEntries(
        STEM_ORDER.map((name) => [name, apiUrl(result.stems[name])]),
      ) as SeparateResult["stems"];
      setState({ phase: "done", file, result: { ...result, stems }, seconds: (Date.now() - startedAt) / 1000 });
    } catch (error) {
      if (id !== run.current) return;
      setState({ phase: "error", message: error instanceof Error ? error.message : "Separation failed." });
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {state.phase !== "done" && (
        <Dropzone
          accept={ACCEPT}
          title="Drop a song here"
          hint="WAV, MP3, FLAC, M4A or OGG, up to 200 MB. Or click to choose a file."
          disabled={state.phase === "working"}
          onFile={start}
        >
          {state.phase === "working" ? (
            <div className="flex w-full max-w-md flex-col items-center gap-4" role="status" aria-live="polite">
              <AudioLines aria-hidden className="size-6 text-lime" />
              <p className="font-display text-xl text-white">Separating {state.file.name}</p>
              <div className="relative h-1 w-full overflow-hidden bg-white/10">
                <span className="scan-bar absolute inset-y-0 left-0 w-2/5 bg-lime" />
              </div>
              <p className="text-sm text-muted-foreground">
                Elapsed <Elapsed since={state.startedAt} />. A four-minute song usually takes 10 to 30 seconds on the GPU.
              </p>
            </div>
          ) : undefined}
        </Dropzone>
      )}

      {state.phase === "error" && (
        <p role="alert" className="rounded-[var(--radius-input)] border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-red-200">
          {state.message}
        </p>
      )}

      {state.phase === "done" && (
        <>
          <div className="flex flex-wrap items-end justify-between gap-4">
            <p className="text-sm text-muted-foreground">
              Done in <span className="font-mono tabular-nums text-white">{state.seconds.toFixed(1)}s</span> with{" "}
              <span className="text-white">{state.result.model}</span>. Solo a stem to hear it on its own.
            </p>
            <Button variant="outline" onClick={() => setState({ phase: "idle" })}>
              Separate another song
            </Button>
          </div>
          <StemMixer stems={state.result.stems} fileName={state.file.name} />
        </>
      )}
    </div>
  );
}
