"use client";

import clsx from "clsx";
import { Download, Pause, Play } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/Button";

export const STEM_ORDER = ["vocals", "drums", "bass", "other"] as const;
export type StemName = (typeof STEM_ORDER)[number];

const STEM_LABEL: Record<StemName, string> = {
  vocals: "Vocals",
  drums: "Drums",
  bass: "Bass",
  other: "Other",
};

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function StemMixer({ stems, fileName }: { stems: Record<StemName, string>; fileName: string }) {
  const audio = useRef<Partial<Record<StemName, HTMLAudioElement>>>({});
  const graph = useRef<{ ctx: AudioContext; gains: Record<StemName, GainNode> } | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [muted, setMuted] = useState<Record<StemName, boolean>>({
    vocals: false,
    drums: false,
    bass: false,
    other: false,
  });
  const [solo, setSolo] = useState<StemName | null>(null);
  const [playError, setPlayError] = useState<string | null>(null);

  const all = useCallback(
    () => STEM_ORDER.map((name) => audio.current[name]).filter(Boolean) as HTMLAudioElement[],
    [],
  );

  const heard = useCallback(
    (name: StemName) => (solo ? solo === name : !muted[name]),
    [muted, solo],
  );

  const applyMix = useCallback(() => {
    const g = graph.current;
    if (!g) return;
    const now = g.ctx.currentTime;
    for (const name of STEM_ORDER) {
      // Gain, not element.muted. A muted media element feeds silence into Web Audio.
      g.gains[name].gain.setValueAtTime(heard(name) ? 1 : 0, now);
    }
  }, [heard]);

  const ensureGraph = useCallback(() => {
    if (graph.current) return graph.current;
    const ctx = new AudioContext();
    const gains = {} as Record<StemName, GainNode>;
    for (const name of STEM_ORDER) {
      const el = audio.current[name];
      if (!el) throw new Error("A stem player is missing.");
      el.muted = false;
      const source = ctx.createMediaElementSource(el);
      const gain = ctx.createGain();
      gain.gain.value = heard(name) ? 1 : 0;
      source.connect(gain);
      gain.connect(ctx.destination);
      gains[name] = gain;
    }
    graph.current = { ctx, gains };
    return graph.current;
  }, [heard]);

  useEffect(() => {
    applyMix();
  }, [applyMix]);

  useEffect(() => {
    const leader = audio.current.vocals;
    if (!leader || !playing) return;
    let frame = 0;
    const tick = () => {
      setTime(leader.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  useEffect(() => {
    return () => {
      all().forEach((el) => el.pause());
      void graph.current?.ctx.close();
      graph.current = null;
    };
  }, [all]);

  const toggle = async () => {
    const els = all();
    if (playing) {
      els.forEach((el) => el.pause());
      setPlaying(false);
      return;
    }
    setPlayError(null);
    const at = audio.current.vocals?.currentTime ?? 0;
    try {
      const g = ensureGraph();
      applyMix();
      await g.ctx.resume();
      await Promise.all(
        els.map(async (el) => {
          el.muted = false;
          if (Math.abs(el.currentTime - at) > 0.3) el.currentTime = at;
          await el.play();
        }),
      );
      setPlaying(true);
    } catch {
      els.forEach((el) => el.pause());
      setPlayError("Playback did not start. Press play again.");
    }
  };

  const seek = (value: number) => {
    all().forEach((el) => {
      el.currentTime = value;
    });
    setTime(value);
  };

  const base = fileName.replace(/\.[^.]+$/, "");

  return (
    <section aria-label="Separated stems" className="rise-in rounded-[var(--radius-panel)] border border-border bg-surface-2">
      <div className="flex flex-wrap items-center gap-4 border-b border-border p-4 sm:p-5">
        <Button size="icon" onClick={toggle} aria-label={playing ? "Pause all stems" : "Play all stems"}>
          {playing ? <Pause aria-hidden /> : <Play aria-hidden />}
        </Button>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm text-white">{fileName}</p>
          <div className="mt-2 flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={duration || 0}
              step={0.01}
              value={Math.min(time, duration || 0)}
              onChange={(event) => seek(Number(event.target.value))}
              aria-label="Position"
              className="h-1 w-full cursor-pointer accent-[var(--lime)]"
            />
            <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
              {formatTime(time)} / {formatTime(duration)}
            </span>
          </div>
        </div>
      </div>
      {playError && (
        <p role="alert" className="border-b border-border px-4 py-2 text-sm text-red-200 sm:px-5">
          {playError}
        </p>
      )}
      {solo && (
        <p className="border-b border-border px-4 py-2 text-sm text-muted-foreground sm:px-5">
          Hearing {STEM_LABEL[solo]} only. Press Solo again to hear the whole song.
        </p>
      )}

      <ul className="divide-y divide-border">
        {STEM_ORDER.map((name, index) => {
          const audible = solo ? solo === name : !muted[name];
          return (
            <li key={name} className="flex flex-wrap items-center gap-3 px-4 py-3 sm:px-5">
              <audio
                ref={(el) => {
                  if (el) audio.current[name] = el;
                }}
                src={stems[name]}
                preload="auto"
                onLoadedMetadata={(event) => {
                  if (name === "vocals") setDuration(event.currentTarget.duration);
                }}
                onEnded={() => name === "vocals" && setPlaying(false)}
              />
              <span className="w-6 font-mono text-xs tabular-nums text-white/35">{String(index + 1).padStart(2, "0")}</span>
              <span className={clsx("flex-1 font-display text-lg transition-colors", audible ? "text-white" : "text-white/30")}>
                {STEM_LABEL[name]}
              </span>
              <span aria-hidden className="hidden h-6 w-28 items-end gap-[3px] sm:flex">
                {Array.from({ length: 14 }, (_, bar) => (
                  <span
                    key={bar}
                    className={clsx("w-1 transition-colors", audible && playing ? "bg-lime/70" : "bg-white/15")}
                    style={{ height: `${25 + ((bar * 37 + index * 53) % 70)}%` }}
                  />
                ))}
              </span>
              <Button
                variant={solo === name ? "primary" : "secondary"}
                size="sm"
                aria-pressed={solo === name}
                onClick={() => setSolo((current) => (current === name ? null : name))}
              >
                Solo
              </Button>
              <Button
                variant="secondary"
                size="sm"
                aria-pressed={muted[name]}
                className={clsx(muted[name] && "border-destructive/60 text-destructive")}
                onClick={() => setMuted((current) => ({ ...current, [name]: !current[name] }))}
              >
                Mute
              </Button>
              <a
                href={stems[name]}
                download={`${base}_${name}.wav`}
                aria-label={`Download ${STEM_LABEL[name]}`}
                className="pressable grid h-8 w-8 place-items-center rounded-[var(--radius-button)] border border-border text-foreground/80 hover:border-lime hover:text-lime"
              >
                <Download aria-hidden className="size-4" />
              </a>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
