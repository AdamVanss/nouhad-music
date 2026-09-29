import type { Metadata } from "next";
import { SeparateClient } from "./SeparateClient";

export const metadata: Metadata = { title: "Nouhad" };

export default function SeparatePage() {
  return (
    <div className="page-gutter mx-auto w-full max-w-5xl py-[var(--section-pad-y)]">
      <p className="eyebrow">Nouhad · Audio</p>
      <h1 className="mt-3 text-4xl text-white sm:text-5xl">Split a song into four stems.</h1>
      <p className="mt-4 text-base leading-relaxed text-muted-foreground">
        Vocals, drums, bass and everything else come back as separate WAV files you can solo, mute and download.
      </p>
      <div className="mt-10">
        <SeparateClient />
      </div>
    </div>
  );
}
