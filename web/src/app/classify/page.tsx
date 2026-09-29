import type { Metadata } from "next";
import { ClassifyClient } from "./ClassifyClient";

export const metadata: Metadata = { title: "Adam" };

export default function ClassifyPage() {
  return (
    <div className="page-gutter mx-auto w-full max-w-6xl py-[var(--section-pad-y)]">
      <p className="eyebrow">Adam · Faces</p>
      <h1 className="mt-3 text-4xl text-white sm:text-5xl">Boy or girl, from one photo.</h1>
      <p className="mt-4 text-base leading-relaxed text-muted-foreground">
        A 2.4M-parameter convolutional network trained from scratch on 21,000 cropped faces.
      </p>
      <div className="mt-10">
        <ClassifyClient />
      </div>
    </div>
  );
}
