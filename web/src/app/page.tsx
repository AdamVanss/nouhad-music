import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { ModelStatus } from "@/components/ModelStatus";

const DOORS = [
  {
    href: "/separate",
    name: "Nouhad",
    kicker: "Audio",
    title: "Split a song into stems.",
    detail: "Vocals, drums, bass and other, as separate files.",
    className: "bg-lime text-primary-foreground hover:bg-lime-dim",
    kickerClass: "text-black/60",
    arrow: "text-primary-foreground",
  },
  {
    href: "/classify",
    name: "Adam",
    kicker: "Faces",
    title: "Boy or girl, from one photo.",
    detail: "A small CNN scores the face it finds.",
    className: "border border-white/20 bg-surface-1 text-white hover:border-lime",
    kickerClass: "text-white/45",
    arrow: "text-lime",
  },
] as const;

export default function HomePage() {
  return (
    <section className="page-gutter flex flex-1 flex-col py-8 sm:py-10">
      <ModelStatus />
      <h1 className="sr-only">Nouhad&apos;s audio tool and Adam&apos;s face classifier</h1>
      <div className="mt-6 grid flex-1 gap-4 md:grid-cols-2 md:gap-5">
        {DOORS.map((door) => (
          <Link
            key={door.href}
            href={door.href}
            className={`pressable group flex min-h-[42vh] flex-col justify-between rounded-[var(--radius-panel)] p-7 sm:min-h-[52vh] sm:p-10 ${door.className}`}
          >
            <span className={`font-mono text-[0.68rem] uppercase tracking-[0.14em] ${door.kickerClass}`}>
              {door.kicker}
            </span>
            <span>
              <span className="block font-display text-6xl leading-none sm:text-7xl lg:text-8xl">{door.name}</span>
              <span className="mt-4 block max-w-sm text-lg leading-snug opacity-80">{door.title}</span>
              <span className="mt-2 block text-sm opacity-60">{door.detail}</span>
            </span>
            <span className={`mt-8 inline-flex items-center gap-2 text-sm font-medium ${door.arrow}`}>
              Open
              <ArrowRight aria-hidden className="size-4 transition-transform duration-200 group-hover:translate-x-1" />
            </span>
          </Link>
        ))}
      </div>
    </section>
  );
}
