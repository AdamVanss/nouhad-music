import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="border-t border-border">
      <div className="page-gutter flex flex-col gap-6 py-10 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-display text-lg text-white">Nouhad &amp; Adam</p>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Nouhad&apos;s stem separator and Adam&apos;s face classifier, on one local GPU.
          </p>
          <p className="eyebrow mt-3">Educational use · Cambridge-MT, MUSDB18 and UTKFace data</p>
        </div>
        <nav aria-label="Footer" className="flex flex-col sm:flex-row sm:gap-x-2">
          <Link
            href="/separate"
            className="pressable flex min-h-[var(--touch-min)] items-center text-sm text-foreground/80 hover:text-lime sm:px-2"
          >
            Nouhad
          </Link>
          <Link
            href="/classify"
            className="pressable flex min-h-[var(--touch-min)] items-center text-sm text-foreground/80 hover:text-lime sm:px-2"
          >
            Adam
          </Link>
        </nav>
      </div>
    </footer>
  );
}
