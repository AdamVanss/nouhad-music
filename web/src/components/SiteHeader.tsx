"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/separate", label: "Nouhad" },
  { href: "/classify", label: "Adam" },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-border/80 bg-background/90 backdrop-blur-sm">
      <div className="page-gutter flex h-[var(--header-h)] items-center justify-between gap-4">
        <Link href="/" className="pressable flex min-h-[var(--touch-min)] items-center gap-3">
          <span aria-hidden className="grid h-9 w-9 grid-cols-2 gap-[3px] bg-surface-1 p-[5px]">
            <span className="bg-lime" />
            <span className="bg-white/80" />
            <span className="bg-white/35" />
            <span className="bg-white/15" />
          </span>
          <span className="font-display text-sm font-semibold tracking-tight text-white sm:text-base">
            Nouhad &amp; Adam
          </span>
        </Link>
        <nav aria-label="Main" className="flex items-center gap-1 sm:gap-6">
          {NAV.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={clsx(
                  "flex min-h-[var(--touch-min)] items-center px-2 text-sm transition-colors duration-200 sm:px-0",
                  item.href === "/" && "hidden sm:flex",
                  active ? "text-lime" : "text-foreground/80 hover:text-white",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
