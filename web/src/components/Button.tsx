import clsx from "clsx";
import Link from "next/link";
import type { ComponentProps } from "react";

const BASE =
  "pressable inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius-button)] text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0";

const VARIANTS = {
  primary: "bg-primary text-primary-foreground hover:bg-lime-dim",
  secondary: "border border-border bg-surface-3 text-foreground hover:border-lime/40",
  outline: "border border-border bg-transparent hover:border-lime hover:text-lime",
} as const;

const SIZES = {
  default: "h-10 px-5",
  sm: "h-8 px-3.5 text-xs",
  lg: "h-12 px-7 text-base",
  icon: "h-10 w-10",
} as const;

type Style = { variant?: keyof typeof VARIANTS; size?: keyof typeof SIZES };

export function buttonClass({ variant = "primary", size = "default" }: Style = {}, className?: string) {
  return clsx(BASE, VARIANTS[variant], SIZES[size], className);
}

export function Button({ variant, size, className, ...props }: ComponentProps<"button"> & Style) {
  return <button className={buttonClass({ variant, size }, className)} {...props} />;
}

export function ButtonLink({ variant, size, className, ...props }: ComponentProps<typeof Link> & Style) {
  return <Link className={buttonClass({ variant, size }, className)} {...props} />;
}
