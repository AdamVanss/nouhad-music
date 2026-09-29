"use client";

import clsx from "clsx";
import { Upload } from "lucide-react";
import { useId, useState, type DragEvent, type ReactNode } from "react";

type Props = {
  accept: string;
  title: string;
  hint: string;
  disabled?: boolean;
  onFile: (file: File) => void;
  children?: ReactNode;
};

export function Dropzone({ accept, title, hint, disabled, onFile, children }: Props) {
  const id = useId();
  const [over, setOver] = useState(false);

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file && !disabled) onFile(file);
  };

  return (
    <label
      htmlFor={id}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      className={clsx(
        "group relative flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 overflow-hidden rounded-[var(--radius-panel)] border border-dashed px-6 py-10 text-center transition-colors duration-200",
        over ? "border-lime bg-lime/[0.06]" : "border-white/20 bg-surface-1 hover:border-lime/60",
        disabled && "pointer-events-none",
        disabled && !children && "opacity-60",
      )}
    >
      <input
        id={id}
        type="file"
        accept={accept}
        disabled={disabled}
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          event.target.value = "";
        }}
      />
      {children ?? (
        <>
          <span className="grid h-11 w-11 place-items-center rounded-[var(--radius-button)] border border-border bg-surface-2 text-lime transition-transform duration-200 group-hover:-translate-y-0.5">
            <Upload aria-hidden className="size-5" />
          </span>
          <span className="font-display text-xl text-white">{title}</span>
          <span className="text-sm text-muted-foreground">{hint}</span>
        </>
      )}
    </label>
  );
}
