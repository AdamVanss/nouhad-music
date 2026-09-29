export type Health = {
  device: string;
  separator: string;
  face_model: { val_accuracy: number } | null;
};

export type SeparateResult = {
  job: string;
  model: string;
  stems: Record<"vocals" | "drums" | "bass" | "other", string>;
};

export type ClassifyResult = {
  label: "Boy" | "Girl";
  probabilities: Record<"Boy" | "Girl", number>;
  box: [number, number, number, number];
  image_size: [number, number];
  crop: string | null;
  val_accuracy: number | null;
};

async function readError(response: Response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // fall through to the status text
  }
  return `Request failed (${response.status}).`;
}

async function upload<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`/api/${path}`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export const separate = (file: File) => upload<SeparateResult>("separate", file);
export const classify = (file: File) => upload<ClassifyResult>("classify", file);

export async function health(): Promise<Health | null> {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

export const apiUrl = (path: string) => `/api${path}`;
