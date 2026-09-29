import type { NextRequest } from "next/server";

const BACKEND = process.env.MODEL_API_URL ?? "http://127.0.0.1:8000";
const FORWARD_REQUEST = ["content-type", "content-length", "range"];
const FORWARD_RESPONSE = [
  "content-type",
  "content-length",
  "content-range",
  "accept-ranges",
  "last-modified",
  "etag",
];

export const dynamic = "force-dynamic";

async function forward(request: NextRequest, path: string[]) {
  const target = `${BACKEND}/${path.map(encodeURIComponent).join("/")}`;
  const headers = new Headers();
  for (const name of FORWARD_REQUEST) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const hasBody = request.method !== "GET" && request.method !== "HEAD";
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      // required by Node fetch when streaming a request body
      ...(hasBody ? { duplex: "half" } : {}),
      cache: "no-store",
    } as RequestInit);
  } catch {
    return Response.json(
      { detail: "The model server is not running. Start it with: python -m uvicorn src.serve:app --port 8000" },
      { status: 502 },
    );
  }
  const out = new Headers();
  for (const name of FORWARD_RESPONSE) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

type Context = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, ctx: Context) {
  return forward(request, (await ctx.params).path);
}

export async function POST(request: NextRequest, ctx: Context) {
  return forward(request, (await ctx.params).path);
}
