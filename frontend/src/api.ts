import type { BatchStatus, Box, Frame, ImageRow, Meta } from "./types"

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      if (typeof body.detail === "string") detail = body.detail
    } catch {
      detail = response.statusText
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  meta: () => request<Meta>("/api/meta"),
  open: (path: string) =>
    request<Meta["dataset"]>("/api/dataset", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  images: () => request<{ images: ImageRow[] }>("/api/images"),
  labels: (name: string) => request<Frame>(`/api/labels/${encodeURIComponent(name)}`),
  save: (name: string, boxes: Box[]) =>
    request<Frame>(`/api/labels/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify({ boxes }),
    }),
  detect: (name: string, threshold: number) =>
    request<Frame>(`/api/detect/${encodeURIComponent(name)}`, {
      method: "POST",
      body: JSON.stringify({ threshold }),
    }),
  batch: () => request<BatchStatus>("/api/batch"),
  batchStart: (threshold: number, hold: string | null) =>
    request<BatchStatus>("/api/batch", {
      method: "POST",
      body: JSON.stringify({ threshold, hold }),
    }),
  batchCancel: () =>
    request<BatchStatus>("/api/batch/cancel", {
      method: "POST",
      body: "{}",
    }),
  mediaUrl: (name: string) => `/api/media/${encodeURIComponent(name)}`,
}
