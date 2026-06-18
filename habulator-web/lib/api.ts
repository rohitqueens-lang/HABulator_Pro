import type { InputFeatures, PredictionResult } from './types'

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://localhost:8000'

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: string | undefined
    try {
      const body = await res.json()
      detail = body?.detail ?? JSON.stringify(body)
    } catch {
      detail = await res.text()
    }
    throw new ApiError(
      `API request failed with status ${res.status}`,
      res.status,
      detail
    )
  }
  return res.json() as Promise<T>
}

export async function predict(
  group: string,
  features: InputFeatures
): Promise<PredictionResult> {
  const payload = {
    group,
    ...features,
  }

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(30_000),
  })

  return handleResponse<PredictionResult>(res)
}

export async function healthCheck(): Promise<{ status: string; model: string; version: string }> {
  const res = await fetch(`${API_BASE}/health`, {
    signal: AbortSignal.timeout(5_000),
  })
  return handleResponse(res)
}
