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

// The backend runs on a free tier that sleeps after ~15 min idle; the first
// request then cold-starts (~30-60s). Allow a generous timeout and one retry so
// a cold start surfaces as a brief wait, not a failure.
const PREDICT_TIMEOUT_MS = 120_000
const MAX_RETRIES = 1

function isRetriable(err: unknown): boolean {
  return (
    err instanceof Error &&
    (err.name === 'TimeoutError' ||
      err.name === 'AbortError' ||
      err.message.toLowerCase().includes('fetch') ||
      err.message.toLowerCase().includes('network'))
  )
}

export async function predict(
  group: string,
  features: InputFeatures
): Promise<PredictionResult> {
  const payload = { group, ...features }

  let lastErr: unknown
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(PREDICT_TIMEOUT_MS),
      })
      return await handleResponse<PredictionResult>(res)
    } catch (err) {
      lastErr = err
      // Real API responses (4xx/5xx -> ApiError) are not transient: fail fast.
      if (err instanceof ApiError) throw err
      if (attempt < MAX_RETRIES && isRetriable(err)) continue
      throw err
    }
  }
  throw lastErr
}

export async function healthCheck(): Promise<{ status: string; model: string; version: string }> {
  const res = await fetch(`${API_BASE}/health`, {
    signal: AbortSignal.timeout(5_000),
  })
  return handleResponse(res)
}

// Fire-and-forget wake-up: pinging /health on page load starts the backend
// spinning up (if asleep) so it is usually ready by the time the user predicts.
// Swallows errors; a generous timeout covers a full cold start.
export async function warmUp(timeoutMs = 120_000): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(timeoutMs),
    })
    return res.ok
  } catch {
    return false
  }
}
