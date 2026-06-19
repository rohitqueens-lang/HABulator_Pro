'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import Header from '@/components/Header'
import GroupTabs from '@/components/GroupTabs'
import InputPanel from '@/components/InputPanel'
import ResultPanel from '@/components/ResultPanel'
import { predict, warmUp, ApiError } from '@/lib/api'
import { FEATURES } from '@/lib/utils'
import type { InputFeatures, PredictionResult } from '@/lib/types'

function buildDefaults(): InputFeatures {
  return Object.fromEntries(
    FEATURES.map((f) => [f.key, f.defaultValue])
  ) as unknown as InputFeatures
}

export default function HomePage() {
  const [activeGroup, setActiveGroup] = useState('EDIAT')
  const [values, setValues] = useState<InputFeatures>(buildDefaults)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [waking, setWaking] = useState(false)
  const wakeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Warm the (free-tier) backend on first load so it is awake by predict time.
  useEffect(() => {
    void warmUp()
  }, [])

  const handleChange = useCallback((key: keyof InputFeatures, value: number) => {
    setValues((prev) => ({ ...prev, [key]: value }))
  }, [])

  const handlePredict = useCallback(async () => {
    setLoading(true)
    setError(null)
    // If the call is slow (cold start), surface a reassuring notice after ~5s.
    if (wakeTimer.current) clearTimeout(wakeTimer.current)
    wakeTimer.current = setTimeout(() => setWaking(true), 5_000)
    try {
      const res = await predict(activeGroup, values)
      setResult(res)
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail ?? err.message)
      } else if (err instanceof Error) {
        if (err.name === 'TimeoutError' || err.name === 'AbortError' || err.message.includes('timeout')) {
          setError('The model server is waking from sleep and did not respond in time. Please click Predict again in ~30 seconds — it should be ready.')
        } else if (err.message.toLowerCase().includes('fetch') || err.message.toLowerCase().includes('network')) {
          setError('Cannot reach the model server. Please check your connection and try again in a moment.')
        } else {
          setError(err.message)
        }
      } else {
        setError('An unexpected error occurred.')
      }
    } finally {
      if (wakeTimer.current) clearTimeout(wakeTimer.current)
      setWaking(false)
      setLoading(false)
    }
  }, [activeGroup, values])

  const handleGroupChange = useCallback((id: string) => {
    setActiveGroup(id)
    setResult(null)
    setError(null)
  }, [])

  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main className="flex-1 mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-6">
        {/* Group tabs */}
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div className="flex flex-col gap-1.5">
            <motion.h1
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.05 }}
              className="font-display text-xl font-semibold tracking-tight text-ink-100"
            >
              Phytoplankton biovolume prediction
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              className="text-sm font-medium text-ink-400"
            >
              Select an algal group, set environmental conditions, and run the model.
            </motion.p>
          </div>
          <GroupTabs activeGroup={activeGroup} onGroupChange={handleGroupChange} />
        </div>

        {/* Error banner */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="rounded-xl px-4 py-3 flex items-start gap-3"
            style={{
              background: 'rgba(239,68,68,0.08)',
              border: '1px solid rgba(239,68,68,0.25)',
            }}
            role="alert"
          >
            <span className="text-red-400 font-semibold text-sm flex-shrink-0 mt-0.5">Error</span>
            <p className="text-red-300/80 text-sm leading-relaxed">{error}</p>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-400/60 hover:text-red-300/80 transition-colors text-xs font-medium flex-shrink-0"
            >
              Dismiss
            </button>
          </motion.div>
        )}

        {/* Cold-start notice */}
        {loading && waking && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl px-4 py-3 flex items-center gap-3"
            style={{
              background: 'rgba(15,92,140,0.08)',
              border: '1px solid rgba(15,92,140,0.25)',
            }}
            role="status"
          >
            <span
              className="h-3.5 w-3.5 flex-shrink-0 rounded-full border-2 border-sky-400/40 border-t-sky-400 animate-spin"
              aria-hidden
            />
            <p className="text-sky-300/90 text-sm leading-relaxed">
              Waking the model server — the first request after a period of inactivity can take
              ~30–60&nbsp;seconds. Hang tight…
            </p>
          </motion.div>
        )}

        {/* Two-column layout */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[420px_1fr] xl:grid-cols-[460px_1fr]">
          <InputPanel
            values={values}
            onChange={handleChange}
            onPredict={handlePredict}
            loading={loading}
          />
          <ResultPanel result={result} loading={loading} />
        </div>
      </main>

      {/* Footer */}
      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="mt-6 border-t border-line py-6"
      >
        <div className="mx-auto flex max-w-7xl flex-col items-center gap-2 px-4 text-center sm:px-6 lg:px-8">
          <p className="font-mono text-[11px] font-medium text-ink-500">
            Habulator v1.0
          </p>
          <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[11px] text-ink-400">
            <span>© 2026 Rohit Shukla</span>
            <span aria-hidden className="text-line-strong">·</span>
            <Link
              href="/privacy"
              className="font-medium underline-offset-2 transition-colors hover:text-ink-200 hover:underline"
            >
              Privacy
            </Link>
          </div>
        </div>
      </motion.footer>
    </div>
  )
}
