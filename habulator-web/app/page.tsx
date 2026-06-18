'use client'

import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import Header from '@/components/Header'
import GroupTabs from '@/components/GroupTabs'
import InputPanel from '@/components/InputPanel'
import ResultPanel from '@/components/ResultPanel'
import { predict, ApiError } from '@/lib/api'
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

  const handleChange = useCallback((key: keyof InputFeatures, value: number) => {
    setValues((prev) => ({ ...prev, [key]: value }))
  }, [])

  const handlePredict = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await predict(activeGroup, values)
      setResult(res)
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail ?? err.message)
      } else if (err instanceof Error) {
        if (err.name === 'TimeoutError' || err.message.includes('timeout')) {
          setError('Request timed out. Make sure the API server is running.')
        } else if (err.message.includes('fetch')) {
          setError('Cannot reach the API server. Please check your connection and that the backend is running.')
        } else {
          setError(err.message)
        }
      } else {
        setError('An unexpected error occurred.')
      }
    } finally {
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

