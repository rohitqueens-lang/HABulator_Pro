'use client'

import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence, useMotionValue, useTransform, animate } from 'framer-motion'
import { Waves, BarChart3 } from 'lucide-react'
import PredictionScale from './PredictionScale'
import ShapChart from './ShapChart'
import { formatMgL } from '@/lib/utils'
import type { PredictionResult } from '@/lib/types'

interface ResultPanelProps {
  result: PredictionResult | null
  loading: boolean
}

function AnimatedNumber({ target, format }: { target: number; format: (v: number) => string }) {
  const mv = useMotionValue(0)
  const displayRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    const controls = animate(mv, target, {
      duration: 1.2,
      ease: [0.25, 0.46, 0.45, 0.94],
      onUpdate: (v) => {
        if (displayRef.current) {
          displayRef.current.textContent = format(v)
        }
      },
    })
    return () => controls.stop()
  }, [target, mv, format])

  return (
    <span
      ref={displayRef}
      className="tnum font-mono font-semibold text-ink-100"
      style={{ fontSize: 'clamp(2rem, 5vw, 3.25rem)', lineHeight: 1, letterSpacing: '-0.02em' }}
    >
      {format(0)}
    </span>
  )
}

function Divider() {
  return (
    <div
      className="w-full"
      style={{ height: '1px', background: 'rgba(15,23,32,0.08)' }}
    />
  )
}

function SkeletonBlock({ h, w = '100%' }: { h: number; w?: string }) {
  return (
    <div
      className="skeleton rounded-lg"
      style={{ height: h, width: w }}
    />
  )
}

function LoadingState() {
  return (
    <motion.div
      key="loading"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col gap-6 px-6 py-6"
    >
      {/* Big number skeleton */}
      <div className="flex flex-col gap-3">
        <SkeletonBlock h={56} w="60%" />
        <SkeletonBlock h={28} w="40%" />
      </div>
      <Divider />
      {/* Range skeleton */}
      <div className="flex flex-col gap-3">
        <SkeletonBlock h={14} w="35%" />
        <SkeletonBlock h={8} />
        <div className="flex justify-between">
          <SkeletonBlock h={12} w="15%" />
          <SkeletonBlock h={12} w="15%" />
          <SkeletonBlock h={12} w="15%" />
        </div>
      </div>
      <Divider />
      {/* SHAP skeleton */}
      <div className="flex flex-col gap-3">
        <SkeletonBlock h={14} w="40%" />
        {[...Array(5)].map((_, i) => (
          <SkeletonBlock key={i} h={28} w={`${85 - i * 8}%`} />
        ))}
      </div>
    </motion.div>
  )
}

function EmptyState() {
  return (
    <motion.div
      key="empty"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-1 flex-col items-center justify-center px-8 py-16"
    >
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.1, duration: 0.5, ease: 'backOut' }}
        className="mb-6 flex h-16 w-16 items-center justify-center rounded-full border border-line bg-surface-2"
      >
        <motion.div
          animate={{ y: [0, -3, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        >
          <Waves size={28} className="text-ink-500" strokeWidth={1.5} />
        </motion.div>
      </motion.div>

      <motion.h3
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-2 font-display text-base font-semibold text-ink-300"
      >
        No prediction yet
      </motion.h3>

      <motion.p
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.28 }}
        className="max-w-[240px] text-center text-sm leading-relaxed text-ink-500"
      >
        Adjust the environmental parameters and run{' '}
        <span className="font-medium text-ink-300">Predict biovolume</span>
      </motion.p>

      {/* Dashed border container */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="mt-8 flex flex-col gap-2 w-full max-w-xs"
      >
        {[...Array(4)].map((_, i) => (
          <div
            key={i}
            className="h-8 rounded-lg"
            style={{
              background: 'rgba(15,23,32,0.015)',
              border: '1px dashed rgba(15,23,32,0.10)',
              opacity: 1 - i * 0.2,
            }}
          />
        ))}
      </motion.div>
    </motion.div>
  )
}

function ResultContent({ result, scale, setScale }: {
  result: PredictionResult
  scale: 'mgL' | 'log'
  setScale: (s: 'mgL' | 'log') => void
}) {
  const heroVal = scale === 'log' ? Math.log1p(result.pred_mgL) : result.pred_mgL
  const heroFmt = scale === 'log' ? (v: number) => v.toFixed(3) : formatMgL
  const heroUnit = scale === 'log' ? 'log' : 'mg/L'

  const sectionVariants = {
    hidden: { opacity: 0, y: 12 },
    show: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: { delay: i * 0.12, duration: 0.45, ease: [0.25, 0.46, 0.45, 0.94] },
    }),
  }

  return (
    <motion.div
      key="result"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col gap-6 px-6 py-6"
    >
      {/* Hero number */}
      <motion.div
        custom={0}
        variants={sectionVariants}
        initial="hidden"
        animate="show"
        className="flex flex-col gap-2"
      >
        <div className="flex items-end justify-between gap-2">
          <div className="flex items-baseline gap-2.5">
            <AnimatedNumber target={heroVal} format={heroFmt} />
            <span className="pb-1 font-mono text-base font-medium text-ink-400">{heroUnit}</span>
          </div>
          {/* scale toggle: raw mg/L vs model log scale */}
          <div className="mb-1 flex overflow-hidden rounded-md border border-line text-[11px]">
            {(['mgL', 'log'] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setScale(s)}
                className={`px-2 py-0.5 font-medium transition-colors ${
                  scale === s ? 'bg-accent-soft text-accent' : 'text-ink-400 hover:text-ink-200'
                }`}
                aria-pressed={scale === s}
                title={s === 'log' ? 'Model log scale, log(1+mg/L)' : 'Biovolume in mg/L'}
              >
                {s === 'mgL' ? 'mg/L' : 'log'}
              </button>
            ))}
          </div>
        </div>
      </motion.div>

      <motion.div custom={1} variants={sectionVariants} initial="hidden" animate="show">
        <Divider />
      </motion.div>

      {/* Prediction range */}
      <motion.div custom={2} variants={sectionVariants} initial="hidden" animate="show">
        <PredictionScale
          lower={result.lower_mgL}
          pred={result.pred_mgL}
          upper={result.upper_mgL}
          group={result.group}
          scale={scale}
        />
      </motion.div>

      <motion.div custom={3} variants={sectionVariants} initial="hidden" animate="show">
        <Divider />
      </motion.div>

      {/* SHAP chart */}
      <motion.div custom={4} variants={sectionVariants} initial="hidden" animate="show">
        <ShapChart entries={result.shap} baseVal={result.base_val} />
      </motion.div>
    </motion.div>
  )
}

export default function ResultPanel({ result, loading }: ResultPanelProps) {
  const [scale, setScale] = useState<'mgL' | 'log'>('mgL')
  return (
    <motion.div
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="surface-card-lg flex flex-col overflow-hidden"
      style={{ minHeight: '500px' }}
    >
      {/* Panel header */}
      <div className="flex flex-shrink-0 items-center gap-3 border-b border-line px-6 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-md border border-accent-line bg-accent-soft">
          <BarChart3 size={15} className="text-accent" strokeWidth={2} />
        </div>
        <div>
          <h2 className="font-display text-[15px] font-semibold text-ink-100">Prediction results</h2>
          <p className="text-[11px] font-medium text-ink-400">
            {result ? `${result.group} biovolume · XGBoost · mg/L` : 'Awaiting input'}
          </p>
        </div>
      </div>

      {/* Content area */}
      <div className="relative flex-1 overflow-auto">
        <AnimatePresence mode="wait">
          {loading ? (
            <LoadingState key="loading" />
          ) : result ? (
            <ResultContent key="result" result={result} scale={scale} setScale={setScale} />
          ) : (
            <EmptyState key="empty" />
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}
