'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Info, ChevronDown, ChevronUp } from 'lucide-react'
import { formatShap, FEAT_DISPLAY_LABELS, FEAT_SIMPLE_LABELS, FEAT_UNITS } from '@/lib/utils'
import type { ShapEntry } from '@/lib/types'

interface ShapChartProps {
  entries: ShapEntry[]
  baseVal: number
}

function InfoTooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative inline-flex">
      <button
        type="button"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onFocus={() => setShow(true)}
        onBlur={() => setShow(false)}
        className="text-ink-500 transition-colors hover:text-ink-300 focus:outline-none"
        aria-label="What is SHAP?"
      >
        <Info size={13} />
      </button>
      <AnimatePresence>
        {show && (
          <motion.div
            initial={{ opacity: 0, y: 4, scale: 0.92 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.92 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 top-6 z-50 w-64 rounded-lg border border-line bg-surface-3 px-3 py-2.5 text-xs leading-relaxed text-ink-300 shadow-e3"
          >
            <span className="font-semibold text-ink-100">SHAP</span> (SHapley
            Additive exPlanations) shows how each feature pushed the prediction up{' '}
            <span style={{ color: '#FB923C' }}>▲</span> or down{' '}
            <span style={{ color: '#38BDF8' }}>▼</span>, in additive log-scale units,
            consistent with the paper.
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

interface ShapBarProps {
  entry: ShapEntry
  maxAbs: number    // max |SHAP| on the additive log scale, for bar scaling
  index: number
  simple: boolean
}

function ShapBar({ entry, maxAbs, index, simple }: ShapBarProps) {
  const isPositive = entry.shap >= 0
  const ratio = maxAbs > 0 ? Math.abs(entry.shap) / maxAbs : 0
  const barPct = ratio * 45 // 45% of half-width
  const color = isPositive ? '#FB923C' : '#38BDF8'
  const label = simple
    ? FEAT_SIMPLE_LABELS[entry.feature] ?? entry.feature
    : FEAT_DISPLAY_LABELS[entry.feature] ?? entry.feature
  const unit = FEAT_UNITS[entry.feature] ?? ''
  const valStr = typeof entry.value === 'number'
    ? unit
      ? `${entry.value.toPrecision(3)} ${unit}`
      : entry.value.toPrecision(3)
    : String(entry.value)

  return (
    <motion.div
      initial={{ opacity: 0, x: isPositive ? -8 : 8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05, duration: 0.35, ease: 'easeOut' }}
      className="flex items-center gap-0 w-full"
      style={{ minHeight: '36px' }}
    >
      {/* Feature name — left half */}
      <div className="flex flex-col justify-center" style={{ width: '40%', paddingRight: '12px' }}>
        <span className="truncate text-xs font-medium leading-tight text-ink-200">{label}</span>
        {!simple && (
          <span className="tnum truncate font-mono text-[10px] leading-tight text-ink-500">
            {valStr}
          </span>
        )}
      </div>

      {/* Chart — right 60% with center zero */}
      <div className="relative flex items-center" style={{ width: '60%', height: '28px' }}>
        {/* Center zero line */}
        <div
          className="absolute top-0 bottom-0 z-10"
          style={{
            left: '50%',
            width: '1px',
            background: 'rgba(15,23,32,0.14)',
          }}
        />

        {/* Bar */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${barPct}%` }}
          transition={{ delay: 0.1 + index * 0.05, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="absolute top-1/2 h-5 -translate-y-1/2 rounded-[3px]"
          style={{
            [isPositive ? 'left' : 'right']: '50%',
            background: color,
            opacity: 0.9,
          }}
        />

        {/* SHAP value label */}
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 + index * 0.05 }}
          className="absolute text-[10px] font-mono font-semibold whitespace-nowrap"
          style={{
            [isPositive ? 'left' : 'right']: `calc(50% + ${barPct}% + 4px)`,
            color,
          }}
        >
          {simple
            ? (ratio > 0.6 ? 'High' : ratio > 0.25 ? 'Mid' : 'Low')
            : `${isPositive ? '+' : ''}${formatShap(entry.shap)}`}
        </motion.span>
      </div>
    </motion.div>
  )
}

export default function ShapChart({ entries, baseVal }: ShapChartProps) {
  const [simple, setSimple] = useState(false)

  // SHAP on the additive log scale — exactly additive and identical to the manuscript figure.
  const sorted = useMemo(
    () => [...entries].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)),
    [entries]
  )
  const displayed = simple ? sorted.slice(0, 4) : sorted
  const maxAbs = Math.max(...sorted.map((e) => Math.abs(e.shap)), 1e-4)

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-[13px] font-semibold text-ink-200">Why this prediction?</h3>
          <span className="text-[10px] font-medium text-ink-500">log scale</span>
          <InfoTooltip text="" />
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-3.5 rounded-[2px]" style={{ background: '#FB923C' }} />
            <span className="text-[10px] font-medium text-ink-400">Increases</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-3.5 rounded-[2px]" style={{ background: '#38BDF8' }} />
            <span className="text-[10px] font-medium text-ink-400">Decreases</span>
          </div>
        </div>
      </div>

      {/* Bars */}
      <div className="inset-card flex flex-col divide-y divide-line px-4 py-3">
        <AnimatePresence mode="popLayout">
          {displayed.map((entry, i) => (
            <div
              key={entry.feature}
              style={{ paddingTop: i === 0 ? 0 : '4px', paddingBottom: '4px' }}
            >
              <ShapBar
                entry={entry}
                maxAbs={maxAbs}
                index={i}
                simple={simple}
              />
            </div>
          ))}
        </AnimatePresence>
      </div>

      {/* Baseline note */}
      {!simple && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="text-center text-[10px] font-medium text-ink-500"
        >
          Model baseline (log scale):{' '}
          <span className="tnum font-mono text-ink-300">{baseVal.toFixed(3)}</span>
        </motion.p>
      )}

      {/* Simple / Technical toggle */}
      <div className="flex justify-center">
        <button
          type="button"
          onClick={() => setSimple((s) => !s)}
          className="flex items-center gap-1.5 rounded-md border border-line bg-surface-2 px-3 py-1.5 text-xs font-medium text-ink-400 transition-colors hover:text-ink-200"
          aria-pressed={simple}
        >
          {simple ? (
            <>
              <ChevronDown size={12} />
              Technical view
            </>
          ) : (
            <>
              <ChevronUp size={12} />
              Simple view
            </>
          )}
        </button>
      </div>
    </div>
  )
}
