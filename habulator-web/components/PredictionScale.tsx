'use client'

import { useRef, useEffect, useState } from 'react'
import { motion, useInView } from 'framer-motion'
import { formatMgL, AXIS_MAX_MGL } from '@/lib/utils'

interface PredictionScaleProps {
  lower: number
  pred: number
  upper: number
  group: string
  scale?: 'mgL' | 'log'
}

const C = '#4F46E5'                 // indigo (change here to recolor)
const BAND = 'rgba(79,70,229,0.16)'

/* Point estimate + asymmetric 90% prediction interval (CQR), drawn as an error bar on a
   FIXED per-group log1p axis. For these zero-bounded, right-skewed biovolumes the interval
   is near-one-sided: the lower bound floors at ~0 while the environmental signal acts on the
   upper bound. Labels are anchored to their marks — the estimate sits above the diamond, the
   bounds sit under their caps — and a faint mg/L scale shows where the interval falls on the
   group's full range. */
export default function PredictionScale({ lower, pred, upper, group, scale = 'mgL' }: PredictionScaleProps) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true })
  const [animated, setAnimated] = useState(false)
  useEffect(() => {
    if (inView) {
      const t = setTimeout(() => setAnimated(true), 80)
      return () => clearTimeout(t)
    }
  }, [inView])

  const axMax = AXIS_MAX_MGL[group] ?? Math.max(upper * 1.3, 1)
  const Lmax = Math.log1p(axMax)
  const pos = (v: number) => Math.min(100, Math.max(0, (Math.log1p(Math.max(0, v)) / Lmax) * 100))
  const loPct = pos(lower), prPct = pos(pred), hiPct = pos(upper)
  const overflow = upper > axMax

  const fmt = scale === 'log' ? (v: number) => Math.log1p(Math.max(0, v)).toFixed(3) : formatMgL
  const unit = scale === 'log' ? 'log' : 'mg/L'

  // faint reference ticks (mg/L) within the axis, always including the axis max
  const ticks = [...[0.1, 1, 10, 100].filter((v) => v < axMax), axMax]

  // edge-aware horizontal anchor for a label centred on position p (%)
  const anchor = (p: number): { left: string; transform: string; textAlign: 'left' | 'center' | 'right' } => {
    if (p < 8) return { left: `${p}%`, transform: 'translateX(0)', textAlign: 'left' }
    if (p > 92) return { left: `${p}%`, transform: 'translateX(-100%)', textAlign: 'right' }
    return { left: `${p}%`, transform: 'translateX(-50%)', textAlign: 'center' }
  }
  const loA = anchor(loPct)
  const hiA = anchor(hiPct)
  const bestLeft = Math.min(93, Math.max(7, prPct))

  const BY = 40 // vertical centre of the bar (px)

  return (
    <div ref={ref} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-[13px] font-semibold text-ink-200">Prediction interval</h3>
        <span className="font-mono text-[10px] text-ink-400">90% · {unit}</span>
      </div>

      <div className="relative w-full" style={{ height: 92 }}>
        {/* best estimate — above the diamond */}
        <div className="absolute flex flex-col items-center" style={{ left: `${bestLeft}%`, top: 0, transform: 'translateX(-50%)' }}>
          <span className="tnum font-mono text-[13px] font-semibold leading-none" style={{ color: C }}>{fmt(pred)}</span>
          <span className="text-[9px] text-ink-500 mt-0.5">best estimate</span>
        </div>

        {/* faint full-axis baseline */}
        <div className="absolute left-0 right-0 rounded-full bg-surface-3" style={{ top: BY - 1, height: 2 }} />

        {/* 90% interval whisker (lower → upper) */}
        <motion.div
          initial={{ scaleX: 0, opacity: 0 }}
          animate={animated ? { scaleX: 1, opacity: 1 } : {}}
          transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="absolute origin-left rounded-full"
          style={{ left: `${loPct}%`, width: `${Math.max(0.5, hiPct - loPct)}%`, top: BY - 3, height: 6, background: BAND, border: `1px solid ${C}` }}
        />

        {/* caps */}
        <div className="absolute -translate-x-1/2" style={{ left: `${loPct}%`, top: BY - 7, width: 2, height: 14, background: C, borderRadius: 1 }} />
        {!overflow && (
          <div className="absolute -translate-x-1/2" style={{ left: `${hiPct}%`, top: BY - 7, width: 2, height: 14, background: C, borderRadius: 1 }} />
        )}
        {overflow && (
          <span className="absolute font-mono text-[12px] font-bold" style={{ left: '99%', top: BY - 9, transform: 'translateX(-100%)', color: C }}>›</span>
        )}

        {/* best-estimate diamond */}
        <motion.div
          initial={{ opacity: 0, scale: 0.4 }}
          animate={animated ? { opacity: 1, scale: 1 } : {}}
          transition={{ delay: 0.4, duration: 0.3 }}
          className="absolute"
          style={{ left: `${prPct}%`, top: BY, width: 11, height: 11, background: C, border: '1.5px solid #FFFFFF', boxShadow: '0 1px 2px rgba(15,23,32,0.18)', transform: 'translate(-50%,-50%) rotate(45deg)', borderRadius: 2 }}
        />

        {/* faint mg/L reference scale */}
        {ticks.map((v) => {
          const tp = pos(v)
          return (
            <div key={v} className="absolute" style={{ left: `${tp}%`, top: BY + 9 }}>
              <div style={{ width: 1, height: 4, background: 'var(--line-strong, #c2cbd4)', transform: 'translateX(-50%)' }} />
              <span className="absolute text-[8px] text-ink-500 whitespace-nowrap"
                    style={{ top: 5, ...(tp > 92 ? { right: 0 } : tp < 8 ? { left: 0 } : { left: '50%', transform: 'translateX(-50%)' }) }}>
                {fmt(v)}
              </span>
            </div>
          )
        })}

        {/* bound labels — under their caps */}
        <div className="absolute flex flex-col" style={{ ...loA, top: 64, alignItems: loA.textAlign === 'right' ? 'flex-end' : loA.textAlign === 'center' ? 'center' : 'flex-start' }}>
          <span className="tnum font-mono text-[12px] font-semibold text-ink-300">{fmt(lower)}</span>
          <span className="text-[9px] text-ink-500">lower</span>
        </div>
        <div className="absolute flex flex-col" style={{ ...hiA, top: 64, alignItems: hiA.textAlign === 'right' ? 'flex-end' : hiA.textAlign === 'center' ? 'center' : 'flex-start' }}>
          <span className="tnum font-mono text-[12px] font-semibold text-ink-300">{fmt(upper)}</span>
          <span className="text-[9px] text-ink-500">upper</span>
        </div>
      </div>
    </div>
  )
}
