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
   is near-one-sided: the lower bound floors at ~0 (near-absence can't be excluded) while the
   environmental signal acts on the upper bound. The diamond is the estimate (value = hero
   number); lower/upper are the CQR bounds (lower is 0 when the 5th percentile floors). */
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

  const Y = 13 // vertical centre of the bar elements (px)

  return (
    <div ref={ref} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-[13px] font-semibold text-ink-200">Prediction interval</h3>
        <span className="font-mono text-[10px] text-ink-400">90% · {unit}</span>
      </div>

      {/* error bar on a fixed (invisible) axis */}
      <div className="relative w-full" style={{ height: 26 }}>
        {/* faint full-axis baseline */}
        <div className="absolute left-0 right-0 rounded-full bg-surface-3" style={{ top: Y - 1, height: 2 }} />

        {/* 90% interval whisker (lower → upper) */}
        <motion.div
          initial={{ scaleX: 0, opacity: 0 }}
          animate={animated ? { scaleX: 1, opacity: 1 } : {}}
          transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="absolute origin-left rounded-full"
          style={{ left: `${loPct}%`, width: `${Math.max(0.5, hiPct - loPct)}%`, top: Y - 3, height: 6, background: BAND, border: `1px solid ${C}` }}
        />

        {/* lower cap */}
        <div className="absolute -translate-x-1/2" style={{ left: `${loPct}%`, top: Y - 7, width: 2, height: 14, background: C, borderRadius: 1 }} />
        {/* upper cap */}
        {!overflow && (
          <div className="absolute -translate-x-1/2" style={{ left: `${hiPct}%`, top: Y - 7, width: 2, height: 14, background: C, borderRadius: 1 }} />
        )}
        {overflow && (
          <span className="absolute font-mono text-[12px] font-bold" style={{ left: '99%', top: Y - 9, transform: 'translateX(-100%)', color: C }}>›</span>
        )}

        {/* best-estimate diamond (value = hero number) */}
        <motion.div
          initial={{ opacity: 0, scale: 0.4 }}
          animate={animated ? { opacity: 1, scale: 1 } : {}}
          transition={{ delay: 0.4, duration: 0.3 }}
          className="absolute"
          style={{ left: `${prPct}%`, top: Y, width: 11, height: 11, background: C, border: '1.5px solid #FFFFFF', boxShadow: '0 1px 2px rgba(15,23,32,0.18)', transform: 'translate(-50%,-50%) rotate(45deg)', borderRadius: 2 }}
        />
      </div>

      {/* values pinned to markers: 0 at left end, estimate at the diamond, upper at right end */}
      <div className="relative h-7">
        <div className="absolute left-0 flex flex-col items-start">
          <span className="tnum font-mono text-[12px] font-semibold text-ink-300">{fmt(lower)}</span>
          <span className="text-[9px] text-ink-500">lower</span>
        </div>
        <div className="absolute flex flex-col items-start" style={{ left: `${Math.min(prPct, 78)}%` }}>
          <span className="tnum font-mono text-[13px] font-semibold leading-none" style={{ color: C }}>{fmt(pred)}</span>
          <span className="text-[9px] text-ink-500">best estimate</span>
        </div>
        <div className="absolute flex flex-col items-center" style={{ left: `${Math.min(hiPct, 94)}%`, transform: 'translateX(-50%)' }}>
          <span className="tnum font-mono text-[12px] font-semibold text-ink-300">{fmt(upper)}</span>
          <span className="text-[9px] text-ink-500">upper</span>
        </div>
      </div>
    </div>
  )
}
