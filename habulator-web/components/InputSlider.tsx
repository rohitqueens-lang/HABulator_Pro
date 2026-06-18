'use client'

import { useState, useCallback, useId } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Thermometer, Droplets, FlaskConical, Zap, Eye, Anchor, Calendar, Info,
} from 'lucide-react'
import { clamp } from '@/lib/utils'
import type { FeatureConfig } from '@/lib/types'

const ICON_MAP = {
  thermometer: Thermometer, droplets: Droplets, flask: FlaskConical, zap: Zap,
  eye: Eye, anchor: Anchor, calendar: Calendar,
} as const

interface InputSliderProps extends Omit<FeatureConfig, 'key'> {
  value: number
  onChange: (value: number) => void
}

export default function InputSlider({
  label, iconName, unit, min, max, step, value, onChange, description,
}: InputSliderProps) {
  const [inputStr, setInputStr] = useState<string | null>(null)
  const [showTooltip, setShowTooltip] = useState(false)
  const [focused, setFocused] = useState(false)
  const id = useId()

  const Icon = ICON_MAP[iconName]
  const pct = ((value - min) / (max - min)) * 100
  const isOutOfRange = value < min || value > max
  const displayStr = inputStr !== null ? inputStr : String(value)

  const handleSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value)); setInputStr(null)
  }, [onChange])

  const handleNumberInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value; setInputStr(raw)
    const v = parseFloat(raw); if (!isNaN(v)) onChange(clamp(v, min, max))
  }, [onChange, min, max])

  const handleNumberBlur = useCallback(() => {
    setInputStr(null); setFocused(false)
    const v = parseFloat(displayStr)
    onChange(!isNaN(v) ? clamp(v, min, max) : value)
  }, [displayStr, onChange, min, max, value])

  return (
    <div className="group flex flex-col gap-2">
      {/* Label row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-md border border-line bg-surface-2">
            <Icon size={13} className="text-ink-300" strokeWidth={2} />
          </div>
          <label htmlFor={id} className="cursor-pointer text-[13px] font-medium text-ink-200">
            {label}
          </label>
          <div className="relative">
            <button
              type="button"
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              onFocus={() => setShowTooltip(true)}
              onBlur={() => setShowTooltip(false)}
              className="flex items-center text-ink-500 transition-colors hover:text-ink-300 focus:outline-none"
              aria-label={`Info about ${label}`}
            >
              <Info size={12} />
            </button>
            <AnimatePresence>
              {showTooltip && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 4 }}
                  transition={{ duration: 0.13 }}
                  className="absolute left-0 top-6 z-50 w-56 rounded-lg border border-line bg-surface-3 px-3 py-2 text-xs leading-relaxed text-ink-300 shadow-e3"
                >
                  {description}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Number input + unit */}
        <div
          className={[
            'flex items-center gap-1.5 rounded-md border px-2 py-1 transition-colors',
            isOutOfRange ? 'border-bloom-high' : focused ? 'border-accent-line bg-accent-soft' : 'border-line bg-surface-2',
          ].join(' ')}
        >
          <input
            id={id}
            type="number"
            value={displayStr}
            min={min}
            max={max}
            step={step}
            onChange={handleNumberInput}
            onFocus={() => setFocused(true)}
            onBlur={handleNumberBlur}
            className="tnum w-16 bg-transparent text-right font-mono text-[13px] font-medium text-ink-100 outline-none"
            aria-label={`${label} value`}
          />
          <span className="select-none text-[11px] font-medium text-ink-400">{unit}</span>
        </div>
      </div>

      {/* Slider row */}
      <div className="relative flex flex-col gap-1">
        <div className="relative flex h-4 items-center">
          <div
            className="pointer-events-none absolute left-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-accent"
            style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
          />
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={handleSlider}
            className="relative z-10 w-full"
            aria-label={`${label} slider`}
          />
        </div>
        <div className="flex justify-between">
          <span className="tnum font-mono text-[10px] text-ink-500">{min} {unit}</span>
          <span className="tnum font-mono text-[10px] text-ink-500">{max} {unit}</span>
        </div>
      </div>

      <AnimatePresence>
        {isOutOfRange && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="text-xs text-bloom-high"
          >
            Value must be between {min} and {max} {unit}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  )
}
