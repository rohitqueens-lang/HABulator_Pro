'use client'

import { useCallback } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight, Loader2, SlidersHorizontal, FunctionSquare } from 'lucide-react'
import InputSlider from './InputSlider'
import { FEATURES, FEAT_DISPLAY_LABELS } from '@/lib/utils'
import type { InputFeatures } from '@/lib/types'

interface InputPanelProps {
  values: InputFeatures
  onChange: (key: keyof InputFeatures, value: number) => void
  onPredict: () => void
  loading: boolean
}

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.05, delayChildren: 0.12 } },
}

const itemVariants = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.25, 0.46, 0.45, 0.94] } },
}

export default function InputPanel({ values, onChange, onPredict, loading }: InputPanelProps) {
  const handleChange = useCallback(
    (key: keyof InputFeatures) => (value: number) => onChange(key, value),
    [onChange]
  )

  // Feature engineering applied before the model — shown read-only so the 6 inputs
  // map transparently to the 8 features attributed in the SHAP panel.
  const doyRad = (2 * Math.PI * values.DOY) / 365
  const derived = [
    { label: FEAT_DISPLAY_LABELS.DOY_sin, formula: 'sin(2π·DOY/365)', value: Math.sin(doyRad).toFixed(3) },
    { label: FEAT_DISPLAY_LABELS.DOY_cos, formula: 'cos(2π·DOY/365)', value: Math.cos(doyRad).toFixed(3) },
    { label: FEAT_DISPLAY_LABELS.NP_ratio, formula: 'NO23 / TP', value: (values.TP > 0 ? values.NO23 / values.TP : 0).toFixed(3) },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="surface-card-lg flex flex-col overflow-hidden"
    >
      {/* Panel header */}
      <div className="flex items-center gap-3 border-b border-line px-6 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-md border border-accent-line bg-accent-soft">
          <SlidersHorizontal size={15} className="text-accent" strokeWidth={2} />
        </div>
        <div>
          <h2 className="font-display text-[15px] font-semibold text-ink-100">Input parameters</h2>
          <p className="text-[11px] font-medium text-ink-400">
            {FEATURES.length} inputs → 8 model features
          </p>
        </div>
      </div>

      {/* Sliders */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="show"
        className="flex flex-col gap-5 px-6 py-5"
      >
        {FEATURES.map(({ key, ...feat }) => (
          <motion.div key={key} variants={itemVariants}>
            <InputSlider {...feat} value={values[key]} onChange={handleChange(key)} />
          </motion.div>
        ))}
      </motion.div>

      {/* Derived features — feature engineering, read-only, updates live */}
      <div className="border-t border-line bg-surface-2/40 px-6 py-4">
        <div className="mb-2.5 flex items-center gap-2">
          <FunctionSquare size={13} className="text-ink-400" strokeWidth={2} />
          <h3 className="text-[12px] font-semibold text-ink-300">Derived features</h3>
          <span className="ml-auto text-[10px] text-ink-500">auto-computed · fed to the model</span>
        </div>
        <div className="flex flex-col gap-2">
          {derived.map((d) => (
            <div key={d.label} className="flex items-center justify-between gap-3">
              <span className="text-[11px] font-medium text-ink-300">{d.label}</span>
              <div className="flex items-center gap-2.5">
                <span className="font-mono text-[9.5px] text-ink-500">{d.formula}</span>
                <span className="tnum w-14 text-right font-mono text-[12px] font-semibold text-ink-100">{d.value}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Predict button */}
      <div className="border-t border-line px-6 py-5">
        <button
          onClick={onPredict}
          disabled={loading}
          className="btn-predict focus-ring flex w-full items-center justify-center gap-2 rounded-lg py-3 text-sm font-semibold tracking-wide"
          aria-label={loading ? 'Computing prediction' : 'Run prediction'}
        >
          {loading ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              <span>Computing…</span>
            </>
          ) : (
            <>
              <span>Predict biovolume</span>
              <ArrowRight size={16} strokeWidth={2.5} />
            </>
          )}
        </button>
        <p className="mt-3 text-center font-mono text-[10px] text-ink-500">
          XGBoost · conformal 90% interval · SHAP
        </p>
      </div>
    </motion.div>
  )
}
