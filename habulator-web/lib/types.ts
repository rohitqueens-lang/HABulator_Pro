export interface InputFeatures {
  TEMP: number        // 0–30 °C
  TP: number          // 0.1–200 µg/L
  SI: number          // 0–5 mg/L
  NO23: number        // 0–3 mg/L
  STN_DEPTH_M: number // 1–400 m
  DOY: number         // 1–365
}

export interface ShapEntry {
  feature: string
  value: number
  shap: number
}

export interface PredictionResult {
  pred_mgL: number
  lower_mgL: number
  upper_mgL: number
  base_val: number
  shap: ShapEntry[]
  coverage: number
  group: string
}

export type BloomLevel = 'low' | 'moderate' | 'elevated' | 'high'

export interface GroupConfig {
  id: string
  label: string
  fullName: string
  color: string
  available: boolean
}

export interface FeatureConfig {
  key: keyof InputFeatures
  label: string
  unit: string
  min: number
  max: number
  step: number
  defaultValue: number
  description: string
  iconName: 'thermometer' | 'droplets' | 'flask' | 'zap' | 'eye' | 'anchor' | 'calendar'
}
