import type { GroupConfig, FeatureConfig } from './types'

// ─── Number formatting ────────────────────────────────────────────────────────
export function formatMgL(v: number): string {
  if (v <= 0) return '0'
  if (v >= 100) return v.toFixed(0)
  if (v >= 10) return v.toFixed(1)
  if (v >= 1) return v.toFixed(2)
  if (v >= 0.01) return v.toFixed(3)
  return v.toExponential(2)
}

export function formatShap(v: number): string {
  const abs = Math.abs(v)
  if (abs >= 10) return v.toFixed(1)
  if (abs >= 1) return v.toFixed(2)
  return v.toFixed(3)
}

export function clamp(val: number, min: number, max: number): number {
  return Math.min(Math.max(val, min), max)
}

// ─── Group configs ────────────────────────────────────────────────────────────
// Okabe–Ito colorblind-safe palette — IDENTICAL to the manuscript figures
// (Fig 1, performance, uncertainty, SHAP). Do not diverge from these hexes.
export const GROUPS: GroupConfig[] = [
  {
    id: 'EDIAT',
    label: 'EDIAT',
    fullName: 'Early diatoms',
    color: '#0072B2',
    available: true,
  },
  {
    id: 'LDIAT',
    label: 'LDIAT',
    fullName: 'Late diatoms',
    color: '#E69F00',
    available: true,
  },
  {
    id: 'CHLOR',
    label: 'CHLOR',
    fullName: 'Chlorophytes',
    color: '#009E73',
    available: true,
  },
  {
    id: 'CRYPT',
    label: 'CRYPT',
    fullName: 'Cryptophytes',
    color: '#CC79A7',
    available: true,
  },
  {
    id: 'CYANO',
    label: 'CYANO',
    fullName: 'Cyanobacteria',
    color: '#D55E00',
    available: true,
  },
]

// ─── Held-out uncertainty stats (single-split seed 42, CQR test set) ──────────
// All values measured on the 15% held-out test set, methodology identical to
// make_uncertainty_main.py (quantiles on 70% fit, conformal Q on 15% calib).
//   coverage      = empirical MARGINAL 90% interval coverage (= coverage_test)
//   meanWidth     = mean 90% interval width, mg/L (= mean_width_test)
//   highThreshold = predicted-value 80th percentile, mg/L (top-quintile edge)
//   q5Coverage    = empirical coverage WITHIN the top predicted-value quintile.
// Above highThreshold, marginal coverage no longer applies; q5Coverage is the
// honest per-group conditional coverage there (computed per group, not borrowed).
export const UQ_STATS: Record<
  string,
  { coverage: number; meanWidth: number; highThreshold: number; q5Coverage: number }
> = {
  EDIAT: { coverage: 0.948, meanWidth: 2.238, highThreshold: 0.276, q5Coverage: 0.843 },
  LDIAT: { coverage: 0.920, meanWidth: 0.350, highThreshold: 0.130, q5Coverage: 0.786 },
  CHLOR: { coverage: 0.931, meanWidth: 1.120, highThreshold: 0.295, q5Coverage: 0.857 },
  CRYPT: { coverage: 0.914, meanWidth: 0.778, highThreshold: 0.373, q5Coverage: 0.929 },
  CYANO: { coverage: 0.922, meanWidth: 1.083, highThreshold: 0.209, q5Coverage: 0.886 },
}

// ─── Observed biovolume distribution per group (mg/L, n=2316) ─────────────────
// Empirical percentiles of the observed targets in habulator_master_v2.csv.
// Used as the data-grounded reference scale behind each prediction (median, IQR,
// 5–95%). Nothing invented — these are the raw observed quantiles.
export const OBS_STATS: Record<
  string,
  { p5: number; p25: number; median: number; p75: number; p95: number; n: number }
> = {
  EDIAT: { p5: 0.00441, p25: 0.01824, median: 0.04788, p75: 0.16716, p95: 4.6816, n: 2316 },
  LDIAT: { p5: 0.0, p25: 0.00498, median: 0.01776, p75: 0.06234, p95: 0.4496, n: 2316 },
  CHLOR: { p5: 0.0, p25: 0.0041, median: 0.03281, p75: 0.18768, p95: 2.0246, n: 2316 },
  CRYPT: { p5: 0.01069, p25: 0.05479, median: 0.12825, p75: 0.29835, p95: 1.0048, n: 2316 },
  CYANO: { p5: 0.00278, p25: 0.01194, median: 0.03565, p75: 0.13568, p95: 1.6473, n: 2316 },
}

// ─── Fixed per-group display axis for the interval gauge (mg/L) ───────────────
// axMax = ~99th percentile of the model's 90% UPPER bounds across all 2316 inputs
// (also covers an extreme high-input corner; the rare >p99 bloom overflows the axis
// and is flagged). This gives the interval gauge a STABLE frame so a small prediction
// sits left and a large one sits right — comparable across inputs, not self-rescaled.
// Geometry is log1p(mg/L); lower bound (often 0) maps to the origin cleanly.
export const AXIS_MAX_MGL: Record<string, number> = {
  EDIAT: 26,
  LDIAT: 4,
  CHLOR: 13,
  CRYPT: 5,
  CYANO: 14,
}

// ─── Per-group predictive skill (headline metric) ─────────────────────────────
// r2 = station-grouped 5-fold cross-validated R²(log) = Nash–Sutcliffe efficiency
//   on log(1+biovolume), the manuscript's headline skill metric (Track A).
//   Values copied verbatim from {group}_metrics.json → skill_track.cv_r2_log.
export const SKILL_STATS: Record<string, { r2: number }> = {
  EDIAT: { r2: 0.677 },
  LDIAT: { r2: 0.246 },
  CHLOR: { r2: 0.450 },
  CRYPT: { r2: 0.345 },
  CYANO: { r2: 0.575 },
}

// Map a CV R²(log) to an interpretable tier + color (used by the skill badge).
export function getSkillTier(r2: number): { label: string; color: string } {
  if (r2 >= 0.65) return { label: 'Strong', color: '#15803D' }   // green
  if (r2 >= 0.50) return { label: 'Good', color: '#0D9488' }     // teal (accent)
  if (r2 >= 0.35) return { label: 'Moderate', color: '#B45309' } // amber
  return { label: 'Limited', color: '#C2410C' }                  // orange
}

// ─── Feature configs ──────────────────────────────────────────────────────────
export const FEATURES: FeatureConfig[] = [
  {
    key: 'TEMP',
    label: 'Temperature',
    unit: '°C',
    min: 0,
    max: 30,
    step: 0.1,
    defaultValue: 12,
    description: 'Surface water temperature. Warmer temps accelerate phytoplankton metabolism.',
    iconName: 'thermometer',
  },
  {
    key: 'TP',
    label: 'Total Phosphorus',
    unit: 'µg/L',
    min: 0.1,
    max: 200,
    step: 0.1,
    defaultValue: 8,
    description: 'Total phosphorus concentration. Primary limiting nutrient for algal growth.',
    iconName: 'droplets',
  },
  {
    key: 'SI',
    label: 'Silica (Si)',
    unit: 'mg/L',
    min: 0,
    max: 5,
    step: 0.01,
    defaultValue: 1.2,
    description: 'Dissolved silica. Essential for diatom cell wall (frustule) construction.',
    iconName: 'flask',
  },
  {
    key: 'NO23',
    label: 'Nitrate + Nitrite',
    unit: 'mg/L',
    min: 0,
    max: 3,
    step: 0.01,
    defaultValue: 0.4,
    description: 'Dissolved inorganic nitrogen. Secondary macronutrient for algal growth.',
    iconName: 'zap',
  },
  {
    key: 'STN_DEPTH_M',
    label: 'Station Depth',
    unit: 'm',
    min: 1,
    max: 400,
    step: 1,
    defaultValue: 85,
    description: 'Maximum depth at sampling station. Deeper sites tend to be more offshore and oligotrophic.',
    iconName: 'anchor',
  },
  {
    key: 'DOY',
    label: 'Day of Year',
    unit: 'DOY',
    min: 1,
    max: 365,
    step: 1,
    defaultValue: 200,
    description: 'Day of year (1–365). Captures seasonal patterns in phytoplankton dynamics.',
    iconName: 'calendar',
  },
]

// ─── Feature labels for SHAP display ─────────────────────────────────────────
export const FEAT_DISPLAY_LABELS: Record<string, string> = {
  TEMP: 'Temperature',
  TP: 'Total Phosphorus',
  SI: 'Silica',
  NO23: 'Nitrate + Nitrite',
  STN_DEPTH_M: 'Station Depth',
  DOY_sin: 'Seasonality (sin)',
  DOY_cos: 'Seasonality (cos)',
  NP_ratio: 'N:P Ratio',
}

export const FEAT_SIMPLE_LABELS: Record<string, string> = {
  TEMP: 'Water temperature',
  TP: 'Phosphorus availability',
  SI: 'Silica availability',
  NO23: 'Nitrogen availability',
  STN_DEPTH_M: 'Station depth',
  DOY_sin: 'Time of year (cycle)',
  DOY_cos: 'Time of year (phase)',
  NP_ratio: 'Nutrient balance',
}

export const FEAT_UNITS: Record<string, string> = {
  TEMP: '°C',
  TP: 'µg/L',
  SI: 'mg/L',
  NO23: 'mg/L',
  STN_DEPTH_M: 'm',
  DOY_sin: '',
  DOY_cos: '',
  NP_ratio: '',
}
