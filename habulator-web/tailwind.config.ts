import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './lib/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // ── Base + layered surfaces (depth via levels, not glow) ──
        base: '#FFFFFF',
        surface: {
          1: '#FFFFFF',
          2: '#F4F6F8',
          3: '#EAEEF2',
        },
        line: {
          DEFAULT: 'rgba(15,23,32,0.10)',
          strong: 'rgba(15,23,32,0.18)',
        },
        // ── Neutral text ramp (dark text on light) ──
        ink: {
          100: '#0F1419',
          200: '#283039',
          300: '#4C5663',
          400: '#6B7480',
          500: '#9AA3AE',
        },
        // ── Single brand accent (water teal) — deepened for contrast on white ──
        accent: {
          DEFAULT: '#0D9488',
          strong: '#0F766E',
          soft: 'rgba(13,148,136,0.10)',
          line: 'rgba(13,148,136,0.28)',
        },
        // ── Meaningful data color (bloom severity) — semantic, kept ──
        bloom: {
          low: '#34D399',
          moderate: '#FBBF24',
          elevated: '#FB923C',
          high: '#F87171',
        },
        // ── SHAP diverging (kept; calmer than neon) ──
        shap: {
          positive: '#FB923C',
          negative: '#38BDF8',
        },
      },
      fontFamily: {
        display: ['var(--font-display)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        sans: ['var(--font-sans)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      borderRadius: {
        card: '12px',
      },
      boxShadow: {
        // Soft neutral elevation for a light surface
        e1: '0 1px 2px rgba(15,23,32,0.06)',
        e2: '0 2px 8px rgba(15,23,32,0.08)',
        e3: '0 10px 30px rgba(15,23,32,0.10)',
      },
    },
  },
  plugins: [],
}

export default config
