import type { Metadata, Viewport } from 'next'
import { IBM_Plex_Sans, IBM_Plex_Mono, Space_Grotesk } from 'next/font/google'
import './globals.css'

// Body / UI — serious, scientific, highly readable
const plexSans = IBM_Plex_Sans({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-sans',
  weight: ['400', '500', '600', '700'],
})

// Numerals & data — tabular monospace (a scientific-instrument cue)
const plexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-mono',
  weight: ['400', '500', '600'],
})

// Display / wordmark — distinct from the body face
const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-display',
  weight: ['500', '600', '700'],
})

export const metadata: Metadata = {
  title: 'Habulator — Great Lakes Phytoplankton Predictor',
  description:
    'Predicts phytoplankton biovolume for the five Laurentian Great Lakes from water-chemistry inputs. ' +
    'XGBoost emulator with conformalized prediction intervals and SHAP attribution. Data: U.S. EPA GLNPO, 2001–2021.',
  keywords: [
    'Great Lakes', 'phytoplankton', 'biovolume', 'prediction',
    'machine learning', 'XGBoost', 'SHAP', 'conformal prediction', 'water quality',
  ],
  openGraph: {
    title: 'Habulator — Great Lakes Phytoplankton Predictor',
    description: 'XGBoost emulator of Great Lakes phytoplankton biovolume with calibrated uncertainty.',
    type: 'website',
  },
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#FFFFFF',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${plexSans.variable} ${plexMono.variable} ${spaceGrotesk.variable}`}
    >
      <body className="font-sans text-ink-100 antialiased">{children}</body>
    </html>
  )
}
