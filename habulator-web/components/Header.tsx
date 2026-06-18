'use client'

import { motion } from 'framer-motion'
import { Github, Waves } from 'lucide-react'

export default function Header() {
  return (
    <motion.header
      initial={{ y: -48, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="sticky top-0 z-50 w-full border-b border-line bg-base/95"
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Wordmark */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-line bg-surface-2">
            <Waves size={17} className="text-accent" strokeWidth={2} />
          </div>
          <div className="flex flex-col leading-none">
            <span className="font-display text-[19px] font-semibold tracking-tight text-ink-100">
              Habulator
            </span>
            <span className="mt-1 text-[11px] font-medium text-ink-400">
              Great Lakes phytoplankton predictor
            </span>
          </div>
        </div>

        {/* Right meta */}
        <div className="flex items-center gap-2.5">
          <div className="hidden items-center rounded-md border border-line bg-surface-1 px-2.5 py-1.5 sm:flex">
            <span className="tnum font-mono text-[11px] text-ink-300">
              EPA GLNPO · 2001–2021
            </span>
          </div>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex h-9 w-9 items-center justify-center rounded-md border border-line bg-surface-1 text-ink-400 transition-colors hover:text-ink-100 hover:border-line-strong"
            aria-label="View source on GitHub"
          >
            <Github size={16} />
          </a>
        </div>
      </div>
    </motion.header>
  )
}
