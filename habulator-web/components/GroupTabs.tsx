'use client'

import { motion } from 'framer-motion'
import { Lock } from 'lucide-react'
import type { GroupConfig } from '@/lib/types'
import { GROUPS } from '@/lib/utils'

interface GroupTabsProps {
  activeGroup: string
  onGroupChange: (id: string) => void
}

function GroupTab({
  group,
  isActive,
  onSelect,
}: {
  group: GroupConfig
  isActive: boolean
  onSelect: () => void
}) {
  return (
    <button
      onClick={group.available ? onSelect : undefined}
      disabled={!group.available}
      className={[
        'relative flex flex-col items-center gap-0.5 rounded-md px-3.5 py-2 transition-colors',
        group.available ? 'cursor-pointer' : 'cursor-not-allowed opacity-40',
        isActive ? 'text-ink-100' : 'text-ink-400 hover:text-ink-200',
      ].join(' ')}
      aria-label={group.fullName}
      aria-selected={isActive}
      role="tab"
    >
      {isActive && (
        <motion.div
          layoutId="activeTabBg"
          className="absolute inset-0 rounded-md border border-accent-line bg-accent-soft"
          transition={{ type: 'spring', bounce: 0.15, duration: 0.45 }}
        />
      )}
      <div className="relative flex items-center gap-1.5">
        <span
          className="h-2 w-2 flex-shrink-0 rounded-full transition-opacity"
          style={{ background: group.color, opacity: isActive ? 1 : 0.5 }}
        />
        <span className="font-display text-[13px] font-semibold tracking-wide">
          {group.label}
        </span>
        {!group.available && <Lock size={10} className="opacity-60" />}
      </div>
      <span
        className={[
          'relative text-[10px] font-medium leading-none whitespace-nowrap',
          isActive ? 'text-accent' : 'text-ink-500',
        ].join(' ')}
      >
        {group.fullName}
      </span>
    </button>
  )
}

export default function GroupTabs({ activeGroup, onGroupChange }: GroupTabsProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: 0.08 }}
      role="tablist"
      aria-label="Phytoplankton group selector"
      className="flex items-center gap-1 rounded-lg border border-line bg-surface-1 p-1"
    >
      {GROUPS.map((group) => (
        <GroupTab
          key={group.id}
          group={group}
          isActive={activeGroup === group.id}
          onSelect={() => onGroupChange(group.id)}
        />
      ))}
    </motion.div>
  )
}
