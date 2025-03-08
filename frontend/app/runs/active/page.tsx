"use client"

import { useRuns } from '@/lib/runs'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { ChipDesigner } from '@/components/chip-designer'

export default function ActiveRunPage() {
  const { activeRun } = useRuns()
  const router = useRouter()

  useEffect(() => {
    if (!activeRun) {
      router.push('/runs')
    }
  }, [activeRun, router])

  if (!activeRun) return null

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">
        {activeRun.name}
      </h1>
      <ChipDesigner />
    </div>
  )
} 