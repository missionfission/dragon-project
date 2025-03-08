"use client"

import { useRuns } from '@/lib/runs'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'

export default function NewRunPage() {
  const { createNewRun } = useRuns()
  const router = useRouter()

  useEffect(() => {
    async function createRun() {
      const runId = await createNewRun()
      router.push('/runs/active')
    }
    createRun()
  }, [createNewRun, router])

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <h2 className="text-xl font-semibold mb-2">Creating new run...</h2>
        <p className="text-muted-foreground">Please wait</p>
      </div>
    </div>
  )
} 