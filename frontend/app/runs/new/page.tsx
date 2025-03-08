"use client"

import { useRuns } from '@/lib/runs'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'

export default function NewRunPage() {
  const { createNewRun } = useRuns()
  const router = useRouter()

  useEffect(() => {
    // Create run and redirect immediately
    createNewRun()
    router.replace('/runs/active') // Use replace instead of push to prevent back navigation
  }, [])

  return null
} 