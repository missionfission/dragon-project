"use client"

import { createContext, useContext, useState, useEffect } from 'react'

export interface Run {
  id: string
  name: string
  createdAt: string
  lastModified: string
  designs: {
    id: string
    timestamp: string
    requirements: any
    results: any
    config: any
  }[]
  systemConfigs: {
    id: string
    timestamp: string
    config: any
  }[]
}

interface RunContextType {
  activeRun: Run | null
  runs: Run[]
  createNewRun: () => string
  switchRun: (runId: string) => void
  saveToRun: (data: any) => void
  updateRun: (runId: string, updatedRun: Run) => void
}

const RunContext = createContext<RunContextType | null>(null)

export function RunProvider({ children }: { children: React.ReactNode }) {
  const [runs, setRuns] = useState<Run[]>([])
  const [activeRun, setActiveRun] = useState<Run | null>(null)

  useEffect(() => {
    // Load runs from localStorage
    const savedRuns = localStorage.getItem('runs')
    if (savedRuns) {
      const parsedRuns = JSON.parse(savedRuns)
      setRuns(parsedRuns)
      
      // Set active run
      const activeRunId = localStorage.getItem('activeRunId')
      if (activeRunId) {
        const active = parsedRuns.find((r: Run) => r.id === activeRunId)
        if (active) setActiveRun(active)
      }
    }
  }, [])

  const createNewRun = () => {
    const newRun: Run = {
      id: Math.random().toString(36).substr(2, 9),
      name: 'Untitled Run',
      createdAt: new Date().toISOString(),
      lastModified: new Date().toISOString(),
      designs: [],
      systemConfigs: []
    }

    const updatedRuns = [...runs, newRun]
    setRuns(updatedRuns)
    setActiveRun(newRun)
    localStorage.setItem('runs', JSON.stringify(updatedRuns))
    localStorage.setItem('activeRunId', newRun.id)
    return newRun.id
  }

  const switchRun = (runId: string) => {
    const run = runs.find(r => r.id === runId)
    if (run) {
      setActiveRun(run)
      localStorage.setItem('activeRunId', runId)
    }
  }

  const saveToRun = (data: any) => {
    if (!activeRun) return

    const updatedRun = {
      ...activeRun,
      lastModified: new Date().toISOString(),
      designs: [...activeRun.designs, {
        id: Math.random().toString(36).substr(2, 9),
        timestamp: new Date().toISOString(),
        ...data
      }]
    }

    const updatedRuns = runs.map(r => 
      r.id === activeRun.id ? updatedRun : r
    )

    setRuns(updatedRuns)
    setActiveRun(updatedRun)
    localStorage.setItem('runs', JSON.stringify(updatedRuns))
  }

  const updateRun = (runId: string, updatedRun: Run) => {
    const updatedRuns = runs.map(run => 
      run.id === runId ? { ...updatedRun, lastModified: new Date().toISOString() } : run
    )
    
    setRuns(updatedRuns)
    if (activeRun?.id === runId) {
      setActiveRun(updatedRun)
    }
    localStorage.setItem('runs', JSON.stringify(updatedRuns))
  }

  return (
    <RunContext.Provider value={{
      activeRun,
      runs,
      createNewRun,
      switchRun,
      saveToRun,
      updateRun
    }}>
      {children}
    </RunContext.Provider>
  )
}

export function useRuns() {
  const context = useContext(RunContext)
  if (!context) {
    throw new Error('useRuns must be used within a RunProvider')
  }
  return context
} 