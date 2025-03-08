"use client"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { History, Plus } from "lucide-react"
import { useRouter } from "next/navigation"
import { useRuns } from "@/lib/runs"
import { useState, useEffect } from "react"

export function RunHeader() {
  const router = useRouter()
  const { activeRun, updateRun } = useRuns()
  const [runName, setRunName] = useState(activeRun?.name || '')
  const [isEditing, setIsEditing] = useState(false)

  useEffect(() => {
    setRunName(activeRun?.name || '')
  }, [activeRun?.name])

  const handleSave = () => {
    if (activeRun && runName.trim() !== '') {
      updateRun(activeRun.id, { 
        ...activeRun, 
        name: runName.trim() 
      })
    }
    setIsEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave()
    }
    if (e.key === 'Escape') {
      setRunName(activeRun?.name || '')
      setIsEditing(false)
    }
  }

  return (
    <div className="border-b">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-6">
          {isEditing ? (
            <Input
              className="w-64 text-lg font-semibold"
              value={runName}
              onChange={(e) => setRunName(e.target.value)}
              onBlur={handleSave}
              onKeyDown={handleKeyDown}
              autoFocus
            />
          ) : (
            <h2 
              className="text-lg font-semibold cursor-pointer hover:text-gray-600"
              onClick={() => setIsEditing(true)}
            >
              {activeRun ? activeRun.name : 'No Active Run'}
            </h2>
          )}
        </div>
        <div className="flex items-center gap-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => router.push('/runs')}
            className="flex items-center gap-2"
          >
            <History className="w-4 h-4" />
            Run History
          </Button>
          <Button
            size="sm"
            onClick={() => router.push('/runs/new')}
            className="flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Run
          </Button>
        </div>
      </div>
    </div>
  )
} 