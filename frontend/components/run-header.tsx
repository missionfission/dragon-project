"use client"

import { Button } from "@/components/ui/button"
import { History, Plus } from "lucide-react"
import { useRouter } from "next/navigation"
import { useRuns } from "@/lib/runs"

export function RunHeader() {
  const router = useRouter()
  const { activeRun } = useRuns()

  return (
    <div className="border-b">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-6">
          <h2 className="text-lg font-semibold">
            {activeRun ? activeRun.name : 'No Active Run'}
          </h2>
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