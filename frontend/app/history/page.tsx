'use client'

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { formatDistanceToNow } from "date-fns/formatDistanceToNow"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Plus } from "lucide-react"
import { useRouter } from "next/navigation"
import { useState, useEffect } from "react"

interface HistoryEntry {
  id: string
  timestamp: string
  requirements: {
    powerBudget: number
    areaConstraint: number
    performanceTarget: number
    selectedWorkloads: string[]
    optimizationPriority: string
  }
  result: {
    totalPower: number
    totalArea: number
    estimatedPerformance: number
    powerEfficiency: number
  }
}

export default function HistoryPage() {
  const router = useRouter()
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchHistory() {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/design-history`)
        if (!res.ok) throw new Error('Failed to fetch history')
        const data = await res.json()
        setHistory(data)
      } catch (error) {
        console.error('Error fetching history:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchHistory()
  }, [])

  return (
    <div className="container mx-auto py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Design History</h1>
        <Button 
          onClick={() => router.push('/')}
          className="bg-primary hover:bg-primary/90"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Design
        </Button>
      </div>

      {loading ? (
        <div className="text-center py-8">Loading...</div>
      ) : history.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center">
            <p className="text-muted-foreground">No design runs yet.</p>
            <Button 
              onClick={() => router.push('/')}
              className="mt-4"
            >
              Create Your First Design
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 md:grid-cols-2">
          {history.map((entry) => (
            <Card key={entry.id} className="hover:bg-accent/50 transition-colors">
              <CardHeader>
                <CardTitle className="flex justify-between">
                  <span>Design Run</span>
                  <span className="text-sm text-muted-foreground">
                    {formatDistanceToNow(new Date(entry.timestamp))} ago
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div>
                    <span className="font-medium">Requirements:</span>
                    <ul className="list-disc list-inside">
                      <li>Power Budget: {entry.requirements.powerBudget}W</li>
                      <li>Area Constraint: {entry.requirements.areaConstraint}mm²</li>
                      <li>Performance Target: {entry.requirements.performanceTarget} MIPS</li>
                    </ul>
                  </div>
                  <div>
                    <span className="font-medium">Results:</span>
                    <ul className="list-disc list-inside">
                      <li>Total Power: {entry.result.totalPower.toFixed(2)}W</li>
                      <li>Total Area: {entry.result.totalArea.toFixed(2)}mm²</li>
                      <li>Performance: {entry.result.estimatedPerformance.toFixed(2)} MIPS</li>
                    </ul>
                  </div>
                  <div className="flex gap-2 mt-4">
                    <Button 
                      variant="outline" 
                      onClick={() => router.push(`/history/${entry.id}`)}
                    >
                      View Details
                    </Button>
                    <Button 
                      variant="default"
                      onClick={() => {
                        localStorage.setItem('loadPreviousDesign', JSON.stringify(entry))
                        router.push('/')
                      }}
                    >
                      Load Design
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
} 