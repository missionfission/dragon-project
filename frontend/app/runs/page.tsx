'use client'

import { useRuns, Run } from '@/lib/runs'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useRouter } from 'next/navigation'
import { formatDistanceToNow } from 'date-fns/formatDistanceToNow'

export default function RunsPage() {
  const { runs, switchRun } = useRuns()
  const router = useRouter()

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Run History</h1>
      
      <div className="grid gap-6 md:grid-cols-2">
        {runs.map((run: Run) => (
          <Card key={run.id} className="hover:bg-accent/50 transition-colors">
            <CardHeader>
              <CardTitle className="flex justify-between">
                <span>{run.name}</span>
                <span className="text-sm text-muted-foreground">
                  Created {formatDistanceToNow(new Date(run.createdAt))} ago
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-muted-foreground">
                    Modified {formatDistanceToNow(new Date(run.lastModified))} ago
                  </p>
                  <p className="text-sm">
                    {run.designs.length} designs • {run.systemConfigs.length} system configs
                  </p>
                </div>
                <Button 
                  onClick={() => {
                    switchRun(run.id)
                    router.push('/runs/active')
                  }}
                >
                  Open Run
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
} 