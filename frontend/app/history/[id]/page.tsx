import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { notFound } from "next/navigation"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"

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
    blocks: Array<{
      id: string
      type: string
      size: { width: number; height: number }
      position: { x: number; y: number }
      powerConsumption: number
      performance: number
      utilization: number
    }>
    totalPower: number
    totalArea: number
    estimatedPerformance: number
    powerEfficiency: number
  }
  optimization_data: {
    graph: any
    animation_frames: string[]
    performance_estimation_frames: string[]
  }
}

async function getHistoryEntry(id: string): Promise<HistoryEntry> {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/design-history/${id}`)
  if (!res.ok) {
    if (res.status === 404) notFound()
    throw new Error('Failed to fetch history entry')
  }
  return res.json()
}

export default async function HistoryDetailPage({ params }: { params: { id: string } }) {
  const router = useRouter()
  const entry = await getHistoryEntry(params.id)

  const handleLoadDesign = () => {
    // Store the design data in localStorage and navigate to design page
    localStorage.setItem('loadPreviousDesign', JSON.stringify(entry))
    router.push('/')
  }

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Design Run Details</h1>
      
      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Requirements</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <h3 className="font-medium">Specifications</h3>
                <ul className="list-disc list-inside mt-2">
                  <li>Power Budget: {entry.requirements.powerBudget}W</li>
                  <li>Area Constraint: {entry.requirements.areaConstraint}mm²</li>
                  <li>Performance Target: {entry.requirements.performanceTarget} MIPS</li>
                  <li>Priority: {entry.requirements.optimizationPriority}</li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium">Selected Workloads</h3>
                <ul className="list-disc list-inside mt-2">
                  {entry.requirements.selectedWorkloads.map((workload) => (
                    <li key={workload}>{workload}</li>
                  ))}
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Add more cards for results, optimization graphs, etc. */}
      </div>

      <Button 
        onClick={handleLoadDesign}
        className="mt-4"
      >
        Load This Design
      </Button>
    </div>
  )
} 