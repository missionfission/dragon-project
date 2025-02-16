"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Cpu, Activity, Loader2 } from "lucide-react"
import axios from 'axios'

interface PerformanceEstimate {
  mips: number;
  powerEfficiency: number;
  utilizationPercentage: number;
  thermalProfile: number;
  animationFrames: string[];
}

export default function ChipPerformanceEstimator() {
  const [isEstimating, setIsEstimating] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [estimate, setEstimate] = useState<PerformanceEstimate | null>(null)
  const [chipSpecs, setChipSpecs] = useState({
    clockSpeed: 1000, // MHz
    coreCount: 4,
    cacheSize: 8, // MB
    memoryBandwidth: 100 // GB/s
  })

  const handleEstimate = async () => {
    setIsEstimating(true)
    try {
      // TODO: Replace with actual API endpoint
      const response = await axios.post('http://localhost:8000/api/estimate-performance', chipSpecs)
      setEstimate(response.data)
      
      // Start animation
      if (response.data.animationFrames?.length > 0) {
        let frame = 0
        const interval = setInterval(() => {
          frame = (frame + 1) % response.data.animationFrames.length
          setCurrentFrame(frame)
        }, 100)
        return () => clearInterval(interval)
      }
    } catch (error) {
      console.error('Failed to estimate performance:', error)
    } finally {
      setIsEstimating(false)
    }
  }

  return (
    <Card className="bg-black/50 border-gray-800">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-2xl">Chip Performance Estimator</CardTitle>
            <CardDescription>Estimate your chip's performance based on specifications</CardDescription>
          </div>
          <Badge variant="secondary" className="h-6">
            <Cpu className="w-3 h-3 mr-1" />
            Beta
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-6">
          {/* Specifications Input */}
          <div className="grid gap-4">
            <div className="space-y-2">
              <Label>Clock Speed (MHz)</Label>
              <Slider
                value={[chipSpecs.clockSpeed]}
                onValueChange={([value]) => setChipSpecs(prev => ({ ...prev, clockSpeed: value }))}
                min={100}
                max={5000}
                step={100}
                className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
              />
              <div className="text-sm text-muted-foreground text-right">{chipSpecs.clockSpeed} MHz</div>
            </div>

            <div className="space-y-2">
              <Label>Core Count</Label>
              <Slider
                value={[chipSpecs.coreCount]}
                onValueChange={([value]) => setChipSpecs(prev => ({ ...prev, coreCount: value }))}
                min={1}
                max={128}
                step={1}
                className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
              />
              <div className="text-sm text-muted-foreground text-right">{chipSpecs.coreCount} Cores</div>
            </div>

            <div className="space-y-2">
              <Label>Cache Size (MB)</Label>
              <Slider
                value={[chipSpecs.cacheSize]}
                onValueChange={([value]) => setChipSpecs(prev => ({ ...prev, cacheSize: value }))}
                min={1}
                max={64}
                step={1}
                className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
              />
              <div className="text-sm text-muted-foreground text-right">{chipSpecs.cacheSize} MB</div>
            </div>

            <div className="space-y-2">
              <Label>Memory Bandwidth (GB/s)</Label>
              <Slider
                value={[chipSpecs.memoryBandwidth]}
                onValueChange={([value]) => setChipSpecs(prev => ({ ...prev, memoryBandwidth: value }))}
                min={10}
                max={1000}
                step={10}
                className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
              />
              <div className="text-sm text-muted-foreground text-right">{chipSpecs.memoryBandwidth} GB/s</div>
            </div>
          </div>

          <Button 
            onClick={handleEstimate}
            disabled={isEstimating}
            className="w-full"
          >
            {isEstimating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Estimating...
              </>
            ) : (
              <>
                <Activity className="mr-2 h-4 w-4" />
                Estimate Performance
              </>
            )}
          </Button>

          {/* Results Display */}
          {estimate && (
            <div className="grid grid-cols-2 gap-4 mt-4">
              <Card className="bg-gray-900/50 border-gray-800">
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-sm font-medium text-muted-foreground">Performance</div>
                    <div className="text-2xl font-bold mt-1">{estimate.mips.toLocaleString()} MIPS</div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-gray-900/50 border-gray-800">
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-sm font-medium text-muted-foreground">Power Efficiency</div>
                    <div className="text-2xl font-bold mt-1">{estimate.powerEfficiency.toFixed(2)} MIPS/W</div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-900/50 border-gray-800">
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-sm font-medium text-muted-foreground">Utilization</div>
                    <div className="text-2xl font-bold mt-1">{estimate.utilizationPercentage}%</div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-900/50 border-gray-800">
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-sm font-medium text-muted-foreground">Thermal Profile</div>
                    <div className="text-2xl font-bold mt-1">{estimate.thermalProfile}°C</div>
                  </div>
                </CardContent>
              </Card>

              {/* Animation Canvas */}
              {estimate.animationFrames?.length > 0 && (
                <div className="col-span-2">
                  <Card className="bg-gray-900/50 border-gray-800">
                    <CardContent className="pt-6">
                      <div className="aspect-video relative bg-black/30 rounded-lg overflow-hidden">
                        <img 
                          src={estimate.animationFrames[currentFrame]}
                          alt={`Performance visualization frame ${currentFrame + 1}`}
                          className="absolute inset-0 w-full h-full object-cover"
                        />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
} 