"use client"

import { useState, useEffect } from "react"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { Cpu, Zap, Maximize2, Activity, Loader2, BarChart2 } from "lucide-react"
import axios from 'axios'

// Add interfaces
interface ChipRequirements {
  powerBudget: number;
  areaConstraint: number;
  performanceTarget: number;
  selectedWorkloads: string[];
  optimizationPriority: 'power' | 'performance' | 'balanced';
}

interface ChipBlock {
  id: string;
  type: string;
  size: { width: number; height: number };
  position: { x: number; y: number };
  powerConsumption: number;
  performance: number;
  utilization: number;
}

interface ChipDesign {
  blocks: ChipBlock[];
  totalPower: number;
  totalArea: number;
  estimatedPerformance: number;
  powerEfficiency: number;
}

interface WorkloadTemplate {
  name: string;
  workloads: string[];
  recommendedPower: number;
  recommendedArea: number;
  recommendedPerformance: number;
}

export default function ChipDesigner() {
  const [isGenerating, setIsGenerating] = useState(false)
  const [optimization, setOptimization] = useState("balanced")
  const [requirements, setRequirements] = useState<ChipRequirements>({
    powerBudget: 100,
    areaConstraint: 100,
    performanceTarget: 1000,
    selectedWorkloads: [],
    optimizationPriority: 'balanced'
  });
  const [chipDesign, setChipDesign] = useState<ChipDesign | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [templates, setTemplates] = useState<WorkloadTemplate[]>([]);
  const [activeTab, setActiveTab] = useState<'design' | 'metrics'>('design');

  // Add template fetching
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/workload-templates');
        setTemplates(response.data.templates);
      } catch (err) {
        console.error('Failed to fetch templates:', err);
      }
    };
    fetchTemplates();
  }, []);

  // Add template selection handler
  const handleTemplateSelect = (template: WorkloadTemplate) => {
    setRequirements({
      ...requirements,
      powerBudget: template.recommendedPower,
      areaConstraint: template.recommendedArea,
      performanceTarget: template.recommendedPerformance,
      selectedWorkloads: template.workloads
    });
  };

  // Add workload toggle handler
  const handleWorkloadToggle = (workload: string) => {
    setRequirements(prev => ({
      ...prev,
      selectedWorkloads: prev.selectedWorkloads.includes(workload)
        ? prev.selectedWorkloads.filter(w => w !== workload)
        : [...prev.selectedWorkloads, workload]
    }));
  };

  // Update the slider handlers to update requirements
  const handlePowerChange = (value: number[]) => {
    setRequirements(prev => ({
      ...prev,
      powerBudget: value[0]
    }));
  };

  const handleAreaChange = (value: number[]) => {
    setRequirements(prev => ({
      ...prev,
      areaConstraint: value[0]
    }));
  };

  const handlePerformanceChange = (value: number[]) => {
    setRequirements(prev => ({
      ...prev,
      performanceTarget: value[0]
    }));
  };

  const handleOptimizationChange = (value: string) => {
    setOptimization(value);
    setRequirements(prev => ({
      ...prev,
      optimizationPriority: value as 'power' | 'performance' | 'balanced'
    }));
  };

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Ensure we're sending the correct data structure
      const requestData = {
        powerBudget: requirements.powerBudget,
        areaConstraint: requirements.areaConstraint,
        performanceTarget: requirements.performanceTarget,
        selectedWorkloads: requirements.selectedWorkloads,
        optimizationPriority: requirements.optimizationPriority
      };
      
      const response = await axios.post('http://localhost:8000/api/generate-chip', requestData);
      setChipDesign(response.data);
      setActiveTab('metrics');
    } catch (err) {
      setError('Failed to generate chip design. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl">Chip Designer</h1>
          <p className="text-muted-foreground">Design your custom chip by specifying requirements and workloads</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {/* Templates Card */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle>Predefined Templates</CardTitle>
              <CardDescription className="text-gray-400">Start with an optimized base configuration</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              {templates.map((template) => (
                <Button
                  key={template.name}
                  variant="outline"
                  className="justify-between"
                  onClick={() => handleTemplateSelect(template)}
                >
                  {template.name}
                  {template.name === 'AI Accelerator' && (
                    <Badge variant="secondary">Popular</Badge>
                  )}
                </Button>
              ))}
            </CardContent>
          </Card>

          {/* Requirements Card */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle>Custom Requirements</CardTitle>
              <CardDescription className="text-gray-400">Fine-tune your chip specifications</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Power Budget */}
              <div className="space-y-4">
                <div className="flex justify-between">
                  <label className="text-sm">Power Budget</label>
                  <span className="text-sm text-muted-foreground">{requirements.powerBudget}W</span>
                </div>
                <Slider
                  value={[requirements.powerBudget]}
                  onValueChange={handlePowerChange}
                  max={300}
                  step={1}
                  className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
                />
              </div>

              {/* Area Constraint */}
              <div className="space-y-4">
                <div className="flex justify-between">
                  <label className="text-sm">Area Constraint</label>
                  <span className="text-sm text-muted-foreground">{requirements.areaConstraint}mm²</span>
                </div>
                <Slider
                  value={[requirements.areaConstraint]}
                  onValueChange={handleAreaChange}
                  max={200}
                  step={1}
                  className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
                />
              </div>

              {/* Performance Target */}
              <div className="space-y-4">
                <div className="flex justify-between">
                  <label className="text-sm">Performance Target</label>
                  <span className="text-sm text-muted-foreground">{requirements.performanceTarget} MIPS</span>
                </div>
                <Slider
                  value={[requirements.performanceTarget]}
                  onValueChange={handlePerformanceChange}
                  max={2000}
                  step={10}
                  className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Optimization Priority */}
        <Card className="bg-gray-900/50 border-gray-800">
          <CardHeader>
            <CardTitle>Optimization Priority</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={optimization} onValueChange={handleOptimizationChange} className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="power" className="data-[state=active]:bg-green-600">
                  <Zap className="w-4 h-4 mr-2" />
                  Power
                </TabsTrigger>
                <TabsTrigger value="balanced" className="data-[state=active]:bg-blue-600">
                  <Activity className="w-4 h-4 mr-2" />
                  Balanced
                </TabsTrigger>
                <TabsTrigger value="performance" className="data-[state=active]:bg-purple-600">
                  <Maximize2 className="w-4 h-4 mr-2" />
                  Performance
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </CardContent>
        </Card>

        {/* Workloads */}
        <Card className="bg-gray-900/50 border-gray-800">
          <CardHeader>
            <CardTitle>Workloads</CardTitle>
            <CardDescription className="text-gray-400">Select the primary workloads for your chip</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3">
              {[
                "Machine Learning",
                "Image Processing",
                "Data Analytics",
                "Cryptography",
                "Network Processing",
                "General Computing",
              ].map((workload) => (
                <label
                  key={workload}
                  className="flex items-center space-x-3 border border-gray-800 rounded-lg p-4 hover:bg-gray-800/50 transition-colors"
                >
                  <Checkbox 
                    id={workload}
                    checked={requirements.selectedWorkloads.includes(workload)}
                    onCheckedChange={() => handleWorkloadToggle(workload)}
                  />
                  <span className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                    {workload}
                  </span>
                </label>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Results Section */}
        {chipDesign && (
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle>Design Results</CardTitle>
              <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as 'design' | 'metrics')} className="w-full">
                <TabsList>
                  <TabsTrigger value="design">
                    <Cpu className="w-4 h-4 mr-2" />
                    Chip Design
                  </TabsTrigger>
                  <TabsTrigger value="metrics">
                    <BarChart2 className="w-4 h-4 mr-2" />
                    Performance Metrics
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </CardHeader>
            <CardContent>
              {activeTab === 'design' ? (
                <div className="border-2 rounded-lg h-[400px] relative bg-muted/50">
                  {chipDesign.blocks.map(block => (
                    <div
                      key={block.id}
                      className="absolute rounded-lg shadow-lg transition-all duration-200 hover:scale-105 hover:shadow-xl backdrop-blur-sm"
                      style={{
                        left: block.position.x,
                        top: block.position.y,
                        width: block.size.width,
                        height: block.size.height,
                        backgroundColor: block.type === 'Computing' ? 'rgba(249, 115, 22, 0.2)' :
                                       block.type === 'Memory' ? 'rgba(59, 130, 246, 0.2)' :
                                       block.type === 'Network' ? 'rgba(16, 185, 129, 0.2)' :
                                       block.type === 'Security' ? 'rgba(239, 68, 68, 0.2)' :
                                       'rgba(107, 114, 128, 0.2)',
                        border: '1px solid',
                        borderColor: block.type === 'Computing' ? 'rgb(249, 115, 22)' :
                                    block.type === 'Memory' ? 'rgb(59, 130, 246)' :
                                    block.type === 'Network' ? 'rgb(16, 185, 129)' :
                                    block.type === 'Security' ? 'rgb(239, 68, 68)' :
                                    'rgb(107, 114, 128)',
                      }}
                    >
                      <div className="h-full p-3 flex flex-col justify-between">
                        <div className="text-xs font-semibold">{block.type}</div>
                        <div className="text-xs text-muted-foreground">
                          Power: {block.powerConsumption.toFixed(1)}W
                          <br />
                          Util: {(block.utilization * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-muted">
                    <div className="text-sm font-medium text-muted-foreground">Total Power</div>
                    <div className="mt-1 text-2xl font-semibold">
                      {chipDesign.totalPower.toFixed(1)}W
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-muted">
                    <div className="text-sm font-medium text-muted-foreground">Total Area</div>
                    <div className="mt-1 text-2xl font-semibold">
                      {chipDesign.totalArea.toFixed(1)}mm²
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-muted">
                    <div className="text-sm font-medium text-muted-foreground">Performance</div>
                    <div className="mt-1 text-2xl font-semibold">
                      {chipDesign.estimatedPerformance.toFixed(0)} MIPS
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-muted">
                    <div className="text-sm font-medium text-muted-foreground">Power Efficiency</div>
                    <div className="mt-1 text-2xl font-semibold">
                      {chipDesign.powerEfficiency.toFixed(1)} MIPS/W
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Error message */}
        {error && (
          <div className="p-4 text-destructive text-sm bg-destructive/10 rounded-lg border border-destructive/20">
            {error}
          </div>
        )}

        {/* Generate Button */}
        <Button
          size="lg"
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
          onClick={handleGenerate}
          disabled={loading}
        >
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating Chip Design...
            </>
          ) : (
            <>
              <Cpu className="mr-2 h-4 w-4" />
              Generate Chip Design
            </>
          )}
        </Button>
      </div>
    </div>
  )
}

