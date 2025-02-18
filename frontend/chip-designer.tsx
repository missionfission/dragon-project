"use client"

import { useState, useEffect, useRef } from "react"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { Cpu, Zap, Maximize2, Activity, Loader2, BarChart2 } from "lucide-react"
import axios from 'axios'
import yaml from 'js-yaml'
import Editor from '@monaco-editor/react'

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

// Add new interfaces for YAML config
interface TechnologyConfig {
  wire_cap: number;
  sense_amp_time: number;
  plogic_node: number;
  logic_node: number;
}

interface MemoryConfig {
  class: string;
  frequency: number;
  banks: number;
  read_ports: number;
  write_ports: number;
  width: number;
  size: number;
  leakage_power: number;
  read_energy?: number;
  write_energy?: number;
}

interface ComputeConfig {
  type1: {
    class: string;
    frequency: number;
    size: number;
    N_PE: number;
    area: number;
    per_op_energy: number;
  };
  type2: {
    class: string;
    frequency: number;
    size: number;
    N_PE: number;
    Tile: {
      TileX: number;
      TileY: number;
      Number: number;
    };
  };
}

interface ChipConfig {
  technology: TechnologyConfig;
  voltage: number;
  memory_levels: number;
  memory: {
    level0: MemoryConfig;
    level1: MemoryConfig;
  };
  mm_compute: ComputeConfig;
  rf: {
    energy: number;
    area: number;
  };
  vector_compute: ComputeConfig;
  force_connectivity: number;
}

// Add this after your existing interfaces
interface ChipLayout {
  blocks: {
    type: string;
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
    details: any;
  }[];
}

// Add this function to convert config to visual layout
function generateChipLayout(config: ChipConfig): ChipLayout {
  const blocks = [];
  // Define container dimensions
  const containerWidth = 700;
  const containerHeight = 500;
  const padding = 20; // Padding from container edges
  
  // Calculate available space
  const usableWidth = containerWidth - (2 * padding);
  const usableHeight = containerHeight - (2 * padding);
  
  // Define relative sizes (percentages of usable space)
  const globalBufferHeight = usableHeight * 0.15;
  const systolicArrayHeight = usableHeight * 0.5;
  const simdUnitsHeight = usableHeight * 0.15;
  const mainMemoryHeight = usableHeight * 0.15;
  
  // Add Global Buffer (24MB split into 2 banks)
  blocks.push({
    type: 'Global Buffer',
    x: padding,
    y: padding,
    width: usableWidth,
    height: globalBufferHeight,
    color: 'rgba(59, 130, 246, 0.3)', // blue with higher opacity
    details: {
      size: '24 MB',
      banks: 2
    }
  });

  // Calculate systolic array position and dimensions
  const systolicStartY = padding + globalBufferHeight + 10;
  const systolicWidth = usableWidth * 0.8;
  const systolicStartX = padding + (usableWidth - systolicWidth) / 2;

  // Calculate PE dimensions based on active compute type
  const activeCompute = config.mm_compute.type1.class === 'systolic_array' 
    ? config.mm_compute.type1 
    : config.mm_compute.type2;

  const peRows = activeCompute.class === 'systolic_array'
    ? Math.floor(Math.sqrt(activeCompute.N_PE))
    : activeCompute.Tile.TileX;
    
  const peCols = activeCompute.class === 'systolic_array'
    ? peRows
    : activeCompute.Tile.TileY;

  // Calculate PE size to fit within systolic array area
  const peSize = Math.min(
    Math.floor(systolicWidth / peCols),
    Math.floor(systolicArrayHeight / peRows)
  ) - 2; // Subtract for gap

  // Add Processing Elements (PEs) in a grid
  for (let i = 0; i < peRows; i++) {
    for (let j = 0; j < peCols; j++) {
      blocks.push({
        type: 'PE',
        x: systolicStartX + (j * (peSize + 2)),
        y: systolicStartY + (i * (peSize + 2)),
        width: peSize,
        height: peSize,
        color: 'rgba(16, 185, 129, 0.3)', // green with higher opacity
        details: {
          frequency: `${activeCompute.frequency} MHz`,
          size: activeCompute.size
        }
      });
    }
  }

  // Add SIMD Units
  const simdStartY = systolicStartY + systolicArrayHeight + 10;
  const simdWidth = (usableWidth - 30) / 4; // 30 is total gap between units
  const simdSpacing = 10;

  for (let i = 0; i < 4; i++) {
    blocks.push({
      type: 'SIMD Unit',
      x: padding + (i * (simdWidth + simdSpacing)),
      y: simdStartY,
      width: simdWidth,
      height: simdUnitsHeight,
      color: 'rgba(249, 115, 22, 0.3)', // orange with higher opacity
      details: {
        type: 'Vector Processing'
      }
    });
  }

  // Add Main Memory
  blocks.push({
    type: 'Main Memory',
    x: padding,
    y: simdStartY + simdUnitsHeight + 10,
    width: usableWidth,
    height: mainMemoryHeight,
    color: 'rgba(107, 114, 128, 0.3)', // gray with higher opacity
    details: {
      class: config.memory.level1.class,
      size: `${config.memory.level1.size / 1000000} MB`,
      banks: config.memory.level1.banks
    }
  });

  // Add container block
  blocks.push({
    type: 'Container',
    x: 0,
    y: 0,
    width: containerWidth,
    height: containerHeight,
    color: 'rgba(0, 0, 0, 0)', // transparent
    details: {},
  });

  return { blocks };
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
  const [optimizationResults, setOptimizationResults] = useState<{
    graph: string;
    animation_frames: string[];
  } | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [chipLayout, setChipLayout] = useState<ChipLayout | null>(null);
  const [config, setConfig] = useState<ChipConfig | null>(null);
  const [yamlContent, setYamlContent] = useState<string>('');

  // Add ref for canvas
  const canvasRef = useRef<HTMLCanvasElement>(null)

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
      const response = await axios.post('http://localhost:8000/api/generate-chip', requirements);
      setChipDesign(response.data);
      
      // Fetch optimization results
      const resultsResponse = await axios.get('http://localhost:8000/api/optimization-results');
      setOptimizationResults(resultsResponse.data);
      
      setActiveTab('metrics');
    } catch (err) {
      setError('Failed to generate chip design. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Add function to render animation frames
  const renderAnimationFrames = (frames: string[]) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let currentFrame = 0
    const images: HTMLImageElement[] = []
    let isPlaying = true

    // Preload all images
    const loadImages = async () => {
      for (const frame of frames) {
        const img = new Image()
        img.src = `data:image/png;base64,${frame}`
        await new Promise((resolve) => {
          img.onload = resolve
        })
        images.push(img)
      }

      // Set canvas size based on first image
      if (images[0]) {
        canvas.width = images[0].width
        canvas.height = images[0].height
      }

      // Start animation loop
      const animate = () => {
        if (!isPlaying) return

        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(images[currentFrame], 0, 0)
        
        currentFrame = (currentFrame + 1) % images.length
        requestAnimationFrame(animate)
      }

      animate()
    }

    loadImages()

    // Return cleanup function
    return () => {
      isPlaying = false
    }
  }

  // Use effect to start animation when results are available
  useEffect(() => {
    if (optimizationResults?.animation_frames) {
      const cleanup = renderAnimationFrames(optimizationResults.animation_frames)
      return cleanup
    }
  }, [optimizationResults])

  // Modify the loadConfig function to update YAML editor content
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await fetch('/default.yaml');
        const yamlText = await response.text();
        setYamlContent(yamlText); // Set the YAML content for editor
        const config = yaml.load(yamlText) as ChipConfig;
        setConfig(config);
        setChipLayout(generateChipLayout(config));
      } catch (err) {
        console.error('Failed to load config:', err);
      }
    };
    loadConfig();
  }, []);

  // Add YAML editor change handler
  const handleYamlChange = (value: string | undefined) => {
    if (!value) return;
    setYamlContent(value);
    try {
      const newConfig = yaml.load(value) as ChipConfig;
      handleConfigUpdate(newConfig);
    } catch (err) {
      console.error('Invalid YAML:', err);
    }
  };

  // Update handleConfigUpdate to work with local state
  const handleConfigUpdate = (newConfig: ChipConfig) => {
    setConfig(newConfig);
    setChipLayout(generateChipLayout(newConfig));
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

          {/* Add YAML Editor Card */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle>Chip Configuration</CardTitle>
              <CardDescription className="text-gray-400">Edit YAML configuration directly</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <Editor
                height="100%"
                defaultLanguage="yaml"
                theme="vs-dark"
                value={yamlContent}
                onChange={handleYamlChange}
                options={{
                  minimap: { enabled: false },
                  fontSize: 12,
                  scrollBeyondLastLine: false,
                }}
              />
            </CardContent>
          </Card>

          {/* Chip Layout Preview Card */}
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle>Chip Layout</CardTitle>
              <CardDescription className="text-gray-400">Real-time visualization of chip architecture</CardDescription>
            </CardHeader>
            <CardContent>
              {chipLayout && (
                <div className="border-2 rounded-lg relative bg-gray-950 overflow-hidden"
                     style={{ width: '700px', height: '500px' }}>
                  {chipLayout.blocks.map((block, index) => (
                    <div
                      key={index}
                      className="absolute rounded-lg shadow-lg transition-all duration-200 hover:scale-105 hover:shadow-xl"
                      style={{
                        left: `${block.x}px`,
                        top: `${block.y}px`,
                        width: `${block.width}px`,
                        height: `${block.height}px`,
                        backgroundColor: block.color,
                        border: block.type === 'Container' ? '2px solid rgba(255,255,255,0.1)' : '1px solid',
                        borderColor: block.color.replace('0.2', '1'),
                      }}
                    >
                      <div className="h-full p-3 flex flex-col justify-between">
                        {block.type !== 'Container' && (
                          <>
                            <div className="text-xs font-semibold text-white">{block.type}</div>
                            {block.details && (
                              <div className="text-xs text-gray-300">
                                {Object.entries(block.details).map(([key, value]) => (
                                  <div key={key}>{`${key}: ${value}`}</div>
                                ))}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
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
        {chipLayout && (
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
                <div className="border-2 rounded-lg h-[600px] relative bg-gray-950">
                  {chipLayout.blocks.map((block, index) => (
                    <div
                      key={index}
                      className="absolute rounded-lg shadow-lg transition-all duration-200 hover:scale-105 hover:shadow-xl"
                      style={{
                        left: `${block.x}px`,
                        top: `${block.y}px`,
                        width: `${block.width}px`,
                        height: `${block.height}px`,
                        backgroundColor: block.color,
                        border: '1px solid',
                        borderColor: block.color.replace('0.2', '1'),
                      }}
                    >
                      <div className="h-full p-3 flex flex-col justify-between">
                        <div className="text-xs font-semibold">{block.type}</div>
                        {block.details && (
                          <div className="text-xs text-muted-foreground">
                            {Object.entries(block.details).map(([key, value]) => (
                              <div key={key}>{`${key}: ${value}`}</div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-muted">
                    <div className="text-sm font-medium text-muted-foreground">Total Power</div>
                    <div className="mt-1 text-2xl font-semibold">
                      {chipDesign?.totalPower.toFixed(1)}W
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-muted">
                    <div className="text-sm font-medium text-muted-foreground">Total Area</div>
                    <div className="mt-1 text-2xl font-semibold">
                      {chipDesign?.totalArea.toFixed(1)}mm²
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-muted">
                    <div className="text-sm font-medium text-muted-foreground">Performance</div>
                    <div className="mt-1 text-2xl font-semibold">
                      {chipDesign?.estimatedPerformance.toFixed(0)} MIPS
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-muted">
                    <div className="text-sm font-medium text-muted-foreground">Power Efficiency</div>
                    <div className="mt-1 text-2xl font-semibold">
                      {chipDesign?.powerEfficiency.toFixed(1)} MIPS/W
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Optimization Results */}
        {optimizationResults && (
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle>Optimization Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Progress Graph</h3>
                  <img 
                    src={`data:image/png;base64,${optimizationResults.graph}`}
                    alt="Optimization Progress"
                    className="w-full rounded-lg"
                  />
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2">Design Evolution</h3>
                  <div className="relative aspect-square w-full max-w-[600px] mx-auto">
                    <canvas
                      ref={canvasRef}
                      className="w-full h-full rounded-lg"
                    />
                  </div>
                </div>
              </div>
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

