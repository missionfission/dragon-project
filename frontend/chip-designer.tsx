"use client"

import { useState, useEffect, useRef } from "react"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { Cpu, Zap, Maximize2, Activity, Loader2, BarChart2, PlusCircle, Trash2, Network, Cpu as CpuIcon, ChevronDown, History, Info } from "lucide-react"
import axios from 'axios'
import yaml from 'js-yaml'
import Editor from '@monaco-editor/react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { FormControl, InputLabel, MenuItem, TextField } from "@mui/material"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

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

// Update the interface for vector compute to match YAML
interface VectorComputeConfig {
  class: string;
  frequency: number;
  size: number;
  N_PE: number;
}

interface ChipConfig {
  name?: string;
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
  vector_compute: VectorComputeConfig;
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

// Add new interfaces for workload types and performance metrics
interface WorkloadPerformance {
  throughput: number;
  latency: number;
  powerEfficiency: number;
  utilizationRate: number;
}

// Add new interface for custom workload
interface CustomWorkload {
  name: string;
  description: string;
  filename: string;
  content: string;
}

// Add to existing interfaces section
interface WorkloadCategory {
  name: string;
  workloads: {
    name: string;
    description: string;
    performance?: WorkloadPerformance;
    isCustom?: boolean;
  }[];
}

// Move this constant outside the component
const DEFAULT_WORKLOAD_CATEGORIES: WorkloadCategory[] = [
  {
    name: "AI/ML Workloads",
    workloads: [
      {
        name: "ResNet-50",
        description: "Deep CNN for image classification",
      },
      {
        name: "BERT",
        description: "Transformer-based NLP model",
      },
      {
        name: "GPT-4", 
        description: "Large language model inference",
      },
      {
        name: "DLRM",
        description: "Deep Learning Recommendation Model",
      },
      {
        name: "SSD",
        description: "Single Shot MultiBox Detector",
      }
    ]
  },
  {
    name: "High Performance Computing",
    workloads: [
      {
        name: "HPCG",
        description: "High Performance Conjugate Gradient",
      },
      {
        name: "LINPACK",
        description: "Linear algebra benchmark",
      },
      {
        name: "STREAM",
        description: "Memory bandwidth benchmark",
      }
    ]
  },
  {
    name: "Graph Processing",
    workloads: [
      {
        name: "BFS",
        description: "Breadth-First Search",
      },
      {
        name: "PageRank",
        description: "Graph ranking algorithm",
      },
      {
        name: "Connected Components",
        description: "Graph connectivity analysis",
      }
    ]
  },
  {
    name: "Cryptography",
    workloads: [
      {
        name: "AES-256",
        description: "Advanced Encryption Standard",
      },
      {
        name: "SHA-3",
        description: "Secure Hash Algorithm",
      },
      {
        name: "RSA",
        description: "Public-key cryptography",
      }
    ]
  }
];

// Add new interfaces for system-level configuration
interface NetworkConfig {
  type: 'ethernet' | 'pcie' | 'nvlink' | 'infinity-fabric';
  bandwidth: number; // GB/s
  latency: number; // ns
  ports: number;
}

interface ProcessorConfig {
  type: 'cpu' | 'gpu' | 'accelerator';
  name: string;
  cores: number;
  frequency: number;
  memory: number;
  tdp: number;
}

interface SystemConfig {
  chips: ChipConfig[];
  processors: ProcessorConfig[];
  networks: NetworkConfig[];
  topology: 'mesh' | 'ring' | 'star' | 'fully-connected';
}

// Add default configurations
const defaultProcessors: ProcessorConfig[] = [
  {
    type: 'cpu',
    name: 'Host CPU',
    cores: 64,
    frequency: 3000,
    memory: 256,
    tdp: 280
  },
  {
    type: 'gpu',
    name: 'GPU Accelerator',
    cores: 6912,
    frequency: 1800,
    memory: 48,
    tdp: 350
  }
];

const defaultNetworks: NetworkConfig[] = [
  {
    type: 'pcie',
    bandwidth: 64,
    latency: 500,
    ports: 64
  },
  {
    type: 'nvlink',
    bandwidth: 300,
    latency: 100,
    ports: 12
  }
];

// Add new state for YAML editors
interface YamlEditors {
  [chipId: string]: string;
}

// Add new function to fetch local YAML file
const fetchDefaultConfig = async () => {
  try {
    const response = await fetch('/default.yaml');
    const yamlText = await response.text();
    return yamlText;
  } catch (error) {
    console.error('Error loading default configuration:', error);
    throw new Error('Failed to load default configuration');
  }
};

// Add new interfaces after the existing ones
interface SavedChipDesign {
  id: string;
  name: string;
  description?: string;
  requirements: ChipRequirements;
  design: ChipDesign;
  config: ChipConfig;
  createdAt: string;
}

// Add new interface for saved system configs
interface SavedSystemConfig {
  id: string;
  name: string;
  description?: string;
  config: SystemConfig;
  createdAt: string;
}

// Add new interface for system performance results
interface SystemPerformanceMetrics {
  totalThroughput: number;
  systemLatency: number;
  powerConsumption: number;
  networkUtilization: number;
  chipUtilizations: {
    chipId: string;
    utilization: number;
  }[];
  interconnectBandwidth: {
    source: string;
    destination: string;
    bandwidth: number;
    utilization: number;
  }[];
}

// Add new interface after the existing interfaces
interface DesignRun {
  id: string;
  timestamp: string;
  requirements: ChipRequirements;
  results: {
    chipDesign: ChipDesign;
    optimizationResults: {
      graph: string;
      animation_frames: string[];
    } | null;
  };
  config: ChipConfig;
}

// Add this interface with existing interfaces
interface PerformanceEstimate {
  mips: number;
  powerEfficiency: number;
  utilizationPercentage: number;
  thermalProfile: number;
  animationFrames: string[];
}

// Add this function after the interfaces and before the component
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
  const vectorArrayHeight = usableHeight * 0.15;
  const mainMemoryHeight = usableHeight * 0.15;
  
  // Add Global Buffer using SRAM parameters
  blocks.push({
    type: 'Global Buffer',
    x: padding,
    y: padding,
    width: usableWidth,
    height: globalBufferHeight,
    color: 'rgba(59, 130, 246, 0.3)', // blue with higher opacity
    details: {
      class: config.memory.level0.class,
      size: `${config.memory.level0.size / 1000000} MB`,
      banks: config.memory.level0.banks,
      frequency: `${config.memory.level0.frequency} MHz`
    }
  });

  // Calculate systolic array and MAC array positions
  const systolicStartY = padding + globalBufferHeight + 10;
  const computeWidth = usableWidth * 0.4; // Each compute block takes 40% of width
  const computeSpacing = usableWidth * 0.1; // 10% spacing between compute blocks
  
  // Left side for systolic array
  const systolicStartX = padding;
  
  // Right side for MAC array
  const macStartX = padding + computeWidth + computeSpacing;

  // Add systolic array PEs on the left
  if (config.mm_compute.type1.class === 'systolic_array') {
    const totalPEs = config.mm_compute.type1.N_PE;
    const peRows = Math.floor(Math.sqrt(totalPEs));
    const peCols = Math.ceil(totalPEs / peRows);
    
    const peSize = Math.min(
      Math.floor(computeWidth / peCols),
      Math.floor(systolicArrayHeight / peRows)
    ) - 2;

    for (let i = 0; i < peRows; i++) {
      for (let j = 0; j < peCols; j++) {
        if ((i * peCols + j) < totalPEs) {
          blocks.push({
            type: 'PE',
            x: systolicStartX + (j * (peSize + 2)),
            y: systolicStartY + (i * (peSize + 2)),
            width: peSize,
            height: peSize,
            color: 'rgba(16, 185, 129, 0.3)', // green with higher opacity
            details: {
              frequency: `${config.mm_compute.type1.frequency} MHz`,
              size: config.mm_compute.type1.size,
              id: `PE ${i * peCols + j + 1}/${totalPEs}`
            }
          });
        }
      }
    }
  }

  // Add MAC array on the right
  if (config.mm_compute.type2.class === 'mac') {
    blocks.push({
      type: 'MAC Array',
      x: macStartX,
      y: systolicStartY,
      width: computeWidth,
      height: systolicArrayHeight,
      color: 'rgba(16, 185, 129, 0.3)', // green with higher opacity
      details: {
        frequency: `${config.mm_compute.type2.frequency} MHz`,
        size: config.mm_compute.type2.size,
        total_instances: `${config.mm_compute.type2.N_PE} MAC units`,
        class: config.mm_compute.type2.class
      }
    });
  }

  // Add Vector Array below both compute blocks
  const vectorStartY = systolicStartY + systolicArrayHeight + 10;
  const vectorArrayWidth = usableWidth * 0.8;
  const vectorStartX = padding + (usableWidth - vectorArrayWidth) / 2;

  blocks.push({
    type: 'Vector Array',
    x: vectorStartX,
    y: vectorStartY,
    width: vectorArrayWidth,
    height: vectorArrayHeight,
    color: 'rgba(249, 115, 22, 0.3)', // orange with higher opacity
    details: {
      class: config.vector_compute.class,
      frequency: `${config.vector_compute.frequency} MHz`,
      total_instances: `${config.vector_compute.N_PE} Vector Units`,
      size: config.vector_compute.size
    }
  });

  // Add Main Memory (DRAM)
  blocks.push({
    type: 'Main Memory',
    x: padding,
    y: vectorStartY + vectorArrayHeight + 10,
    width: usableWidth,
    height: mainMemoryHeight,
    color: 'rgba(107, 114, 128, 0.3)', // gray with higher opacity
    details: {
      class: config.memory.level1.class,
      size: `${config.memory.level1.size / 1000000} MB`,
      banks: config.memory.level1.banks,
      frequency: `${config.memory.level1.frequency} MHz`
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

// Environment variables
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
  const [systemConfig, setSystemConfig] = useState<SystemConfig>({
    chips: [],
    processors: [
      {
        type: 'cpu',
        name: 'Host CPU',
        cores: 64,
        frequency: 3000,
        memory: 256,
        tdp: 280
      },
      {
        type: 'gpu',
        name: 'GPU Accelerator',
        cores: 6912,
        frequency: 1800,
        memory: 48,
        tdp: 350
      }
    ],
    networks: [
      {
        type: 'pcie',
        bandwidth: 64,
        latency: 500,
        ports: 64
      },
      {
        type: 'nvlink',
        bandwidth: 300,
        latency: 100,
        ports: 12
      }
    ],
    topology: 'mesh'
  });

  // Add ref for canvas
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Add new state for YAML editors
  const [yamlEditors, setYamlEditors] = useState<YamlEditors>({});

  // Add new state variables after the existing ones
  const [savedDesigns, setSavedDesigns] = useState<SavedChipDesign[]>([]);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [designName, setDesignName] = useState('');
  const [designDescription, setDesignDescription] = useState('');

  // Add new state variables for system config saving
  const [showSaveSystemDialog, setShowSaveSystemDialog] = useState(false);
  const [systemConfigName, setSystemConfigName] = useState('');
  const [systemConfigDescription, setSystemConfigDescription] = useState('');
  const [savedSystemConfigs, setSavedSystemConfigs] = useState<SavedSystemConfig[]>([]);

  // Add new state for system performance
  const [systemPerformance, setSystemPerformance] = useState<SystemPerformanceMetrics | null>(null);
  const [calculatingPerformance, setCalculatingPerformance] = useState(false);

  // Add new state for custom workloads dialog
  const [showAddWorkloadDialog, setShowAddWorkloadDialog] = useState(false);
  const [customWorkload, setCustomWorkload] = useState<CustomWorkload>({
    name: '',
    description: '',
    filename: '',
    content: ''
  });

  // Inside the ChipDesigner component, add the state:
  const [workloadCategories, setWorkloadCategories] = useState<WorkloadCategory[]>(DEFAULT_WORKLOAD_CATEGORIES);

  // Add new state for design history
  const [designHistory, setDesignHistory] = useState<DesignRun[]>([]);
  const [showHistoryDialog, setShowHistoryDialog] = useState(false);

  // Add new state for performance estimation
  const [isEstimating, setIsEstimating] = useState(false)
  const [performanceEstimate, setPerformanceEstimate] = useState<PerformanceEstimate | null>(null)

  // Move the saveSystemConfig function inside the component
  const saveSystemConfig = async () => {
    const newConfig: SavedSystemConfig = {
      id: Math.random().toString(36).substr(2, 9),
      name: systemConfigName,
      description: systemConfigDescription,
      config: systemConfig,
      createdAt: new Date().toISOString()
    };

    try {
      const existingConfigs = JSON.parse(localStorage.getItem('savedSystemConfigs') || '[]');
      const updatedConfigs = [...existingConfigs, newConfig];
      localStorage.setItem('savedSystemConfigs', JSON.stringify(updatedConfigs));
      setSavedSystemConfigs(updatedConfigs);
      setShowSaveSystemDialog(false);
      setSystemConfigName('');
      setSystemConfigDescription('');
    } catch (error) {
      setError('Failed to save system configuration');
      console.error('Error saving system config:', error);
    }
  };

  // Move the SaveSystemConfigDialog component inside too
  const SaveSystemConfigDialog = () => (
    <Dialog open={showSaveSystemDialog} onOpenChange={setShowSaveSystemDialog}>
      <DialogContent className="bg-gray-900 text-white">
        <DialogHeader>
          <DialogTitle>Save System Configuration</DialogTitle>
          <DialogDescription className="text-gray-400">
            Save your system configuration for future use
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="name">Configuration Name</Label>
            <Input
              id="name"
              value={systemConfigName}
              onChange={(e) => setSystemConfigName(e.target.value)}
              className="bg-gray-800"
            />
          </div>
          <div>
            <Label htmlFor="description">Description (optional)</Label>
            <Textarea
              id="description"
              value={systemConfigDescription}
              onChange={(e) => setSystemConfigDescription(e.target.value)}
              className="bg-gray-800"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setShowSaveSystemDialog(false)}>
            Cancel
          </Button>
          <Button onClick={saveSystemConfig}>Save Configuration</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  // Add template fetching
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/workload-templates`);
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
      const response = await axios.post(`${API_URL}/api/generate-chip`, requirements);
      setChipDesign(response.data);
      
      // Fetch optimization results
      const resultsResponse = await axios.get(`${API_URL}/api/optimization-results`);
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

  // Add to your existing useEffect blocks:
  useEffect(() => {
    // Check for previous design data
    const previousDesign = localStorage.getItem('loadPreviousDesign')
    if (previousDesign) {
      const designData = JSON.parse(previousDesign)
      // Load the previous design data
      setRequirements(designData.requirements)
      setChipDesign(designData.result)
      if (designData.optimization_data) {
        setOptimizationResults(designData.optimization_data)
      }
      // Clear the stored data
      localStorage.removeItem('loadPreviousDesign')
    }
  }, [])

  // Initialize first chip with default.yaml on component mount
  useEffect(() => {
    const initializeFirstChip = async () => {
      try {
        const defaultYaml = await fetchDefaultConfig();
        const newChipConfig = yaml.load(defaultYaml) as ChipConfig;
        newChipConfig.name = "Chip 1"; // Add default name
        
        setYamlEditors({
          'chip-0': defaultYaml
        });

        // Update system config with the parsed chip config
        setSystemConfig(prev => ({
          ...prev,
          chips: [newChipConfig]
        }));

        // Also set the individual chip config
        setConfig(newChipConfig);
      } catch (error) {
        setError('Failed to load initial configuration');
        console.error('Error:', error);
      }
    };

    initializeFirstChip();
  }, []);

  // Update addChip to include chip name
  const addChip = async () => {
    try {
      const defaultYaml = await fetchDefaultConfig();
      const newChipConfig = yaml.load(defaultYaml) as ChipConfig;
      const chipIndex = systemConfig.chips.length;
      newChipConfig.name = `Chip ${chipIndex + 1}`; // Add default name
      
      const newChipId = `chip-${chipIndex}`;
      
      setYamlEditors(prev => ({
        ...prev,
        [newChipId]: defaultYaml
      }));

      // Update system config with the new chip
      setSystemConfig(prev => ({
        ...prev,
        chips: [...prev.chips, newChipConfig]
      }));
    } catch (error) {
      setError('Failed to load default configuration');
      console.error('Error:', error);
    }
  };

  const removeChip = (index: number) => {
    setSystemConfig(prev => ({
      ...prev,
      chips: prev.chips.filter((_, i) => i !== index)
    }));
  };

  const updateProcessor = (index: number, updates: Partial<ProcessorConfig>) => {
    setSystemConfig(prev => ({
      ...prev,
      processors: prev.processors.map((p, i) => 
        i === index ? { ...p, ...updates } : p
      )
    }));
  };

  const updateNetwork = (index: number, updates: Partial<NetworkConfig>) => {
    setSystemConfig(prev => ({
      ...prev,
      networks: prev.networks.map((n, i) => 
        i === index ? { ...n, ...updates } : n
      )
    }));
  };

  // Add handler for chip name change
  const handleChipNameChange = (index: number, name: string) => {
    setSystemConfig(prev => ({
      ...prev,
      chips: prev.chips.map((chip, i) => 
        i === index ? { ...chip, name } : chip
      )
    }));
  };

  // Update handleChipYamlChange to properly update both chip config and system config
  const handleChipYamlChange = (chipId: string, value: string) => {
    try {
      // Update YAML editor content
      setYamlEditors(prev => ({
        ...prev,
        [chipId]: value
      }));

      // Parse YAML and update system config
      const chipConfig = yaml.load(value) as ChipConfig;
      const chipIndex = parseInt(chipId.split('-')[1]);
      
      // Update the system config with the new chip config
      setSystemConfig(prev => ({
        ...prev,
        chips: prev.chips.map((chip, i) => 
          i === chipIndex ? { ...chipConfig, name: chip.name } : chip
        )
      }));

      // If this is the currently selected chip, update the individual chip config too
      if (chipIndex === systemConfig.chips.findIndex(chip => chip === config)) {
        setConfig(chipConfig);
      }

      setError(null);
    } catch (error) {
      setError('Invalid YAML configuration');
      console.error('YAML parsing error:', error);
    }
  };

  // Add these functions before the return statement
  const saveChipDesign = async () => {
    if (!chipDesign || !config) return;

    const newDesign: SavedChipDesign = {
      id: Math.random().toString(36).substr(2, 9),
      name: designName,
      description: designDescription,
      requirements,
      design: chipDesign,
      config,
      createdAt: new Date().toISOString()
    };

    try {
      // Save to local storage for now (replace with API call in production)
      const existingDesigns = JSON.parse(localStorage.getItem('savedChipDesigns') || '[]');
      const updatedDesigns = [...existingDesigns, newDesign];
      localStorage.setItem('savedChipDesigns', JSON.stringify(updatedDesigns));
      setSavedDesigns(updatedDesigns);
      setShowSaveDialog(false);
      setDesignName('');
      setDesignDescription('');
    } catch (error) {
      setError('Failed to save chip design');
      console.error('Error saving design:', error);
    }
  };

  const loadSavedDesign = (design: SavedChipDesign) => {
    setRequirements(design.requirements);
    setChipDesign(design.design);
    setConfig(design.config);
    
    // Update YAML editor content
    const yamlString = yaml.dump(design.config);
    setYamlContent(yamlString);
    
    // Update chip layout
    setChipLayout(generateChipLayout(design.config));
  };

  // Add useEffect to load saved designs
  useEffect(() => {
    const loadSavedDesigns = () => {
      try {
        const designs = JSON.parse(localStorage.getItem('savedChipDesigns') || '[]');
        setSavedDesigns(designs);
      } catch (error) {
        console.error('Error loading saved designs:', error);
      }
    };
    
    loadSavedDesigns();
  }, []);

  // Add new component for saved designs dropdown in the system configuration section
  const SavedDesignsDropdown = () => (
    <Select
      onValueChange={(designId) => {
        const design = savedDesigns.find(d => d.id === designId);
        if (design) loadSavedDesign(design);
      }}
    >
      <SelectTrigger className="w-[200px]">
        <SelectValue placeholder="Load saved design" />
      </SelectTrigger>
      <SelectContent>
        {savedDesigns.map((design) => (
          <SelectItem key={design.id} value={design.id}>
            {design.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );

  // Add Dialog component for saving designs
  const SaveDesignDialog = () => (
    <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
      <DialogContent className="bg-gray-900 text-white">
        <DialogHeader>
          <DialogTitle>Save Chip Design</DialogTitle>
          <DialogDescription className="text-gray-400">
            Save your chip design for future reference
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="name">Design Name</Label>
            <Input
              id="name"
              value={designName}
              onChange={(e) => setDesignName(e.target.value)}
              className="bg-gray-800"
            />
          </div>
          <div>
            <Label htmlFor="description">Description (optional)</Label>
            <Textarea
              id="description"
              value={designDescription}
              onChange={(e) => setDesignDescription(e.target.value)}
              className="bg-gray-800"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setShowSaveDialog(false)}>
            Cancel
          </Button>
          <Button onClick={saveChipDesign}>Save Design</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  // Update the calculateSystemPerformance function
  const calculateSystemPerformance = async () => {
    try {
      setCalculatingPerformance(true);
      setError(null);

      // Ensure we have valid system configuration
      if (!systemConfig.chips.length) {
        setError('No chips configured in the system');
        return;
      }

      if (!requirements.selectedWorkloads.length) {
        setError('No workloads selected');
        return;
      }

      // Ensure we have at least one network configured
      if (!systemConfig.networks.length) {
        setError('No networks configured in the system');
        return;
      }

      // Prepare the request payload
      const payload = {
        systemConfig: {
          chips: systemConfig.chips.map(chip => ({
            ...chip,
            name: chip.name || `Chip ${systemConfig.chips.indexOf(chip)}`
          })),
          processors: systemConfig.processors.map(proc => ({
            type: proc.type,
            name: proc.name,
            cores: proc.cores,
            frequency: proc.frequency,
            memory: proc.memory,
            tdp: proc.tdp
          })),
          networks: systemConfig.networks.map(net => ({
            type: net.type,
            bandwidth: net.bandwidth,
            latency: net.latency,
            ports: net.ports
          })),
          topology: systemConfig.topology
        },
        workloads: requirements.selectedWorkloads,
        optimizationPriority: requirements.optimizationPriority
      };

      console.log('Sending system performance calculation request:', payload);
      const response = await axios.post(`${API_URL}/api/calculate-system-performance`, payload);
      console.log('Received system performance response:', response.data);
      setSystemPerformance(response.data);
    } catch (err) {
      setError('Failed to calculate system performance: ' + (err as Error).message);
      console.error('Error calculating performance:', err);
    } finally {
      setCalculatingPerformance(false);
    }
  };

  // Add new component for system performance display
  const SystemPerformanceDisplay = () => {
    if (!systemPerformance) return null;

    return (
      <Card className="bg-gray-900/50 border-gray-800">
        <CardHeader>
          <CardTitle>System Performance Results</CardTitle>
          <CardDescription>Overall system performance metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Main Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 rounded-lg bg-gray-800/50">
                <div className="text-sm text-gray-400">Total Throughput</div>
                <div className="text-xl font-semibold mt-1">
                  {systemPerformance.totalThroughput.toFixed(1)} TOPS
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50">
                <div className="text-sm text-gray-400">System Latency</div>
                <div className="text-xl font-semibold mt-1">
                  {systemPerformance.systemLatency.toFixed(2)} ms
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50">
                <div className="text-sm text-gray-400">Power Consumption</div>
                <div className="text-xl font-semibold mt-1">
                  {systemPerformance.powerConsumption.toFixed(1)} W
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50">
                <div className="text-sm text-gray-400">Network Utilization</div>
                <div className="text-xl font-semibold mt-1">
                  {systemPerformance.networkUtilization.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Chip Utilizations */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Chip Utilizations</h3>
              <div className="grid gap-4 md:grid-cols-2">
                {systemPerformance.chipUtilizations.map((chip) => (
                  <div key={chip.chipId} className="p-4 rounded-lg bg-gray-800/50">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">{chip.chipId}</span>
                      <span className="text-lg font-semibold">{chip.utilization.toFixed(1)}%</span>
                    </div>
                    <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500"
                        style={{ width: `${chip.utilization}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Interconnect Bandwidth */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Interconnect Performance</h3>
              <div className="grid gap-4">
                {systemPerformance.interconnectBandwidth.map((connection, index) => (
                  <div key={index} className="p-4 rounded-lg bg-gray-800/50">
                    <div className="flex justify-between items-center mb-2">
                      <div className="space-y-1">
                        <div className="text-sm text-gray-400">
                          {connection.source} → {connection.destination}
                        </div>
                        <div className="text-lg font-semibold">
                          {connection.bandwidth.toFixed(1)} GB/s
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-gray-400">Utilization</div>
                        <div className="text-lg font-semibold">
                          {connection.utilization.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-green-500"
                        style={{ width: `${connection.utilization}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  // Update the handleWorkloadFileUpload function
  const handleWorkloadFileUpload = async (file: File) => {
    try {
      const content = await file.text();
      setCustomWorkload(prev => ({
        ...prev,
        filename: file.name,
        content: content
      }));
    } catch (error) {
      setError('Failed to read workload file');
      console.error('Error:', error);
    }
  };

  // Add function to save custom workload
  const saveCustomWorkload = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/save-workload`, customWorkload);
      if (response.data.success) {
        setShowAddWorkloadDialog(false);
        setCustomWorkload({
          name: '',
          description: '',
          filename: '',
          content: ''
        });
      }
    } catch (error) {
      setError('Failed to save custom workload');
      console.error('Error:', error);
    }
  };

  // Add new dialog component for adding workloads
  const AddWorkloadDialog = () => (
    <Dialog open={showAddWorkloadDialog} onOpenChange={setShowAddWorkloadDialog}>
      <DialogContent className="bg-gray-900 text-white">
        <DialogHeader>
          <DialogTitle>Add Custom Workload</DialogTitle>
          <DialogDescription className="text-gray-400">
            Upload your custom workload file and provide details
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="workload-name">Workload Name</Label>
            <Input
              id="workload-name"
              value={customWorkload.name}
              onChange={(e) => setCustomWorkload(prev => ({ ...prev, name: e.target.value }))}
              className="bg-gray-800"
            />
          </div>
          <div>
            <Label htmlFor="workload-description">Description</Label>
            <Textarea
              id="workload-description"
              value={customWorkload.description}
              onChange={(e) => setCustomWorkload(prev => ({ ...prev, description: e.target.value }))}
              className="bg-gray-800"
            />
          </div>
          <div>
            <Label htmlFor="workload-file">Workload File</Label>
            <Input
              id="workload-file"
              type="file"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleWorkloadFileUpload(file);
              }}
              className="bg-gray-800"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setShowAddWorkloadDialog(false)}>
            Cancel
          </Button>
          <Button onClick={saveCustomWorkload}>Add Workload</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  // Modify the Workloads card content
  const WorkloadsSection = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Selected Workloads</h3>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowAddWorkloadDialog(true)}
        >
          Add Custom Workload
        </Button>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {workloadCategories.map((category) => (
          <div key={category.name} className="space-y-2">
            <h4 className="font-medium text-gray-400">{category.name}</h4>
            <div className="space-y-2">
              {category.workloads.map((workload) => (
                <div
                  key={workload.name}
                  className="flex items-center justify-between p-2 rounded-lg bg-gray-800/50"
                >
                  <div>
                    <div className="font-medium">{workload.name}</div>
                    <div className="text-sm text-gray-400">{workload.description}</div>
                  </div>
                  <Checkbox
                    checked={requirements.selectedWorkloads.includes(workload.name)}
                    onCheckedChange={(checked) => {
                      setRequirements(prev => ({
                        ...prev,
                        selectedWorkloads: checked
                          ? [...prev.selectedWorkloads, workload.name]
                          : prev.selectedWorkloads.filter(w => w !== workload.name)
                      }));
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  // Add after the existing handleGenerate function
  const saveToHistory = () => {
    if (!chipDesign || !config) return;
    
    const newRun: DesignRun = {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toISOString(),
      requirements,
      results: {
        chipDesign,
        optimizationResults
      },
      config
    };

    const updatedHistory = [...designHistory, newRun];
    setDesignHistory(updatedHistory);
    localStorage.setItem('designHistory', JSON.stringify(updatedHistory));
  };

  // Add useEffect to load history
  useEffect(() => {
    const savedHistory = localStorage.getItem('designHistory');
    if (savedHistory) {
      setDesignHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Add new component for history dialog
  const HistoryDialog = () => (
    <Dialog open={showHistoryDialog} onOpenChange={setShowHistoryDialog}>
      <DialogContent className="bg-gray-900 text-white">
        <DialogHeader>
          <DialogTitle>Design History</DialogTitle>
          <DialogDescription className="text-gray-400">
            View and compare previous design iterations
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          {designHistory.map((run) => (
            <div key={run.id} className="p-4 border border-gray-800 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">{run.id}</div>
                  <div className="text-sm text-gray-400">
                    {new Date(run.timestamp).toLocaleString()}
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setRequirements(run.requirements);
                    setConfig(run.config);
                    setShowHistoryDialog(false);
                  }}
                >
                  Load
                </Button>
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );

  // Add this function after existing functions
  const handleEstimatePerformance = async () => {
    setIsEstimating(true);
    try {
      const estimationParams = {
        clockSpeed: config?.mm_compute?.type1?.frequency || 1000,
        coreCount: config?.mm_compute?.type1?.N_PE || 4,
        cacheSize: config?.memory?.level0?.size ? config.memory.level0.size / (1024 * 1024) : 8, // Convert to MB
        memoryBandwidth: config?.memory?.level1?.frequency || 100
      };
      
      const estimateResponse = await axios.post(`${API_URL}/api/estimate-performance`, estimationParams);
      setPerformanceEstimate(estimateResponse.data);
    } catch (error) {
      setError('Failed to estimate performance: ' + (error as Error).message);
      console.error('Error:', error);
    } finally {
      setIsEstimating(false);
    }
  };

  // Update processor form section
  const ProcessorForm = ({ processor: proc, index }: { processor: ProcessorConfig, index: number }) => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Name</Label>
          <Input
            type="text"
            value={proc.name}
            onChange={(e) => updateProcessor(index, { name: e.target.value })}
          />
        </div>
        <div className="space-y-2">
          <Label>Type</Label>
          {proc.type === 'cpu' ? (
            <Select
              value={proc.name}
              onValueChange={(value) => updateProcessor(index, { name: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select CPU type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="RISC-V">RISC-V</SelectItem>
                <SelectItem value="ARM">ARM</SelectItem>
              </SelectContent>
            </Select>
          ) : proc.type === 'gpu' ? (
            <Select
              value={proc.name}
              onValueChange={(value) => updateProcessor(index, { name: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select GPU type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="NVIDIA A100">NVIDIA A100</SelectItem>
                <SelectItem value="NVIDIA B100">NVIDIA B100</SelectItem>
                <SelectItem value="NVIDIA H100">NVIDIA H100</SelectItem>
              </SelectContent>
            </Select>
          ) : (
            <Input
              type="text"
              value={proc.name}
              onChange={(e) => updateProcessor(index, { name: e.target.value })}
              placeholder="Accelerator name"
            />
          )}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Cores</Label>
          <Input
            type="number"
            value={proc.cores}
            onChange={(e) => updateProcessor(index, { cores: parseInt(e.target.value) || 0 })}
          />
        </div>
        <div className="space-y-2">
          <Label>Frequency (MHz)</Label>
          <Input
            type="number"
            value={proc.frequency}
            onChange={(e) => updateProcessor(index, { frequency: parseFloat(e.target.value) || 0 })}
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Memory (GB)</Label>
          <Input
            type="number"
            value={proc.memory}
            onChange={(e) => updateProcessor(index, { memory: parseFloat(e.target.value) || 0 })}
          />
        </div>
        <div className="space-y-2">
          <Label>TDP (W)</Label>
          <Input
            type="number"
            value={proc.tdp}
            onChange={(e) => updateProcessor(index, { tdp: parseFloat(e.target.value) || 0 })}
          />
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl">Chip and System Performance Estimator and Designer</h1>
          <p className="text-muted-foreground">Estimate your chip/system performance based on specifications, Design your custom chip by specifying requirements and workloads</p>
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
        <WorkloadsSection />
        <AddWorkloadDialog />

        {/* Results Section */}
        {chipLayout && (
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle>Design Results</CardTitle>
              <div className="flex justify-between items-center">
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
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowHistoryDialog(true)}
                    className="flex items-center gap-2"
                  >
                    <History className="w-4 h-4" />
                    History
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowSaveDialog(true)}
                  >
                    Save Design
                  </Button>
                </div>
              </div>
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

        {/* Generate and Estimate Buttons */}
        <div className="flex gap-4">
          <div className="flex-1 flex items-center gap-2">
            <Button
              size="lg"
              className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              onClick={handleGenerate}
              disabled={loading || isEstimating}
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
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Info className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent className="max-w-[300px]">
                <p>For non-AI workloads, the design is generated from control data flow graph analysis. For AI workloads, optimized templates are used as the base configuration.</p>
              </TooltipContent>
            </Tooltip>
          </div>

          <div className="flex-1 flex items-center gap-2">
            <Button
              size="lg"
              variant="outline"
              className="flex-1"
              onClick={handleEstimatePerformance}
              disabled={loading || isEstimating || !config}
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
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Info className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent className="max-w-[300px]">
                <p>For non-AI workloads, performance is estimated after the chip is designed. Chip component editing and visualization features are coming soon.</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* System Configuration */}
        <Card className="bg-gray-900/50 border-gray-800">
          <CardHeader>
            <div className="flex justify-between items-center">
              <div>
                <CardTitle>System Configuration</CardTitle>
                <CardDescription className="text-gray-400">Configure multi-chip system and networking</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Chips Section */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Custom Chips</h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={addChip}
                  className="flex items-center gap-2"
                >
                  <PlusCircle className="w-4 h-4" />
                  Add Chip
                </Button>
              </div>
              <div className="grid gap-6">
                {systemConfig.chips.map((chip, index) => (
                  <div
                    key={`chip-${index}`}
                    className="space-y-4 border border-gray-800 rounded-lg p-4"
                  >
                    <div className="flex items-center justify-between">
                      <div className="space-y-2 w-full">
                        <input
                          type="text"
                          value={chip?.name || `Chip ${index + 1}`}
                          onChange={(e) => handleChipNameChange(index, e.target.value)}
                          className="bg-transparent border-none font-medium text-lg focus:outline-none focus:ring-1 focus:ring-blue-500 rounded px-1 w-full"
                        />
                        <p className="text-sm text-gray-400">
                          {chip?.mm_compute?.type1?.class || chip?.mm_compute?.type2?.class || 'Custom'} Architecture
                          {chip?.vector_compute?.class && ' + Vector Compute'}
                        </p>
                      </div>
                      {index > 0 && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeChip(index)}
                          className="text-red-500 hover:text-red-400 ml-4"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      )}
                    </div>

                    {/* YAML Editor */}
                    <div className="h-[300px] border border-gray-700 rounded-lg">
                      <Editor
                        height="100%"
                        defaultLanguage="yaml"
                        theme="vs-dark"
                        value={yamlEditors[`chip-${index}`]}
                        onChange={(value) => handleChipYamlChange(`chip-${index}`, value || '')}
                        options={{
                          minimap: { enabled: false },
                          fontSize: 12,
                          scrollBeyondLastLine: false,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Processor Configuration */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Processor Configuration</h3>
              <div className="space-y-4">
                {systemConfig.processors.map((proc, index) => (
                  <div key={index} className="p-4 border border-gray-800 rounded-lg space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <CpuIcon className="w-4 h-4" />
                        <span className="font-medium">{proc.name}</span>
                      </div>
                      <Badge variant="outline">{proc.type.toUpperCase()}</Badge>
                    </div>
                    <ProcessorForm processor={proc} index={index} onUpdate={updateProcessor} />
                  </div>
                ))}
              </div>
            </div>

            {/* Network Configuration */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Network Configuration</h3>
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <Network className="w-4 h-4" />
                  <Select
                    value={systemConfig.topology}
                    onValueChange={(value) => 
                      setSystemConfig(prev => ({ ...prev, topology: value as SystemConfig['topology'] }))
                    }
                  >
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Select topology" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="mesh">Mesh Network</SelectItem>
                      <SelectItem value="ring">Ring Network</SelectItem>
                      <SelectItem value="star">Star Network</SelectItem>
                      <SelectItem value="fully-connected">Fully Connected</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  {systemConfig.networks.map((network, index) => (
                    <div key={index} className="p-4 border border-gray-800 rounded-lg space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{network.type.toUpperCase()}</span>
                        <Badge variant="outline">{network.bandwidth} GB/s</Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm text-gray-400">Latency (ns)</label>
                          <input
                            type="number"
                            value={network.latency}
                            onChange={(e) => updateNetwork(index, { latency: parseInt(e.target.value) })}
                            className="w-full bg-gray-800/50 border border-gray-700 rounded-md px-2 py-1"
                          />
                        </div>
                        <div>
                          <label className="text-sm text-gray-400">Ports</label>
                          <input
                            type="number"
                            value={network.ports}
                            onChange={(e) => updateNetwork(index, { ports: parseInt(e.target.value) })}
                            className="w-full bg-gray-800/50 border border-gray-700 rounded-md px-2 py-1"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* System Configurations Section */}
            <div className="space-y-4 mb-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Saved System Configurations</h3>
                <div className="flex items-center gap-4">
                  <Select
                    onValueChange={(configId) => {
                      const config = savedSystemConfigs.find(c => c.id === configId);
                      if (config) setSystemConfig(config.config);
                    }}
                  >
                    <SelectTrigger className="w-[200px]">
                      <SelectValue placeholder="Load system config" />
                    </SelectTrigger>
                    <SelectContent>
                      {savedSystemConfigs.map((config) => (
                        <SelectItem key={config.id} value={config.id}>
                          {config.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowSaveSystemDialog(true)}
                  >
                    Save System Config
                  </Button>
                </div>
              </div>
            </div>

            {/* Add Performance Analysis section at bottom */}
            <div className="border-t border-gray-800 pt-6 mt-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">System Performance Analysis</h3>
                <h4 className="text-lg font-semibold">Uses CPU, GPU and accelerator simulators and maps the workload to the entire system</h4>
                <Button
                  onClick={calculateSystemPerformance}
                  disabled={calculatingPerformance || !requirements.selectedWorkloads.length || !systemConfig.chips.length}
                  className="flex items-center gap-2"
                >
                  {calculatingPerformance ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Calculating...
                    </>
                  ) : (
                    <>
                      <BarChart2 className="w-4 h-4" />
                      Calculate Performance
                    </>
                  )}
                </Button>
              </div>

              {/* Performance Results */}
              {systemPerformance && <SystemPerformanceDisplay />}
            </div>
          </CardContent>
        </Card>

        {/* Save Design Dialog */}
        <SaveDesignDialog />

        {/* Save System Config Dialog */}
        <SaveSystemConfigDialog />

        {/* History Dialog */}
        <HistoryDialog />

        {/* Performance Estimate Results */}
        {performanceEstimate && (
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle>Performance Estimation</CardTitle>
              <CardDescription>Estimated performance metrics based on current configuration</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-lg bg-gray-800/50">
                  <div className="text-sm text-gray-400">Performance</div>
                  <div className="text-xl font-semibold mt-1">
                    {performanceEstimate.mips.toLocaleString()} MIPS
                  </div>
                </div>
                
                <div className="p-4 rounded-lg bg-gray-800/50">
                  <div className="text-sm text-gray-400">Power Efficiency</div>
                  <div className="text-xl font-semibold mt-1">
                    {performanceEstimate.powerEfficiency.toFixed(2)} MIPS/W
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-gray-800/50">
                  <div className="text-sm text-gray-400">Utilization</div>
                  <div className="text-xl font-semibold mt-1">
                    {performanceEstimate.utilizationPercentage}%
                  </div>
                </div>

                <div className="p-4 rounded-lg bg-gray-800/50">
                  <div className="text-sm text-gray-400">Thermal Profile</div>
                  <div className="text-xl font-semibold mt-1">
                    {performanceEstimate.thermalProfile}°C
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

