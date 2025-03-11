from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import json
import math
import random
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import imageio
import traceback  # Add for better error tracking
from datetime import datetime
from uuid import uuid4
import yaml
import logging  # Add for better logging
from src.src_main import (
    design_runner,
    visualize_performance_estimation,
    analyze_network_utilization,
    analyze_network_latency,
    analyze_workload_distribution,
    generate_system_visualization,
    get_backprop_memory,
    Mapper
)
from src.synthesis.hls import parse_graph, get_stats
import os
from src.common_models import (
    alexnet_graph,
    vggnet_graph, 
    resnet_50_graph,
    bert_graph,
    gpt2_graph,
    langmodel_graph
)
import ast

from src.ir.cfg.staticfg import CFGBuilder

# design_runner([vggnet_graph()])
# # for node in dlrm_graph.nodes:
# #     print(node.in_edge_mem + node.mem_fetch + node.out_edge_mem, node.compute_expense )

# design_tech_runner([dlrm_graph])

# 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chip Designer API",
    description="API for generating and managing chip designs",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add these constants near the top after imports
LOGS_DIR = Path("logs")
CONFIGS_DIR = Path("configs")
DEFAULT_CONFIG_FILE = CONFIGS_DIR / "default.yaml"

# Ensure required directories exist
LOGS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)

# First define all the base config models
class TechnologyConfig(BaseModel):
    wire_cap: float
    sense_amp_time: float
    plogic_node: int
    logic_node: int

class MemoryConfig(BaseModel):
    class_type: str
    frequency: int
    banks: int
    read_ports: int
    write_ports: int
    width: int
    size: int
    leakage_power: float
    read_energy: Optional[float] = None
    write_energy: Optional[float] = None

    class Config:
        allow_population_by_field_name = True

class ComputeConfig(BaseModel):
    type1: Dict[str, Any]
    type2: Dict[str, Any]

class VectorComputeConfig(BaseModel):
    class_type: str
    frequency: int
    size: int
    N_PE: int

    class Config:
        allow_population_by_field_name = True

class ProcessorConfig(BaseModel):
    type: str
    name: str
    cores: int
    frequency: float
    memory: float
    tdp: float

    class Config:
        from_attributes = True

class NetworkConfig(BaseModel):
    type: str
    bandwidth: float
    latency: float
    ports: int

    class Config:
        from_attributes = True

# Define ChipConfig before SystemConfig
class ChipConfig(BaseModel):
    technology: TechnologyConfig
    voltage: float
    memory_levels: int
    memory: Dict[str, MemoryConfig]
    mm_compute: ComputeConfig
    rf: Dict[str, float]
    vector_compute: VectorComputeConfig
    force_connectivity: int

    @classmethod
    def get_default_config(cls):
        """Return a default chip configuration"""
        return {
            "technology": {
                "wire_cap": 0.1,
                "sense_amp_time": 100,
                "plogic_node": 7,
                "logic_node": 7
            },
            "voltage": 0.8,
            "memory_levels": 2,
            "memory": {
                "level0": {
                    "class_type": "SRAM",
                    "frequency": 1000,
                    "banks": 16,
                    "read_ports": 2,
                    "write_ports": 2,
                    "width": 32,
                    "size": 1048576,
                    "leakage_power": 0.1
                },
                "level1": {
                    "class_type": "DRAM",
                    "frequency": 3200,
                    "banks": 8,
                    "read_ports": 1,
                    "write_ports": 1,
                    "width": 64,
                    "size": 8589934592,
                    "leakage_power": 0.5
                }
            },
            "mm_compute": {
                "type1": {
                    "class_type": "systolic_array",
                    "frequency": 1000,
                    "size": 256,
                    "N_PE": 256,
                    "area": 2.0,
                    "per_op_energy": 0.1
                },
                "type2": {
                    "class_type": "mac",
                    "frequency": 1000,
                    "size": 128,
                    "N_PE": 128,
                    "Tile": {
                        "TileX": 8,
                        "TileY": 8,
                        "Number": 16
                    }
                }
            },
            "rf": {
                "energy": 0.1,
                "area": 0.5
            },
            "vector_compute": {
                "class_type": "vector",
                "frequency": 1000,
                "size": 128,
                "N_PE": 128
            },
            "force_connectivity": 0
        }

# Now define SystemConfig
class SystemConfig(BaseModel):
    chips: List[ChipConfig] = Field(default_factory=lambda: [ChipConfig(**ChipConfig.get_default_config())])
    processors: List[ProcessorConfig] = Field(default_factory=list)
    networks: List[NetworkConfig] = Field(default_factory=list)
    topology: str = "mesh"

# Finally define ChipRequirements that uses SystemConfig
class ChipRequirements(BaseModel):
    powerBudget: float = Field(..., gt=0, description="Power budget in Watts")
    areaConstraint: float = Field(..., gt=0, description="Area constraint in mm²")
    performanceTarget: float = Field(..., gt=0, description="Performance target in MIPS")
    selectedWorkloads: List[str] = Field(..., description="List of selected workload types")
    optimizationPriority: Optional[str] = Field("balanced", description="Priority for optimization: 'power', 'performance', or 'balanced'")
    workloadTypes: Dict[str, str] = Field(default_factory=dict, description="Mapping of workload names to their types (inference/training)")
    systemConfig: Optional[SystemConfig] = None

class ChipBlock(BaseModel):
    id: str
    type: str
    size: Dict[str, float]
    position: Dict[str, float]
    powerConsumption: float
    performance: float
    utilization: float

class ChipDesign(BaseModel):
    blocks: List[ChipBlock]
    totalPower: float
    totalArea: float
    estimatedPerformance: float
    powerEfficiency: float

@dataclass
class OptimizationState:
    iteration: int
    power: float
    performance: float
    area: float
    design: Dict
    perf_estimation_frames: List[str] = None

class DesignOptimizer:
    def __init__(self, requirements: ChipRequirements):
        self.requirements = requirements
        self.best_design = None
        self.optimization_states = []
        self.iteration = 0
        
        # Ensure default config exists
        if not DEFAULT_CONFIG_FILE.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Default configuration file not found at {DEFAULT_CONFIG_FILE}"
            )
        
        # Convert workload names to graph objects
        self.graph_set = self._prepare_graph_set(requirements.selectedWorkloads)
    
    def _prepare_graph_set(self, workload_types: List[str]) -> List[Any]:
        """Prepare the set of workload graphs for optimization"""
        graphs = []
        logger.info("Preparing workload graphs")
        
        for workload in workload_types:
            if workload == "ResNet-50":
                graphs.append(resnet_50_graph())
            elif workload == "VGG16":
                graphs.append(vggnet_graph())
            elif workload == "BERT":
                graphs.append(bert_graph())
            elif workload == "GPT2":
                graphs.append(gpt2_graph())
            elif workload == "AlexNet":
                graphs.append(alexnet_graph())
            elif workload == "LangModel":
                g1, g2 = langmodel_graph()
                graphs.extend([g1, g2])
            elif workload == "AES-256":
                cfg = CFGBuilder().build_from_file(
                    "aes.py",
                    "nonai_models/aes.py",
                )
                graphs.append(cfg)
            elif workload == "SHA-3":
                cfg = CFGBuilder().build_from_file(
                    "sha3.py", 
                    "nonai_models/sha3.py",
                )
                graphs.append(cfg)
            elif workload == "RSA":
                cfg = CFGBuilder().build_from_file(
                    "rsa.py",
                    "nonai_models/rsa.py", 
                )
                graphs.append(cfg)
            else:
                logger.warning(f"Unknown workload type: {workload}")
                continue
            
            logger.info(f"Added graph for workload: {workload}")
        
        if not graphs:
            raise ValueError("No valid workload graphs could be prepared")
        
        return graphs
    
    def optimize(self, iterations=10):
        """Run optimization using design_runner"""
        try:
            # Configure optimization parameters
            workload_types_map = self.requirements.workloadTypes or {}
        
            for workload in self.requirements.selectedWorkloads:
                is_training = workload_types_map.get(workload) == 'training'

            backprop = is_training
            print_stats = True
            
            # Create stats filename with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = LOGS_DIR / f"stats_{timestamp}_{self.iteration}.txt"
            
            # Initialize variables
            time = None
            energy = None
            area = None
            
            # Run design optimization
            try:
                for graph in self.graph_set:
                    # Check if this is a non-AI workload
                    is_non_ai = isinstance(graph, CFGBuilder().build_from_file.__class__)
                    
                    if is_non_ai:
                        # For non-AI workloads, use parse_graph and get_stats
                        print("Processing non-AI workload...")
                        parse_graph(graph, dse_input=0, dse_given=False, given_bandwidth=1000000)
                        get_stats(graph)
                        # Set default metrics since parse_graph/get_stats don't return values
                        time = [1.0]  # 1 second default
                        energy = [100.0]  # 100 units default
                        area = 100.0  # 100 mm² default
                    else:
                        # For AI workloads, use design_runner
                        print("Processing AI workload...")
                        time, energy, area = design_runner(
                            graph_set=[graph],
                            backprop=backprop,
                            print_stats=print_stats,
                            stats_file=str(stats_file)
                        )
            except Exception as e:
                logger.error(f"Design optimization failed: {str(e)}")
                # Set default values if optimization fails
                time = [1.0]  # 1 second default
                energy = [100.0]  # 100 units default
                area = 100.0  # 100 mm² default
            
            # Generate performance estimation visualization
            perf_frames = []
            for graph in self.graph_set:
                mapper = Mapper(hwfile=str(DEFAULT_CONFIG_FILE))
                frames = visualize_performance_estimation(mapper, graph, backprop)
                perf_frames.extend(frames)
            
            # Store optimization state
            state = OptimizationState(
                iteration=self.iteration,
                power=energy[0],
                performance=1/time[0],
                area=area,
                design=self._create_design_from_metrics(time, energy, area),
                perf_estimation_frames=perf_frames
            )
            self.optimization_states.append(state)
            
            # Update best design if needed
            current_design = ChipDesign(**state.design)
            if self.best_design is None or self._is_better_design(current_design):
                self.best_design = current_design
            
            self.iteration += 1
            
            return self.best_design
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Optimization failed: {str(e)}"
            )
    
    def _create_design_from_metrics(self, time: List[float], energy: List[float], area: float) -> Dict:
        """Convert optimization metrics to ChipDesign format"""
        # Calculate block sizes based on area distribution
        total_area = area
        block_width = math.sqrt(total_area)
        
        blocks = []
        
        # Add compute block
        blocks.append(ChipBlock(
            id="compute",
            type="Computing",
            size={"width": block_width * 0.5, "height": block_width * 0.5},
            position={"x": 0, "y": 0},
            powerConsumption=energy[0] * 0.6,  # 60% of total power
            performance=1/time[0],  # Convert time to performance
            utilization=0.85
        ))
        
        # Add memory block if ML workload present
        if "Machine Learning" in self.requirements.selectedWorkloads:
            blocks.append(ChipBlock(
                id="memory",
                type="Memory",
                size={"width": block_width * 0.5, "height": block_width * 0.3},
                position={"x": block_width * 0.5, "y": 0},
                powerConsumption=energy[0] * 0.4,  # 40% of total power
                performance=1/time[0] * 0.8,  # 80% of compute performance
                utilization=0.75
            ))
        
        return {
            "blocks": [block.dict() for block in blocks],
            "totalPower": energy[0],
            "totalArea": area,
            "estimatedPerformance": 1/time[0],
            "powerEfficiency": 1/(time[0] * energy[0])
        }
    
    def _is_better_design(self, design: ChipDesign) -> bool:
        """Evaluate if new design is better based on optimization priority"""
        if not self.best_design:
            return True
            
        priority = self.requirements.optimizationPriority
        
        if priority == "power":
            return design.powerEfficiency > self.best_design.powerEfficiency
        elif priority == "performance":
            return design.estimatedPerformance > self.best_design.estimatedPerformance
        else:  # balanced
            current_score = (design.powerEfficiency + design.estimatedPerformance) / 2
            best_score = (self.best_design.powerEfficiency + self.best_design.estimatedPerformance) / 2
            return current_score > best_score
    
    def generate_optimization_graph(self) -> str:
        """Generate optimization progress visualization"""
        plt.figure(figsize=(10, 6))
        iterations = [s.iteration for s in self.optimization_states]
        power = [s.power for s in self.optimization_states]
        perf = [s.performance for s in self.optimization_states]
        
        plt.plot(iterations, power, label='Power (W)', marker='o')
        plt.plot(iterations, perf, label='Performance (MIPS)', marker='s')
        
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True)
        
        # Save to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
    
    def generate_animation_frames(self) -> List[str]:
        """Generate frames showing design evolution"""
        frames = []
        for state in self.optimization_states:
            frame = self._generate_design_frame(state.design)
            frames.append(frame)
        return frames
    
    def _generate_design_frame(self, design: Dict) -> str:
        """Generate a single frame visualizing the chip design"""
        plt.figure(figsize=(8, 8))
        
        # Plot blocks
        for block in design['blocks']:
            x = block['position']['x']
            y = block['position']['y']
            w = block['size']['width']
            h = block['size']['height']
            
            color = {
                'Computing': 'lightcoral',
                'Memory': 'lightblue',
                'Network': 'lightgreen',
                'Security': 'plum'
            }.get(block['type'], 'gray')
            
            plt.gca().add_patch(
                plt.Rectangle((x, y), w, h, 
                            facecolor=color,
                            edgecolor='black',
                            alpha=0.7)
            )
            plt.text(x + w/2, y + h/2, block['type'],
                    ha='center', va='center')
        
        plt.xlim(0, 400)
        plt.ylim(0, 400)
        plt.title(f'Chip Design Layout')
        
        # Save to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()

    def get_performance_estimation_animation(self) -> List[str]:
        """Get all performance estimation visualization frames"""
        frames = []
        for state in self.optimization_states:
            if hasattr(state, 'perf_estimation_frames'):
                frames.extend(state.perf_estimation_frames)
        return frames

def calculate_block_metrics(block_type: str, power_budget: float, performance_target: float) -> Dict:
    """Calculate power consumption and performance metrics for a block"""
    base_metrics = {
        "Computing": {"power": 0.3, "perf": 0.4},
        "Memory": {"power": 0.2, "perf": 0.25},
        "Network": {"power": 0.25, "perf": 0.2},
        "Security": {"power": 0.15, "perf": 0.15}
    }
    
    metrics = base_metrics.get(block_type, {"power": 0.1, "perf": 0.1})
    return {
        "power": metrics["power"] * power_budget,
        "performance": metrics["perf"] * performance_target,
        "utilization": 0.7 + (0.2 * random.random())  # Random utilization between 70-90%
    }

# Add new model for history
class DesignHistory(BaseModel):
    id: str
    timestamp: datetime
    requirements: ChipRequirements
    result: ChipDesign
    optimization_data: Dict[str, Any]

# Add in-memory storage (replace with database in production)
design_history: List[DesignHistory] = []

@app.post("/api/generate-chip")
async def generate_chip(requirements: ChipRequirements):
    try:
        logger.info("Starting chip generation with requirements: %s", requirements.dict())
        
        # Ensure system config has at least one chip
        if requirements.systemConfig:
            if not requirements.systemConfig.chips:
                logger.info("No chips in system config, adding default chip")
                requirements.systemConfig.chips = [ChipConfig(**ChipConfig.get_default_config())]
        else:
            logger.info("No system config provided, creating default configuration")
            requirements.systemConfig = SystemConfig(
                chips=[ChipConfig(**ChipConfig.get_default_config())],
                processors=[{
                    'type': 'cpu',
                    'name': 'Host CPU',
                    'cores': 64,
                    'frequency': 3000.0,
                    'memory': 256.0,
                    'tdp': 280.0
                },
                {
                    'type': 'gpu',
                    'name': 'GPU Accelerator',
                    'cores': 6912,
                    'frequency': 1800.0,
                    'memory': 48.0,
                    'tdp': 350.0
                }],
                networks=[{
                    'type': 'pcie',
                    'bandwidth': 64.0,
                    'latency': 500.0,
                    'ports': 64
                },
                {
                    'type': 'nvlink',
                    'bandwidth': 300.0,
                    'latency': 100.0,
                    'ports': 12
                }],
                topology="mesh"
            )

        # Initialize optimizer with the requirements
        logger.info("Initializing DesignOptimizer")
        optimizer = DesignOptimizer(requirements)
        
        logger.info("Starting optimization process")
        best_design = optimizer.optimize(iterations=10)
        
        logger.info("Generating visualization data")
        # Get performance estimation frames
        perf_frames = optimizer.get_performance_estimation_animation()
        
        # Generate optimization visualization
        optimization_graph = optimizer.generate_optimization_graph()
        
        # Generate design evolution animation
        design_frames = optimizer.generate_animation_frames()
        
        optimization_data = {
            "graph": optimization_graph,
            "animation_frames": design_frames,
            "performance_estimation": {
                "frames": perf_frames,
                "frameCount": len(perf_frames)
            }
        }
        
        # Store in history
        history_entry = DesignHistory(
            id=str(uuid4()),
            timestamp=datetime.now(),
            requirements=requirements,
            result=best_design,
            optimization_data=optimization_data
        )
        design_history.append(history_entry)
        
        logger.info("Chip generation completed successfully")
        app.state.last_optimization = optimization_data
        return best_design
    
    except Exception as e:
        logger.error("Error in generate_chip: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to generate chip design",
                "error": str(e),
                "type": type(e).__name__
            }
        )

@app.get("/api/optimization-results")
async def get_optimization_results():
    """Get the visualization data from the last optimization run"""
    if not hasattr(app.state, 'last_optimization'):
        raise HTTPException(status_code=404, detail="No optimization results available")
    
    return app.state.last_optimization

@app.get("/api/performance-estimation")
async def get_performance_estimation():
    """Get the performance estimation visualization frames"""
    if not hasattr(app.state, 'last_optimization'):
        raise HTTPException(status_code=404, detail="No optimization results available")
    
    optimizer = app.state.last_optimization
    frames = optimizer["performance_estimation_frames"]
    
    return {
        "frames": frames,
        "frameCount": len(frames)
    }

@app.get("/api/workload-templates")
async def get_workload_templates():
    """Get predefined workload templates for common use cases"""
    return {
        "templates": [
            {
                "name": "AI Accelerator",
                "workloads": ["Machine Learning", "Data Analytics"],
                "recommendedPower": 150,
                "recommendedArea": 150,
                "recommendedPerformance": 1500
            },
            {
                "name": "Network Processor",
                "workloads": ["Network Processing", "Cryptography"],
                "recommendedPower": 100,
                "recommendedArea": 120,
                "recommendedPerformance": 1200
            },
            {
                "name": "Image Processor",
                "workloads": ["Image Processing", "Machine Learning"],
                "recommendedPower": 120,
                "recommendedArea": 130,
                "recommendedPerformance": 1300
            }
        ]
    }

class ChipSpecs(BaseModel):
    clockSpeed: float = Field(..., gt=0, description="Clock speed in MHz")
    coreCount: int = Field(..., gt=0, description="Number of cores")
    cacheSize: float = Field(..., gt=0, description="Cache size in MB")
    memoryBandwidth: float = Field(..., gt=0, description="Memory bandwidth in GB/s")

@app.post("/api/estimate-performance")
async def estimate_performance(specs: ChipSpecs):
    # Calculate base MIPS (Million Instructions Per Second)
    # Using a simplified performance model:
    # MIPS = clock_speed * cores * IPC * efficiency_factor
    IPC = 2.5  # Instructions per cycle (typical for modern processors)
    efficiency_factor = 0.8  # Account for various inefficiencies
    
    base_mips = specs.clockSpeed * specs.coreCount * IPC * efficiency_factor
    
    # Apply cache and memory bandwidth effects
    # Cache effect: Larger cache generally improves performance up to a point
    cache_factor = 1.0 + (math.log2(specs.cacheSize) / 10)
    
    # Memory bandwidth effect: Higher bandwidth helps with data-intensive operations
    bandwidth_factor = 1.0 + (math.log2(specs.memoryBandwidth) / 15)
    
    total_mips = base_mips * cache_factor * bandwidth_factor
    
    # Calculate power consumption (in Watts)
    # Simple power model: P = C * V^2 * f
    # Where C is proportional to core count and cache size
    base_power = specs.clockSpeed * specs.coreCount * 0.1  # Basic CPU power
    cache_power = specs.cacheSize * 0.5  # Cache power consumption
    memory_power = specs.memoryBandwidth * 0.05  # Memory system power
    total_power = base_power + cache_power + memory_power
    
    # Calculate power efficiency
    power_efficiency = total_mips / total_power
    
    # Calculate utilization based on memory bandwidth and core count
    max_theoretical_bandwidth = specs.coreCount * specs.clockSpeed * 8  # bytes/s
    utilization = min(95, (specs.memoryBandwidth * 1000 / max_theoretical_bandwidth) * 100)
    
    # Calculate thermal profile (°C)
    # Simple thermal model based on power consumption and core density
    base_temp = 45  # Base temperature
    power_temp_factor = total_power * 0.1
    density_factor = (specs.coreCount / 16) * 10  # Normalized to 16 cores
    thermal_profile = base_temp + power_temp_factor + density_factor
    
    # Generate performance visualization frames
    frames = []
    num_frames = 30
    
    for i in range(num_frames):
        # Create a figure for visualization
        plt.figure(figsize=(10, 6))
        
        # Plot utilization over time (simulated)
        time_points = np.linspace(0, 10, 100)
        base_utilization = utilization + np.random.normal(0, 2, 100)
        utilization_data = base_utilization + 5 * np.sin(time_points + i/5)
        plt.plot(time_points, utilization_data, 'b-', alpha=0.6, label='Core Utilization')
        
        # Plot memory bandwidth usage
        bandwidth_usage = (specs.memoryBandwidth * 0.7) + (specs.memoryBandwidth * 0.3 * np.random.rand(100))
        plt.plot(time_points, bandwidth_usage, 'r-', alpha=0.6, label='Memory Bandwidth')
        
        plt.title(f'Performance Visualization (t={i/10:.1f}s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Utilization / Bandwidth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        frames.append(f"data:image/png;base64,{img_str}")
        plt.close()
    
    return {
        "mips": total_mips,
        "powerEfficiency": power_efficiency,
        "utilizationPercentage": utilization,
        "thermalProfile": thermal_profile,
        "animationFrames": frames
    }

# Add endpoint to get history
@app.get("/api/design-history", response_model=List[DesignHistory])
async def get_design_history():
    return design_history

# Add new endpoint to get a specific design history entry
@app.get("/api/design-history/{design_id}", response_model=DesignHistory)
async def get_design_history_entry(design_id: str):
    for entry in design_history:
        if entry.id == design_id:
            return entry
    raise HTTPException(status_code=404, detail="Design not found")

# Add new models for system configuration
class NetworkConfig(BaseModel):
    type: str
    bandwidth: float
    latency: float
    ports: int

class ProcessorConfig(BaseModel):
    type: str
    name: str
    cores: int
    frequency: float
    memory: float
    tdp: float

class SystemOptimizer:
    def __init__(self, requirements: ChipRequirements):
        self.requirements = requirements
        self.best_design = None
        self.optimization_states = []
        self.iteration = 0
        
        # Convert workload names to graph objects
        self.graph_set = self._prepare_graph_set(requirements.selectedWorkloads)
        
    def _prepare_graph_set(self, workload_types: List[str]) -> List[Any]:
        """
        Convert workload types to corresponding graph objects.
        Uses backprop=True for training mode workloads.
        """
        graphs = []
        
        # Get workload types mapping for training/inference
        workload_types_map = self.requirements.workloadTypes or {}
        
        for workload in workload_types:
            # Check if this is a training workload
            is_training = workload_types_map.get(workload) == 'training'
            
            # Initialize appropriate graph based on workload
            if workload == "ResNet-50":
                graph = resnet_50_graph()
                graphs.append(graph)
            elif workload == "BERT":
                graph = bert_graph()
                graphs.append(graph)
            elif workload == "GPT-4":
                graph = gpt2_graph()  # Using GPT-2 as base for GPT-4
                graphs.append(graph)
            elif workload == "DLRM":
                graph = langmodel_graph()  # Using language model for DLRM
                graphs.append(graph)
            elif workload == "SSD":
                graph = alexnet_graph()  # Using AlexNet as base for SSD
                graphs.append(graph)
            elif workload == "HPCG":
                graph = CFGBuilder().build_from_file(
                    "hpcg.py",
                    "nonai_models/hpcg.py",
                )
                graphs.append(graph)
            elif workload == "LINPACK":
                graph = CFGBuilder().build_from_file(
                    "linpack.py",
                    "nonai_models/linpack.py",
                )
                graphs.append(graph)
            elif workload == "STREAM":
                graph = CFGBuilder().build_from_file(
                    "stream.py",
                    "nonai_models/stream.py",
                )
                graphs.append(graph)
            elif workload == "BFS":
                graph = CFGBuilder().build_from_file(
                    "bfs.py",
                    "nonai_models/bfs.py",
                )
                graphs.append(graph)
            elif workload == "PageRank":
                graph = CFGBuilder().build_from_file(
                    "pagerank.py",
                    "nonai_models/pagerank.py",
                )
                graphs.append(graph)
            elif workload == "Connected Components":
                graph = CFGBuilder().build_from_file(
                    "connected_components.py",
                    "nonai_models/connected_components.py",
                )
                graphs.append(graph)
            elif workload == "AES-256":
                graph = CFGBuilder().build_from_file(
                    "aes.py",
                    "nonai_models/aes.py",
                )
                graphs.append(graph)
            elif workload == "SHA-3":
                graph = CFGBuilder().build_from_file(
                    "sha3.py",
                    "nonai_models/sha3.py",
                )
                graphs.append(graph)
            elif workload == "RSA":
                graph = CFGBuilder().build_from_file(
                    "rsa.py",
                    "nonai_models/rsa.py",
                )
                graphs.append(graph)
            else:
                logger.warning(f"Unknown workload type: {workload}")
        
        return graphs
    
    def optimize(self, iterations=10):
        """Run optimization using design_runner"""
        try:
            # Configure optimization parameters
            workload_types_map = self.requirements.workloadTypes or {}
        
            for workload in self.requirements.selectedWorkloads:
            # Check if this is a training workload
                is_training = workload_types_map.get(workload) == 'training'
            backprop = is_training
            print_stats = True
            
            # Run design optimization
            time, energy, area = design_runner(
                graph_set=self.graph_set,
                backprop=backprop,
                print_stats=print_stats,
                file="default.yaml",
                stats_file=f"logs/stats_{self.iteration}.txt"
            )
            
            # Generate performance estimation visualization
            perf_frames = []
            for graph in self.graph_set:
                mapper = Mapper(hwfile="default.yaml")
                frames = visualize_performance_estimation(mapper, graph, backprop)
                perf_frames.extend(frames)
            
            # Store optimization state
            state = OptimizationState(
                iteration=self.iteration,
                power=energy[0],
                performance=1/time[0],
                area=area,
                design=self._create_design_from_metrics(time, energy, area),
                perf_estimation_frames=perf_frames  # Add frames to state
            )
            self.optimization_states.append(state)
            
            # Update best design if needed
            current_design = ChipDesign(**state.design)
            if self.best_design is None or self._is_better_design(current_design):
                self.best_design = current_design
            
            self.iteration += 1
            
            return self.best_design
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    def _create_design_from_metrics(self, time: List[float], energy: List[float], area: float) -> Dict:
        """Convert optimization metrics to ChipDesign format"""
        # Calculate block sizes based on area distribution
        total_area = area
        block_width = math.sqrt(total_area)
        
        blocks = []
        
        # Add compute block
        blocks.append(ChipBlock(
            id="compute",
            type="Computing",
            size={"width": block_width * 0.5, "height": block_width * 0.5},
            position={"x": 0, "y": 0},
            powerConsumption=energy[0] * 0.6,  # 60% of total power
            performance=1/time[0],  # Convert time to performance
            utilization=0.85
        ))
        
        # Add memory block if ML workload present
        if "Machine Learning" in self.requirements.selectedWorkloads:
            blocks.append(ChipBlock(
                id="memory",
                type="Memory",
                size={"width": block_width * 0.5, "height": block_width * 0.3},
                position={"x": block_width * 0.5, "y": 0},
                powerConsumption=energy[0] * 0.4,  # 40% of total power
                performance=1/time[0] * 0.8,  # 80% of compute performance
                utilization=0.75
            ))
        
        return {
            "blocks": [block.dict() for block in blocks],
            "totalPower": energy[0],
            "totalArea": area,
            "estimatedPerformance": 1/time[0],
            "powerEfficiency": 1/(time[0] * energy[0])
        }
    
    def _is_better_design(self, design: ChipDesign) -> bool:
        """Evaluate if new design is better based on optimization priority"""
        if not self.best_design:
            return True
            
        priority = self.requirements.optimizationPriority
        
        if priority == "power":
            return design.powerEfficiency > self.best_design.powerEfficiency
        elif priority == "performance":
            return design.estimatedPerformance > self.best_design.estimatedPerformance
        else:  # balanced
            current_score = (design.powerEfficiency + design.estimatedPerformance) / 2
            best_score = (self.best_design.powerEfficiency + self.best_design.estimatedPerformance) / 2
            return current_score > best_score
    
    def generate_optimization_graph(self) -> str:
        """Generate optimization progress visualization"""
        plt.figure(figsize=(10, 6))
        iterations = [s.iteration for s in self.optimization_states]
        power = [s.power for s in self.optimization_states]
        perf = [s.performance for s in self.optimization_states]
        
        plt.plot(iterations, power, label='Power (W)', marker='o')
        plt.plot(iterations, perf, label='Performance (MIPS)', marker='s')
        
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True)
        
        # Save to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
    
    def generate_animation_frames(self) -> List[str]:
        """Generate frames showing design evolution"""
        frames = []
        for state in self.optimization_states:
            frame = self._generate_design_frame(state.design)
            frames.append(frame)
        return frames
    
    def _generate_design_frame(self, design: Dict) -> str:
        """Generate a single frame visualizing the chip design"""
        plt.figure(figsize=(8, 8))
        
        # Plot blocks
        for block in design['blocks']:
            x = block['position']['x']
            y = block['position']['y']
            w = block['size']['width']
            h = block['size']['height']
            
            color = {
                'Computing': 'lightcoral',
                'Memory': 'lightblue',
                'Network': 'lightgreen',
                'Security': 'plum'
            }.get(block['type'], 'gray')
            
            plt.gca().add_patch(
                plt.Rectangle((x, y), w, h, 
                            facecolor=color,
                            edgecolor='black',
                            alpha=0.7)
            )
            plt.text(x + w/2, y + h/2, block['type'],
                    ha='center', va='center')
        
        plt.xlim(0, 400)
        plt.ylim(0, 400)
        plt.title(f'Chip Design Layout')
        
        # Save to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()

    def get_performance_estimation_animation(self) -> List[str]:
        """Get all performance estimation visualization frames"""
        frames = []
        for state in self.optimization_states:
            if hasattr(state, 'perf_estimation_frames'):
                frames.extend(state.perf_estimation_frames)
        return frames

# Add new function for system-level performance estimation
def estimate_system_performance(system_config: SystemConfig, workloads: List[str]):
    """Estimate performance for multi-chip system configuration"""
    
    results = {
        'chips': [],
        'network': {
            'bandwidth_utilization': [],
            'latency_distribution': [],
            'bottlenecks': []
        },
        'workload_distribution': {}
    }
    
    # Analyze each chip
    for chip in system_config.chips:
        chip_perf = estimate_chip_performance(chip)
        results['chips'].append(chip_perf)
    
    # Analyze network performance
    for network in system_config.networks:
        util = analyze_network_utilization(network, workloads)
        results['network']['bandwidth_utilization'].append(util)
        
        latency = analyze_network_latency(network, system_config.topology)
        results['network']['latency_distribution'].append(latency)
    
    # Analyze workload distribution
    results['workload_distribution'] = analyze_workload_distribution(
        workloads, 
        system_config.chips,
        system_config.processors
    )
    
    return results

@app.get("/api/default-config")
async def get_default_config():
    """Return the default chip configuration from default.yaml"""
    try:
        with open("default.yaml", "r") as f:
            default_config = f.read()
        return {"config": default_config}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load default configuration: {str(e)}"
        )

# Add new endpoint for system performance calculation
@app.post("/api/calculate-system-performance")
async def calculate_system_performance(
    request: Dict[str, Any]
):
    """Calculate performance metrics for multi-chip system"""
    try:
        system_config = request.get("systemConfig", {})
        workloads = request.get("workloads", [])
        
        # Calculate total system metrics
        total_throughput = 0
        total_power = 0
        processor_metrics = []
        
        # Track chip utilizations
        chip_utils = []
        interconnect_bw = []
        
        # First analyze processors (CPU, GPU, RISC-V)
        for i, processor in enumerate(system_config.get('processors', [])):
            processor_perf = None
            
            if processor.get('type', '').lower() == 'riscv':
                # Calculate RISC-V performance for each workload
                riscv_perf = [estimate_riscv_performance(processor, workload) for workload in workloads]
                processor_perf = {
                    'type': 'riscv',
                    'total_mips': sum(p['mips'] for p in riscv_perf),
                    'avg_utilization': sum(p['utilization'] for p in riscv_perf) / len(riscv_perf) if riscv_perf else 0,
                    'power_efficiency': sum(p['power_efficiency'] for p in riscv_perf) / len(riscv_perf) if riscv_perf else 0,
                    'memory_bandwidth': max((p['memory_bandwidth'] for p in riscv_perf), default=0)
                }
                total_throughput += processor_perf['total_mips'] * 0.001  # Convert to GIPS
                
            elif processor.get('type', '').lower() == 'gpu':
                # Calculate GPU performance for each workload
                gpu_perf = [estimate_gpu_performance(processor, workload) for workload in workloads]
                processor_perf = {
                    'type': 'gpu',
                    'total_tflops': sum(p['tflops'] for p in gpu_perf),
                    'sm_utilization': sum(p['sm_utilization'] for p in gpu_perf) / len(gpu_perf) if gpu_perf else 0,
                    'memory_bandwidth': max((p['memory_bandwidth'] for p in gpu_perf), default=0),
                    'power_efficiency': sum(p['power_efficiency'] for p in gpu_perf) / len(gpu_perf) if gpu_perf else 0
                }
                total_throughput += processor_perf['total_tflops'] * 1000  # Convert to GIPS equivalent
            
            if processor_perf:
                processor_metrics.append({
                    "processorId": f"processor_{i}",
                    "metrics": processor_perf
                })
                total_power += processor.get('tdp', 0)
        
        # Then analyze accelerator chips
        for i, chip in enumerate(system_config.get('chips', [])):
            # Calculate chip utilization directly without mapper
            utilization = calculate_chip_utilization(None, chip, workloads)
            
            # Estimate chip performance based on configuration
            throughput = (chip['mm_compute']['type1'].get('N_PE', 256) * 
                        chip['mm_compute']['type1'].get('frequency', 1000) * 
                        utilization) / 1e6  # Convert to GIPS
            
            power = (chip['memory']['level0'].get('leakage_power', 0.1) +
                    chip['memory']['level1'].get('leakage_power', 0.5)) * utilization
            
            total_throughput += throughput
            total_power += power
            
            chip_utils.append({
                "chipId": f"chip_{i}",
                "utilization": utilization
            })

        # Calculate network metrics
        networks = system_config.get('networks', [])
        system_latency = sum(net.get('latency', 0) for net in networks) / len(networks) if networks else 0
        
        # Calculate interconnect bandwidth utilization
        interconnect_bw = []
        chips = system_config.get('chips', [])
        for i, network in enumerate(networks):
            for j in range(len(chips)):
                for k in range(j + 1, len(chips)):
                    # Calculate actual network utilization between chips
                    utilization = calculate_network_utilization(
                        network_config=network,
                        source_chip=chips[j],
                        dest_chip=chips[k],
                        workloads=workloads
                    )
                    
                    interconnect_bw.append({
                        "source": f"chip_{j}",
                        "destination": f"chip_{k}",
                        "bandwidth": network.get('bandwidth', 0),
                        "utilization": utilization  # Already in percentage
                    })

        # Calculate average network utilization in percentage
        network_utilization = sum(bw['utilization'] for bw in interconnect_bw) / len(interconnect_bw) if interconnect_bw else 0

        return {
            "totalThroughput": total_throughput,
            "systemLatency": system_latency,
            "powerConsumption": total_power,
            "networkUtilization": network_utilization,  # Already in percentage
            "chipUtilizations": chip_utils,
            "interconnectBandwidth": interconnect_bw,
            "processorMetrics": processor_metrics
        }

    except Exception as e:
        logger.error(f"Failed to calculate system performance: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Traceback: {traceback.format_exc()}"
        )

# Add default configurations
defaultProcessors = [
    ProcessorConfig(
        type='cpu',
        name='Host CPU',
        cores=64,
        frequency=3000,
        memory=256,
        tdp=280
    ),
    ProcessorConfig(
        type='gpu',
        name='GPU Accelerator',
        cores=6912,
        frequency=1800,
        memory=48,
        tdp=350
    )
]

defaultNetworks = [
    NetworkConfig(
        type='pcie',
        bandwidth=64,
        latency=500,
        ports=64
    ),
    NetworkConfig(
        type='nvlink',
        bandwidth=300,
        latency=100,
        ports=12
    )
]

def estimate_chip_performance(chip_config):
    """Estimate performance metrics for a single chip configuration"""
    try:
        mapper = Mapper(hwfile="default.yaml")
        mapper.complete_config(chip_config)
        
        time, energy, _, _, area = mapper.save_stats(
            mapper,
            backprop=False,
            memory=get_backprop_memory([]),
            print_stats=False
        )
        
        return {
            'performance': 1/time[0] if time[0] > 0 else 0,
            'power': energy[0],
            'area': area,
            'utilization': random.uniform(0.6, 0.9)  # Simulated for now
        }
    except Exception as e:
        print(f"Error estimating chip performance: {e}")
        return {
            'performance': 0,
            'power': 0,
            'area': 0,
            'utilization': 0
        }

def get_workload_characteristics(workload_name: str, workload_type: Optional[str] = None):
    """Get compute and memory characteristics for specific workloads"""
    base_characteristics = {
        # AI/ML Workloads with inference/training differentiation
        "ResNet-50": {
            "inference": {
                "compute_intensity": 0.85,
                "memory_intensity": 0.6,
                "data_transfer": 400,
                "parallel_friendly": True
            },
            "training": {
                "compute_intensity": 0.90,
                "memory_intensity": 0.75,
                "data_transfer": 800,  # Double for training (weights + gradients)
                "parallel_friendly": True
            }
        },
        "BERT": {
            "inference": {
                "compute_intensity": 0.75,
                "memory_intensity": 0.8,
                "data_transfer": 600,
                "parallel_friendly": True
            },
            "training": {
                "compute_intensity": 0.85,
                "memory_intensity": 0.9,
                "data_transfer": 1200,
                "parallel_friendly": True
            }
        },
        "GPT-4": {
            "inference": {
                "compute_intensity": 0.9,
                "memory_intensity": 0.85,
                "data_transfer": 800,
                "parallel_friendly": True
            },
            "training": {
                "compute_intensity": 0.95,
                "memory_intensity": 0.95,
                "data_transfer": 1600,
                "parallel_friendly": True
            }
        },
        "DLRM": {
            "inference": {
                "compute_intensity": 0.6,
                "memory_intensity": 0.9,
                "data_transfer": 700,
                "parallel_friendly": True
            },
            "training": {
                "compute_intensity": 0.7,
                "memory_intensity": 0.95,
                "data_transfer": 1400,
                "parallel_friendly": True
            }
        },
        "SSD": {
            "inference": {
                "compute_intensity": 0.8,
                "memory_intensity": 0.7,
                "data_transfer": 300,
                "parallel_friendly": True
            },
            "training": {
                "compute_intensity": 0.85,
                "memory_intensity": 0.8,
                "data_transfer": 600,
                "parallel_friendly": True
            }
        },
        
        # Non-AI/ML workloads remain the same
        "HPCG": {
            "compute_intensity": 0.7,
            "memory_intensity": 0.85,
            "data_transfer": 250,
            "parallel_friendly": True
        },
        "LINPACK": {
            "compute_intensity": 0.95,
            "memory_intensity": 0.7,
            "data_transfer": 400,
            "parallel_friendly": True
        },
        "STREAM": {
            "compute_intensity": 0.3,
            "memory_intensity": 0.95,
            "data_transfer": 600,
            "parallel_friendly": True
        },
        "BFS": {
            "compute_intensity": 0.4,
            "memory_intensity": 0.9,
            "data_transfer": 200,
            "parallel_friendly": False
        },
        "PageRank": {
            "compute_intensity": 0.5,
            "memory_intensity": 0.85,
            "data_transfer": 300,
            "parallel_friendly": True
        },
        "Connected Components": {
            "compute_intensity": 0.45,
            "memory_intensity": 0.8,
            "data_transfer": 250,
            "parallel_friendly": False
        },
        "AES-256": {
            "compute_intensity": 0.9,
            "memory_intensity": 0.4,
            "data_transfer": 100,
            "parallel_friendly": True
        },
        "SHA-3": {
            "compute_intensity": 0.85,
            "memory_intensity": 0.3,
            "data_transfer": 80,
            "parallel_friendly": True
        },
        "RSA": {
            "compute_intensity": 0.95,
            "memory_intensity": 0.2,
            "data_transfer": 50,
            "parallel_friendly": False
        }
    }
    
    # Default characteristics for unknown workloads
    default_chars = {
        "compute_intensity": 0.5,
        "memory_intensity": 0.5,
        "data_transfer": 200,
        "parallel_friendly": True
    }
    
    # Handle AI/ML workloads with inference/training modes
    workload_data = base_characteristics.get(workload_name)
    if isinstance(workload_data, dict) and "inference" in workload_data:
        # This is an AI/ML workload
        mode = workload_type if workload_type in ["inference", "training"] else "inference"
        return workload_data[mode]
    
    # Return regular characteristics for non-AI/ML workloads
    return workload_data if workload_data else default_chars

def calculate_chip_utilization(mapper, chip_config, workloads, workload_types=None):
    """Calculate realistic chip utilization based on workload characteristics and hardware capabilities"""
    try:
        workload_types = workload_types or {}
        # Get hardware capabilities
        compute_capacity = (chip_config['mm_compute']['type1'].get('N_PE', 0) * 
                          chip_config['mm_compute']['type1'].get('frequency', 1000))
        memory_bandwidth = (chip_config['memory']['level0'].get('banks', 16) * 
                          chip_config['memory']['level0'].get('width', 32) * 
                          chip_config['memory']['level0'].get('frequency', 1000) / 8)  # Convert to bytes/s
        
        # Calculate actual resource usage
        compute_usage = 0
        memory_usage = 0
        
        for workload in workloads:
            # Get workload type if available
            workload_type = workload_types.get(workload)
            
            # Get specific workload characteristics with type consideration
            chars = get_workload_characteristics(workload, workload_type)
            
            # Account for data dependencies and parallel execution
            dependency_factor = 0.2  # Simplified dependency factor
            
            # Adjust parallel factor based on workload characteristics
            base_parallel_factor = min(1.0, len(workloads) / chip_config['mm_compute']['type1'].get('N_PE', 256))
            parallel_factor = base_parallel_factor if chars['parallel_friendly'] else base_parallel_factor * 0.5
            
            # Calculate resource usage considering workload characteristics
            compute_usage += chars['compute_intensity'] * parallel_factor * (1 - dependency_factor)
            memory_usage += chars['memory_intensity'] * parallel_factor
            
            # Add extra utilization for complementary operations
            if chars['compute_intensity'] > chars['memory_intensity']:
                memory_usage += chars['compute_intensity'] * 0.2  # Memory ops during compute
            else:
                compute_usage += chars['memory_intensity'] * 0.2  # Compute ops during memory access
        
        # Calculate overall utilization considering both compute and memory bottlenecks
        compute_utilization = min(1.0, compute_usage)
        memory_utilization = min(1.0, memory_usage)
        
        # Overall utilization is limited by the more constrained resource
        utilization = min(compute_utilization, memory_utilization)
        
        # Add some realistic variation based on system state
        variation = random.uniform(-0.05, 0.05)  # ±5% variation
        utilization = max(0.1, min(0.95, utilization + variation))  # Clamp between 10% and 95%
        
        return utilization
        
    except Exception as e:
        logger.error(f"Error calculating chip utilization: {str(e)}")
        return 0.5  # Return moderate utilization on error

def calculate_network_utilization(network_config, source_chip, dest_chip, workloads):
    """Calculate realistic network utilization between two chips"""
    try:
        # Get network capabilities
        bandwidth = float(network_config.get('bandwidth', 1))  # GB/s
        ports = int(network_config.get('ports', 1))
        
        # Calculate data movement requirements
        total_data = 0
        max_parallel_transfers = 0
        
        for workload in workloads:
            chars = get_workload_characteristics(workload)
            total_data += chars['data_transfer']
            
            # Track maximum parallel transfers needed
            if chars['parallel_friendly']:
                max_parallel_transfers += 1
        
        # Convert MB to GB
        total_data = total_data / 1024
        
        # Calculate theoretical maximum bandwidth
        max_bandwidth = bandwidth * ports
        
        # Calculate base utilization based on required vs available bandwidth
        # Assume data transfer happens over 1 second intervals
        utilization = total_data / max_bandwidth if max_bandwidth > 0 else 0
        
        # Account for protocol overhead and network contention
        protocol_overhead = {
            'pcie': 0.15,    # PCIe has higher overhead
            'nvlink': 0.08,  # NVLink is more efficient
            'ethernet': 0.2, # Ethernet has highest overhead
            'infinity-fabric': 0.1  # AMD Infinity Fabric
        }.get(str(network_config.get('type', '')).lower(), 0.1)
        
        # Calculate contention based on parallel transfers and available ports
        contention_factor = min(1.0, max_parallel_transfers / ports) if ports > 0 else 0
        
        # Apply network-specific adjustments
        utilization = utilization * (1 + protocol_overhead) * (1 + contention_factor)
        
        # Add realistic variation based on network type
        base_variation = 0.05  # Base 5% variation
        if str(network_config.get('type', '')).lower() == 'ethernet':
            base_variation = 0.1  # More variation for Ethernet
        
        variation = random.uniform(-base_variation, base_variation)
        utilization = max(0.05, min(0.95, utilization + variation))
        
        return utilization * 100  # Convert to percentage
        
    except Exception as e:
        logger.error(f"Error calculating network utilization: {str(e)}")
        return 30  # Return default 30% utilization on error

def estimate_riscv_performance(processor_config, workload):
    """Estimate RISC-V processor performance for a given workload"""
    try:
        # Get workload characteristics
        chars = get_workload_characteristics(workload)
        
        # RISC-V specific parameters
        ipc_base = {  # Instructions Per Cycle baseline for different workload types
            "AI/ML": 1.2,      # Most ML ops vectorized
            "HPC": 1.5,        # Good for numerical computation
            "Graph": 0.8,      # Branch heavy, less predictable
            "Crypto": 1.8      # Dedicated crypto extensions
        }
        
        # Determine workload type
        workload_type = "AI/ML" if workload in ["ResNet-50", "BERT", "GPT-4", "DLRM", "SSD"] else \
                       "HPC" if workload in ["HPCG", "LINPACK", "STREAM"] else \
                       "Graph" if workload in ["BFS", "PageRank", "Connected Components"] else \
                       "Crypto" if workload in ["AES-256", "SHA-3", "RSA"] else "HPC"
        
        # Calculate base IPC
        base_ipc = ipc_base.get(workload_type, 1.0)
        
        # Adjust IPC based on processor configuration
        frequency_ghz = processor_config['frequency'] / 1000  # Convert MHz to GHz
        core_scaling = min(1.0, (processor_config['cores'] / 4) ** 0.7)  # Diminishing returns
        
        # Calculate MIPS (Millions of Instructions Per Second)
        mips = frequency_ghz * base_ipc * processor_config['cores'] * core_scaling * 1000
        
        # Memory impact
        memory_bandwidth_factor = min(1.0, (processor_config['memory'] / 32) ** 0.5)  # Memory size impact
        memory_impact = 1.0 - (chars['memory_intensity'] * (1 - memory_bandwidth_factor))
        
        # Adjust performance based on workload characteristics
        compute_scaling = 1.0 - (0.3 * chars['compute_intensity'])  # RISC-V less efficient for heavy compute
        
        # Final performance metrics
        performance = {
            'mips': mips * memory_impact * compute_scaling,
            'utilization': min(0.95, chars['compute_intensity'] * core_scaling),
            'power_efficiency': mips / processor_config['tdp'],
            'memory_bandwidth': processor_config['memory'] * frequency_ghz * 0.1  # Rough estimate GB/s
        }
        
        return performance
        
    except Exception as e:
        print(f"Error estimating RISC-V performance: {e}")
        return {
            'mips': 0,
            'utilization': 0,
            'power_efficiency': 0,
            'memory_bandwidth': 0
        }

def estimate_gpu_performance(processor_config, workload):
    """Estimate GPU performance for a given workload"""
    try:
        # Get workload characteristics
        chars = get_workload_characteristics(workload)
        
        # GPU architecture parameters
        cuda_cores_per_sm = 128
        sm_count = processor_config['cores'] / cuda_cores_per_sm
        memory_bandwidth = processor_config['memory'] * 14  # Approximate GB/s per GB of memory
        
        # Workload-specific GPU efficiency
        gpu_efficiency = {
            "ResNet-50": 0.85,  # Excellent GPU utilization
            "BERT": 0.80,       # Good for transformer ops
            "GPT-4": 0.82,      # Good for large matrix ops
            "DLRM": 0.70,       # Memory bound for embeddings
            "SSD": 0.75,        # Good for convolutions
            "HPCG": 0.65,       # Limited by memory access
            "LINPACK": 0.90,    # Excellent for dense linear algebra
            "STREAM": 0.95,     # Perfect for memory bandwidth
            "BFS": 0.40,        # Poor for irregular access
            "PageRank": 0.50,   # Limited by graph structure
            "Connected Components": 0.45,  # Graph algorithm limitations
            "AES-256": 0.70,    # Good with dedicated units
            "SHA-3": 0.65,      # Decent parallelization
            "RSA": 0.60         # Limited by sequential parts
        }.get(workload, 0.60)
        
        # Calculate theoretical TFLOPS
        frequency_ghz = processor_config['frequency'] / 1000
        theoretical_tflops = (processor_config['cores'] * 2 * frequency_ghz) / 1000  # FMA = 2 ops
        
        # Actual performance considering efficiency
        achieved_tflops = theoretical_tflops * gpu_efficiency
        
        # Memory bandwidth utilization
        memory_utilization = min(1.0, chars['memory_intensity'] * 1.2)  # GPU memory system more efficient
        effective_bandwidth = memory_bandwidth * memory_utilization
        
        # Calculate SM utilization
        sm_utilization = min(0.95, chars['compute_intensity'] * gpu_efficiency)
        
        # Power efficiency (TFLOPS/W)
        power_efficiency = achieved_tflops / processor_config['tdp']
        
        # Final performance metrics
        performance = {
            'tflops': achieved_tflops,
            'sm_utilization': sm_utilization,
            'memory_bandwidth': effective_bandwidth,
            'power_efficiency': power_efficiency,
            'overall_utilization': min(sm_utilization, memory_utilization)
        }
        
        return performance
        
    except Exception as e:
        print(f"Error estimating GPU performance: {e}")
        return {
            'tflops': 0,
            'sm_utilization': 0,
            'memory_bandwidth': 0,
            'power_efficiency': 0,
            'overall_utilization': 0
        }

class CustomWorkloadKernel(BaseModel):
    """Model for a single computational kernel in a custom workload"""
    name: str = Field(..., description="Name of the kernel")
    code_path: str = Field(..., description="Path to the Python file containing the kernel code")
    compute_intensity: float = Field(..., ge=0, le=1, description="Compute intensity (0-1)")
    memory_intensity: float = Field(..., ge=0, le=1, description="Memory intensity (0-1)")
    data_size: int = Field(..., gt=0, description="Size of data processed in bytes")
    preferred_processor: str = Field("any", description="Preferred processor type (cpu, gpu, accelerator, any)")
    parallelizable: bool = Field(True, description="Whether the kernel can be parallelized")
    dependencies: List[str] = Field(default_factory=list, description="Names of kernels that must complete before this one")

class CustomWorkload(BaseModel):
    """Model for defining a custom workload with multiple kernels"""
    name: str = Field(..., description="Name of the workload")
    description: str = Field(..., description="Description of the workload")
    kernels: List[CustomWorkloadKernel] = Field(..., description="List of computational kernels")
    total_data_size: int = Field(..., gt=0, description="Total size of data processed in bytes")
    target_latency_ms: Optional[float] = Field(None, gt=0, description="Target latency in milliseconds")

class CFGWorkloadAnalyzer:
    """Analyzer for workloads using Control Flow Graph analysis"""
    def __init__(self, code_path: str):
        self.code_path = code_path
        self.cfg = None
        self.basic_blocks = []
        self.execution_paths = []
        
    def build_cfg(self):
        """Build Control Flow Graph from the source code"""
        try:
            self.cfg = CFGBuilder().build_from_file(
                os.path.basename(self.code_path),
                self.code_path
            )
            self._analyze_basic_blocks()
            return True
        except Exception as e:
            logger.error(f"Error building CFG: {str(e)}")
            return False
            
    def _analyze_basic_blocks(self):
        """Analyze basic blocks from the CFG"""
        if not self.cfg:
            return
            
        for block in self.cfg.blocks:
            block_info = self._analyze_block_characteristics(block)
            self.basic_blocks.append(block_info)
            
    def _analyze_block_characteristics(self, block) -> Dict[str, Any]:
        """Analyze characteristics of a basic block"""
        try:
            # Count different types of operations
            compute_ops = 0
            memory_ops = 0
            control_ops = 0
            
            for node in block.nodes:
                if isinstance(node, ast.BinOp):
                    compute_ops += 1
                elif isinstance(node, (ast.Load, ast.Store, ast.Subscript)):
                    memory_ops += 1
                elif isinstance(node, (ast.If, ast.While, ast.For)):
                    control_ops += 1
                    
            total_ops = max(1, compute_ops + memory_ops + control_ops)
            
            return {
                "block_id": id(block),
                "compute_intensity": compute_ops / total_ops,
                "memory_intensity": memory_ops / total_ops,
                "control_intensity": control_ops / total_ops,
                "parallelizable": self._is_block_parallelizable(block),
                "estimated_cycles": self._estimate_block_cycles(block),
                "data_dependencies": self._get_data_dependencies(block)
            }
        except Exception as e:
            logger.error(f"Error analyzing block characteristics: {str(e)}")
            return {
                "block_id": id(block),
                "compute_intensity": 0.33,
                "memory_intensity": 0.33,
                "control_intensity": 0.33,
                "parallelizable": False,
                "estimated_cycles": 100,
                "data_dependencies": []
            }
            
    def _is_block_parallelizable(self, block) -> bool:
        """Determine if a basic block can be parallelized"""
        try:
            # Check for loop constructs that might be parallelizable
            has_loop = any(isinstance(node, (ast.For, ast.While)) for node in block.nodes)
            has_reduction = any(isinstance(node, ast.BinOp) and 
                              isinstance(node.op, (ast.Add, ast.Mult)) for node in block.nodes)
            
            # Check for dependencies that would prevent parallelization
            has_dependencies = bool(self._get_data_dependencies(block))
            
            return has_loop and not has_dependencies
        except Exception as e:
            logger.error(f"Error checking block parallelization: {str(e)}")
            return False
            
    def _estimate_block_cycles(self, block) -> int:
        """Estimate CPU cycles needed for a basic block"""
        try:
            cycles = 0
            for node in block.nodes:
                if isinstance(node, ast.BinOp):
                    cycles += 1  # Basic arithmetic
                elif isinstance(node, ast.Call):
                    cycles += 10  # Function call overhead
                elif isinstance(node, (ast.Load, ast.Store)):
                    cycles += 4  # Memory operations
                elif isinstance(node, (ast.If, ast.While, ast.For)):
                    cycles += 2  # Control flow
            return max(1, cycles)
        except Exception as e:
            logger.error(f"Error estimating block cycles: {str(e)}")
            return 10  # Default cycles
            
    def _get_data_dependencies(self, block) -> List[str]:
        """Get data dependencies for a basic block"""
        try:
            dependencies = set()
            for node in block.nodes:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    dependencies.add(node.id)
            return list(dependencies)
        except Exception as e:
            logger.error(f"Error getting data dependencies: {str(e)}")
            return []
            
    def map_to_processors(self, system_config: SystemConfig) -> Dict[str, Any]:
        """Map basic blocks to available processors"""
        try:
            mapping = {}
            for block in self.basic_blocks:
                best_processor = self._find_best_processor_for_block(block, system_config)
                mapping[block["block_id"]] = {
                    "processor": best_processor,
                    "estimated_execution_time": self._estimate_execution_time(block, best_processor),
                    "characteristics": block
                }
            return mapping
        except Exception as e:
            logger.error(f"Error mapping blocks to processors: {str(e)}")
            return {}
            
    def _find_best_processor_for_block(self, block: Dict[str, Any], system_config: SystemConfig) -> Dict[str, Any]:
        """Find the best processor for a given basic block"""
        try:
            best_processor = None
            best_score = -1
            
            for processor in system_config.processors:
                score = self._calculate_processor_score(block, processor)
                if score > best_score:
                    best_score = score
                    best_processor = processor
                    
            # Also consider accelerator chips
            for chip in system_config.chips:
                if 'mm_compute' in chip:
                    score = self._calculate_accelerator_score(block, chip)
                    if score > best_score:
                        best_score = score
                        best_processor = chip
                        
            return best_processor or system_config.processors[0]  # Default to first processor
        except Exception as e:
            logger.error(f"Error finding best processor: {str(e)}")
            return system_config.processors[0]
            
    def _calculate_processor_score(self, block: Dict[str, Any], processor: Dict[str, Any]) -> float:
        """Calculate score for a processor based on block characteristics"""
        try:
            score = 0
            proc_type = processor.get('type', '').lower()
            
            # CPU is good for control-intensive blocks
            if proc_type == 'cpu':
                score += block['control_intensity'] * 2
                if block['parallelizable'] and processor.get('cores', 1) > 1:
                    score += 1
                    
            # GPU is good for compute-intensive, parallelizable blocks
            elif proc_type == 'gpu':
                if block['parallelizable']:
                    score += block['compute_intensity'] * 3
                if block['memory_intensity'] > 0.7:
                    score += 1
                    
            # RISC-V might be good for simple, sequential blocks
            elif proc_type == 'riscv':
                if not block['parallelizable']:
                    score += 1
                score += (1 - block['compute_intensity']) * 0.5
                
            return score
        except Exception as e:
            logger.error(f"Error calculating processor score: {str(e)}")
            return 0
            
    def _calculate_accelerator_score(self, block: Dict[str, Any], chip: Dict[str, Any]) -> float:
        """Calculate score for an accelerator chip based on block characteristics"""
        try:
            score = 0
            
            # Accelerators are good for compute-intensive, parallelizable blocks
            if block['parallelizable']:
                score += block['compute_intensity'] * 2
                
            # Consider accelerator's compute capabilities
            compute_units = chip['mm_compute']['type1'].get('N_PE', 0)
            frequency = chip['mm_compute']['type1'].get('frequency', 0)
            
            # More compute units and higher frequency increase score
            score += (compute_units / 256) * 0.5
            score += (frequency / 1000) * 0.5
            
            return score
        except Exception as e:
            logger.error(f"Error calculating accelerator score: {str(e)}")
            return 0
            
    def _estimate_execution_time(self, block: Dict[str, Any], processor: Dict[str, Any]) -> float:
        """Estimate execution time for a block on a given processor"""
        try:
            base_cycles = block['estimated_cycles']
            
            # Adjust cycles based on processor type and characteristics
            if processor.get('type', '').lower() == 'gpu':
                if block['parallelizable']:
                    base_cycles /= 32  # Assume 32-way parallelism
            elif processor.get('type', '').lower() == 'cpu':
                if block['parallelizable']:
                    cores = processor.get('cores', 1)
                    base_cycles /= (cores ** 0.7)  # Sub-linear scaling
            elif 'mm_compute' in processor:
                compute_units = processor['mm_compute']['type1'].get('N_PE', 256)
                frequency = processor['mm_compute']['type1'].get('frequency', 1000)
                base_cycles *= (256 / compute_units) * (1000 / frequency)
                
            # Convert cycles to time (ms)
            frequency_mhz = (processor.get('frequency', 1000) 
                           if 'frequency' in processor 
                           else processor.get('mm_compute', {}).get('type1', {}).get('frequency', 1000))
            
            execution_time_ms = (base_cycles / frequency_mhz) * 1000
            return max(0.001, execution_time_ms)  # Minimum 1µs
            
        except Exception as e:
            logger.error(f"Error estimating execution time: {str(e)}")
            return 1.0  # Default 1ms

def analyze_custom_workload_with_cfg(workload: CustomWorkload, system_config: SystemConfig) -> Dict[str, Any]:
    """Analyze a custom workload using CFG analysis"""
    try:
        results = {
            "kernels": {},
            "total_execution_time": 0,
            "processor_utilization": {},
            "bottlenecks": []
        }
        
        for kernel in workload.kernels:
            # Create CFG analyzer for each kernel
            analyzer = CFGWorkloadAnalyzer(kernel.code_path)
            if not analyzer.build_cfg():
                logger.warning(f"Failed to build CFG for kernel: {kernel.name}")
                continue
                
            # Map blocks to processors
            mapping = analyzer.map_to_processors(system_config)
            
            # Calculate kernel metrics
            kernel_time = sum(m["estimated_execution_time"] for m in mapping.values())
            processor_distribution = {}
            
            for block_id, block_mapping in mapping.items():
                proc = block_mapping["processor"]
                proc_id = proc.get('name', str(id(proc)))
                if proc_id not in processor_distribution:
                    processor_distribution[proc_id] = 0
                processor_distribution[proc_id] += block_mapping["estimated_execution_time"]
                
            results["kernels"][kernel.name] = {
                "execution_time_ms": kernel_time,
                "processor_distribution": processor_distribution,
                "block_mapping": mapping
            }
            
            results["total_execution_time"] += kernel_time
            
            # Update processor utilization
            for proc_id, time in processor_distribution.items():
                if proc_id not in results["processor_utilization"]:
                    results["processor_utilization"][proc_id] = 0
                results["processor_utilization"][proc_id] += time
                
            # Identify bottlenecks
            if kernel_time > (workload.target_latency_ms or float('inf')) * 0.2:  # If kernel takes >20% of target time
                results["bottlenecks"].append({
                    "kernel": kernel.name,
                    "execution_time_ms": kernel_time,
                    "cause": "Long execution time"
                })
                
        return results
        
    except Exception as e:
        logger.error(f"Error in CFG-based workload analysis: {str(e)}")
        return {
            "error": str(e),
            "kernels": {},
            "total_execution_time": 0,
            "processor_utilization": {},
            "bottlenecks": []
        }

@app.post("/api/analyze-custom-workload")
async def analyze_custom_workload_endpoint(workload: CustomWorkload, system_config: SystemConfig):
    """Analyze a custom workload using CFG-based analysis"""
    try:
        # Validate workload kernels
        for kernel in workload.kernels:
            if not os.path.exists(kernel.code_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Kernel code file not found: {kernel.code_path}"
                )
        
        # Perform CFG-based analysis
        analysis_results = analyze_custom_workload_with_cfg(workload, system_config)
        
        # Add visualization data
        visualization_data = {
            "timeline": generate_execution_timeline(analysis_results),
            "processor_utilization": generate_utilization_chart(analysis_results),
            "bottleneck_analysis": generate_bottleneck_visualization(analysis_results)
        }
        
        return {
            "analysis": analysis_results,
            "visualization": visualization_data
        }
        
    except Exception as e:
        logger.error(f"Error analyzing custom workload: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze custom workload: {str(e)}"
        )

def generate_execution_timeline(analysis_results: Dict[str, Any]) -> str:
    """Generate a Gantt chart visualization of kernel execution timeline"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Collect timeline data
        kernels = []
        start_times = []
        durations = []
        processors = []
        
        for kernel_name, kernel_data in analysis_results["kernels"].items():
            for proc, time in kernel_data["processor_distribution"].items():
                kernels.append(kernel_name)
                start_times.append(kernel_data["block_mapping"][list(kernel_data["block_mapping"].keys())[0]]["estimated_execution_time"])
                durations.append(time)
                processors.append(proc)
        
        # Create Gantt chart
        y_pos = np.arange(len(kernels))
        plt.barh(y_pos, durations, left=start_times)
        plt.yticks(y_pos, [f"{k} ({p})" for k, p in zip(kernels, processors)])
        
        plt.xlabel('Time (ms)')
        plt.title('Kernel Execution Timeline')
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
        
    except Exception as e:
        logger.error(f"Error generating execution timeline: {str(e)}")
        return ""

def generate_utilization_chart(analysis_results: Dict[str, Any]) -> str:
    """Generate a bar chart of processor utilization"""
    try:
        plt.figure(figsize=(10, 6))
        
        processors = list(analysis_results["processor_utilization"].keys())
        utilizations = list(analysis_results["processor_utilization"].values())
        
        # Calculate utilization percentage
        total_time = analysis_results["total_execution_time"]
        utilization_pct = [100 * u / total_time for u in utilizations]
        
        plt.bar(processors, utilization_pct)
        plt.ylabel('Utilization (%)')
        plt.title('Processor Utilization')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
        
    except Exception as e:
        logger.error(f"Error generating utilization chart: {str(e)}")
        return ""

def generate_bottleneck_visualization(analysis_results: Dict[str, Any]) -> str:
    """Generate a visualization of system bottlenecks"""
    try:
        plt.figure(figsize=(10, 6))
        
        # Collect bottleneck data
        kernels = [b["kernel"] for b in analysis_results["bottlenecks"]]
        times = [b["execution_time_ms"] for b in analysis_results["bottlenecks"]]
        
        if not kernels:
            plt.text(0.5, 0.5, 'No bottlenecks detected', 
                    horizontalalignment='center',
                    verticalalignment='center')
        else:
            plt.bar(kernels, times, color='red', alpha=0.6)
            plt.ylabel('Execution Time (ms)')
            plt.title('Performance Bottlenecks')
            plt.xticks(rotation=45)
            
            # Add threshold line if target latency exists
            for kernel_data in analysis_results["kernels"].values():
                if "target_latency_ms" in kernel_data:
                    plt.axhline(y=kernel_data["target_latency_ms"], 
                              color='green', 
                              linestyle='--', 
                              label='Target Latency')
                    break
        
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
        
    except Exception as e:
        logger.error(f"Error generating bottleneck visualization: {str(e)}")
        return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000) 