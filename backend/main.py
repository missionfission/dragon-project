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
import re

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
    tech_node: Optional[str] = Field(None, description="Technology node for CFG building")

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
        self.accelerator_mapper = None
        self.tech_node = requirements.tech_node if hasattr(requirements, 'tech_node') else '45nm'
        
        if requirements.systemConfig:
            self.accelerator_mapper = AcceleratorFunctionMapper(requirements.systemConfig)
        
        # Convert workload names to graph objects
        self.graph_set = self._prepare_graph_set(requirements.selectedWorkloads)
    
    def _prepare_graph_set(self, workload_types: List[str]) -> List[Any]:
        """Prepare the set of workload graphs for optimization"""
        graphs = []
        logger.info("Preparing workload graphs")
        
        for workload in workload_types:
            # Get the Python file for this workload
            workflow_file = self._get_workflow_file(workload)
            
            if workflow_file and os.path.exists(workflow_file):
                # Analyze workflow for accelerator offloading
                if self.accelerator_mapper:
                    self.accelerator_mapper.analyze_workflow(workflow_file)
                
                # Build CFG for the workflow with technology node
                cfg = CFGBuilder(tech_node=self.tech_node).build_from_file(
                    os.path.basename(workflow_file),
                    workflow_file
                )
                graphs.append(cfg)
            else:
                # Handle built-in workloads
                graph = self._get_builtin_graph(workload)
                if graph:
                    graphs.append(graph)
                else:
                    logger.warning(f"Unknown workload type: {workload}")
        
        if not graphs:
            raise ValueError("No valid workload graphs could be prepared")
        
        return graphs
    
    def _get_workflow_file(self, workload: str) -> Optional[str]:
        """Get the Python file path for a workload"""
        # Map workload names to their Python files
        workflow_files = {
            "CustomWorkload": "workflows/custom_workflow.py",
            "HPCG": "nonai_models/hpcg.py",
            "LINPACK": "nonai_models/linpack.py",
            "STREAM": "nonai_models/stream.py",
            "BFS": "nonai_models/bfs.py",
            "PageRank": "nonai_models/pagerank.py",
            "Connected Components": "nonai_models/connected_components.py",
            "AES-256": "nonai_models/aes.py",
            "SHA-3": "nonai_models/sha3.py",
            "RSA": "nonai_models/rsa.py"
        }
        return workflow_files.get(workload)
    
    def _get_builtin_graph(self, workload: str) -> Optional[Any]:
        """Get built-in graph for standard workloads"""
        if workload == "ResNet-50":
            return resnet_50_graph()
        elif workload == "VGG16":
            return vggnet_graph()
        elif workload == "BERT":
            return bert_graph()
        elif workload == "GPT2":
            return gpt2_graph()
        elif workload == "AlexNet":
            return alexnet_graph()
        elif workload == "LangModel":
            g1, g2 = langmodel_graph()
            return [g1, g2]
        return None
    
    def optimize(self, iterations=10):
        """Run optimization using design_runner"""
        try:
            # Get accelerator offloading plan if available
            offload_plan = None
            if self.accelerator_mapper:
                offload_plan = self.accelerator_mapper.get_offload_plan()
                logger.info(f"Accelerator offloading plan: {offload_plan}")
            
            # Configure optimization parameters
            workload_types_map = self.requirements.workloadTypes or {}
            
            for workload in self.requirements.selectedWorkloads:
                is_training = workload_types_map.get(workload) == 'training'
            
            backprop = is_training
            print_stats = True
            
            # Create stats filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = LOGS_DIR / f"stats_{timestamp}_{self.iteration}.txt"
            
            # Run design optimization with offload plan
            time, energy, area = design_runner(
                graph_set=self.graph_set,
                backprop=backprop,
                print_stats=print_stats,
                file="default.yaml",
                stats_file=str(stats_file),
                offload_plan=offload_plan
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
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

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

class CFGFunctionAnalyzer:
    """Analyzer for function-level offloading decisions using CFG analysis"""
    def __init__(self, code_path: str, function_name: str = None):
        self.code_path = code_path
        self.function_name = function_name
        self.cfg = None
        self.function_cfgs = {}
        self.offload_candidates = {}
        
    def analyze_functions(self):
        """Build and analyze CFGs for all functions in the code"""
        try:
            # Build main CFG
            self.cfg = CFGBuilder().build_from_file(
                os.path.basename(self.code_path),
                self.code_path
            )
            
            # Extract all function definitions
            with open(self.code_path, 'r') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Build CFG for each function
                    func_cfg = self._extract_function_cfg(node)
                    if func_cfg:
                        self.function_cfgs[node.name] = {
                            'cfg': func_cfg,
                            'characteristics': self._analyze_function_characteristics(func_cfg, node)
                        }
            
            return True
        except Exception as e:
            logger.error(f"Error analyzing functions: {str(e)}")
            return False
    
    def _extract_function_cfg(self, func_node: ast.FunctionDef) -> Any:
        """Extract CFG for a specific function"""
        try:
            # Create a new module with just this function
            module = ast.Module(body=[func_node], type_ignores=[])
            code = compile(module, '<string>', 'exec')
            
            # Build CFG for the function
            cfg_builder = CFGBuilder()
            return cfg_builder.build_from_code(code)
        except Exception as e:
            logger.error(f"Error extracting function CFG: {str(e)}")
            return None
    
    def _analyze_function_characteristics(self, cfg: Any, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze characteristics of a function for offloading decisions"""
        try:
            # Count different types of operations
            stats = {
                'compute_ops': 0,
                'memory_ops': 0,
                'control_ops': 0,
                'function_calls': 0,
                'data_size': 0,
                'parallel_loops': 0,
                'sequential_loops': 0
            }
            
            # Analyze function arguments for data size estimation
            for arg in func_node.args.args:
                if hasattr(arg, 'annotation'):
                    stats['data_size'] += self._estimate_data_size(arg.annotation)
            
            # Analyze function body
            for node in ast.walk(func_node):
                if isinstance(node, ast.BinOp):
                    stats['compute_ops'] += 1
                elif isinstance(node, (ast.Load, ast.Store, ast.Subscript)):
                    stats['memory_ops'] += 1
                elif isinstance(node, (ast.If, ast.While)):
                    stats['control_ops'] += 1
                elif isinstance(node, ast.Call):
                    stats['function_calls'] += 1
                elif isinstance(node, ast.For):
                    if self._is_parallel_loop(node):
                        stats['parallel_loops'] += 1
                    else:
                        stats['sequential_loops'] += 1
            
            # Calculate characteristics
            total_ops = max(1, sum(stats.values()))
            return {
                'compute_intensity': stats['compute_ops'] / total_ops,
                'memory_intensity': stats['memory_ops'] / total_ops,
                'control_intensity': stats['control_ops'] / total_ops,
                'parallelizable': stats['parallel_loops'] > 0,
                'data_transfer_size': stats['data_size'],
                'complexity': self._estimate_complexity(stats),
                'estimated_cycles': self._estimate_function_cycles(stats),
                'offload_benefit_score': self._calculate_offload_benefit(stats)
            }
        except Exception as e:
            logger.error(f"Error analyzing function characteristics: {str(e)}")
            return {}
    
    def _estimate_data_size(self, annotation: ast.AST) -> int:
        """Estimate data size from type annotations"""
        try:
            if isinstance(annotation, ast.Subscript):
                if hasattr(annotation, 'value') and isinstance(annotation.value, ast.Name):
                    if annotation.value.id == 'List':
                        return 8 * 1024  # Assume 8KB for lists
                    elif annotation.value.id == 'Array':
                        return 16 * 1024  # Assume 16KB for arrays
            return 1024  # Default 1KB for other types
        except Exception as e:
            logger.error(f"Error estimating data size: {str(e)}")
            return 1024
    
    def _is_parallel_loop(self, node: ast.For) -> bool:
        """Check if a loop can be parallelized"""
        try:
            # Check for dependencies in loop body
            dependencies = set()
            writes = set()
            
            for inner_node in ast.walk(node):
                if isinstance(inner_node, ast.Name):
                    if isinstance(inner_node.ctx, ast.Load):
                        dependencies.add(inner_node.id)
                    elif isinstance(inner_node, ast.Store):
                        writes.add(inner_node.id)
            
            # If there's no overlap between reads and writes, loop might be parallel
            return len(dependencies.intersection(writes)) == 0
        except Exception as e:
            logger.error(f"Error checking loop parallelization: {str(e)}")
            return False
    
    def _estimate_complexity(self, stats: Dict[str, int]) -> str:
        """Estimate computational complexity"""
        if stats['sequential_loops'] > 1:
            return "O(n^2)"
        elif stats['sequential_loops'] == 1:
            return "O(n)"
        else:
            return "O(1)"
    
    def _estimate_function_cycles(self, stats: Dict[str, int]) -> int:
        """Estimate CPU cycles for the function"""
        try:
            cycles = 0
            # Basic operation costs
            cycles += stats['compute_ops'] * 1  # 1 cycle per compute op
            cycles += stats['memory_ops'] * 4   # 4 cycles per memory op
            cycles += stats['control_ops'] * 2  # 2 cycles per control op
            cycles += stats['function_calls'] * 10  # 10 cycles per function call
            
            # Loop overhead
            loop_factor = 100  # Assume average of 100 iterations per loop
            cycles += (stats['sequential_loops'] + stats['parallel_loops']) * loop_factor
            
            return max(1, cycles)
        except Exception as e:
            logger.error(f"Error estimating function cycles: {str(e)}")
            return 100
    
    def _calculate_offload_benefit(self, stats: Dict[str, int]) -> float:
        """Calculate benefit score for offloading (0-1)"""
        try:
            # Factors that make offloading beneficial
            parallel_score = 0.4 if stats['parallel_loops'] > 0 else 0
            compute_score = 0.3 if stats['compute_ops'] > stats['memory_ops'] else 0
            size_score = 0.3 if stats['data_size'] < 32*1024 else 0  # Prefer smaller data transfers
            
            return parallel_score + compute_score + size_score
        except Exception as e:
            logger.error(f"Error calculating offload benefit: {str(e)}")
            return 0.0
    
    def find_offload_candidates(self, system_config: SystemConfig) -> Dict[str, Any]:
        """Find and analyze functions that are good candidates for offloading"""
        try:
            candidates = {}
            
            for func_name, func_data in self.function_cfgs.items():
                characteristics = func_data['characteristics']
                
                # Calculate offloading score
                score = self._calculate_offload_score(characteristics, system_config)
                
                # If score is above threshold, consider for offloading
                if score > 0.6:  # 60% threshold
                    best_processor = self._find_best_processor(characteristics, system_config)
                    estimated_speedup = self._estimate_speedup(characteristics, best_processor)
                    
                    candidates[func_name] = {
                        'score': score,
                        'characteristics': characteristics,
                        'recommended_processor': best_processor,
                        'estimated_speedup': estimated_speedup,
                        'estimated_energy_savings': self._estimate_energy_savings(
                            characteristics, best_processor
                        )
                    }
            
            self.offload_candidates = candidates
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding offload candidates: {str(e)}")
            return {}
    
    def _calculate_offload_score(self, characteristics: Dict[str, Any], system_config: SystemConfig) -> float:
        """Calculate overall score for offloading a function"""
        try:
            # Weights for different factors
            weights = {
                'compute_intensity': 0.3,
                'parallelizable': 0.25,
                'data_transfer': 0.2,
                'complexity': 0.15,
                'benefit_score': 0.1
            }
            
            score = 0
            score += weights['compute_intensity'] * characteristics['compute_intensity']
            score += weights['parallelizable'] * (1.0 if characteristics['parallelizable'] else 0.0)
            score += weights['data_transfer'] * (1.0 - min(1.0, characteristics['data_transfer_size'] / (1024*1024)))
            score += weights['complexity'] * (1.0 if characteristics['complexity'] in ['O(n)', 'O(n^2)'] else 0.0)
            score += weights['benefit_score'] * characteristics['offload_benefit_score']
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating offload score: {str(e)}")
            return 0.0
    
    def _find_best_processor(self, characteristics: Dict[str, Any], system_config: SystemConfig) -> Dict[str, Any]:
        """Find the best processor for offloading a function"""
        try:
            best_processor = None
            best_score = -1
            
            # Check accelerator chips
            for chip in system_config.chips:
                if 'mm_compute' in chip:
                    score = self._calculate_accelerator_score(characteristics, chip)
                    if score > best_score:
                        best_score = score
                        best_processor = {
                            'type': 'accelerator',
                            'config': chip
                        }
            
            # Check other processors
            for processor in system_config.processors:
                score = self._calculate_processor_score(characteristics, processor)
                if score > best_score:
                    best_score = score
                    best_processor = {
                        'type': processor.get('type', 'unknown'),
                        'config': processor
                    }
            
            return best_processor
            
        except Exception as e:
            logger.error(f"Error finding best processor: {str(e)}")
            return None
    
    def _calculate_accelerator_score(self, characteristics: Dict[str, Any], chip: Dict[str, Any]) -> float:
        """Calculate score for an accelerator chip"""
        try:
            score = 0
            
            # Accelerators are good for compute-intensive, parallel work
            if characteristics['parallelizable']:
                score += characteristics['compute_intensity'] * 2
            
            # Consider accelerator capabilities
            compute_units = chip['mm_compute']['type1'].get('N_PE', 0)
            frequency = chip['mm_compute']['type1'].get('frequency', 0)
            
            score += (compute_units / 256) * 0.5
            score += (frequency / 1000) * 0.5
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating accelerator score: {str(e)}")
            return 0.0
    
    def _calculate_processor_score(self, characteristics: Dict[str, Any], processor: Dict[str, Any]) -> float:
        """Calculate score for a processor"""
        try:
            score = 0
            proc_type = processor.get('type', '').lower()
            
            if proc_type == 'gpu':
                if characteristics['parallelizable']:
                    score += characteristics['compute_intensity'] * 3
                if characteristics['memory_intensity'] > 0.7:
                    score += 1
            elif proc_type == 'cpu':
                score += characteristics['control_intensity'] * 2
                if characteristics['parallelizable'] and processor.get('cores', 1) > 1:
                    score += 1
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating processor score: {str(e)}")
            return 0.0
    
    def _estimate_speedup(self, characteristics: Dict[str, Any], processor: Dict[str, Any]) -> float:
        """Estimate speedup from offloading"""
        try:
            if not processor:
                return 1.0
                
            base_cycles = characteristics['estimated_cycles']
            
            if processor['type'] == 'accelerator':
                compute_units = processor['config']['mm_compute']['type1'].get('N_PE', 256)
                frequency_ratio = processor['config']['mm_compute']['type1'].get('frequency', 1000) / 1000
                return min(20.0, (compute_units / 32) * frequency_ratio)
                
            elif processor['type'] == 'gpu':
                if characteristics['parallelizable']:
                    return min(15.0, processor['config'].get('cores', 1000) / 100)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error estimating speedup: {str(e)}")
            return 1.0
    
    def _estimate_energy_savings(self, characteristics: Dict[str, Any], processor: Dict[str, Any]) -> float:
        """Estimate energy savings percentage from offloading"""
        try:
            if not processor:
                return 0.0
                
            if processor['type'] == 'accelerator':
                # Accelerators are typically more energy efficient
                compute_intensity = characteristics['compute_intensity']
                return min(80.0, compute_intensity * 100)
                
            elif processor['type'] == 'gpu':
                # GPUs might save energy for highly parallel workloads
                if characteristics['parallelizable']:
                    return min(60.0, characteristics['compute_intensity'] * 80)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error estimating energy savings: {str(e)}")
            return 0.0

@app.post("/api/analyze-function-offloading")
async def analyze_function_offloading(
    code_path: str,
    function_name: Optional[str] = None,
    system_config: SystemConfig = None
):
    """Analyze function(s) for potential offloading to accelerators"""
    try:
        # Validate input
        if not os.path.exists(code_path):
            raise HTTPException(
                status_code=400,
                detail=f"Code file not found: {code_path}"
            )
        
        # Create analyzer
        analyzer = CFGFunctionAnalyzer(code_path, function_name)
        
        # Analyze functions
        if not analyzer.analyze_functions():
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze functions"
            )
        
        # Find offload candidates
        candidates = analyzer.find_offload_candidates(system_config)
        
        # Generate visualizations
        visualizations = {
            "offload_benefits": _generate_offload_benefits_chart(candidates),
            "processor_distribution": _generate_processor_distribution_chart(candidates),
            "speedup_analysis": _generate_speedup_analysis_chart(candidates)
        }
        
        return {
            "offload_candidates": candidates,
            "visualizations": visualizations
        }
        
    except Exception as e:
        logger.error(f"Error analyzing function offloading: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze function offloading: {str(e)}"
        )

def _generate_offload_benefits_chart(candidates: Dict[str, Any]) -> str:
    """Generate visualization of offloading benefits"""
    try:
        plt.figure(figsize=(10, 6))
        
        functions = list(candidates.keys())
        scores = [c['score'] for c in candidates.values()]
        energy_savings = [c['estimated_energy_savings'] for c in candidates.values()]
        
        x = np.arange(len(functions))
        width = 0.35
        
        plt.bar(x - width/2, scores, width, label='Offload Score', color='skyblue')
        plt.bar(x + width/2, energy_savings, width, label='Energy Savings (%)', color='lightgreen')
        
        plt.xlabel('Functions')
        plt.ylabel('Score / Percentage')
        plt.title('Offloading Benefits Analysis')
        plt.xticks(x, functions, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
        
    except Exception as e:
        logger.error(f"Error generating offload benefits chart: {str(e)}")
        return ""

def _generate_processor_distribution_chart(candidates: Dict[str, Any]) -> str:
    """Generate visualization of processor distribution"""
    try:
        plt.figure(figsize=(10, 6))
        
        processor_counts = {}
        for candidate in candidates.values():
            proc_type = candidate['recommended_processor']['type']
            processor_counts[proc_type] = processor_counts.get(proc_type, 0) + 1
        
        plt.pie(
            processor_counts.values(),
            labels=processor_counts.keys(),
            autopct='%1.1f%%',
            colors=['lightcoral', 'lightblue', 'lightgreen']
        )
        
        plt.title('Recommended Processor Distribution')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
        
    except Exception as e:
        logger.error(f"Error generating processor distribution chart: {str(e)}")
        return ""

def _generate_speedup_analysis_chart(candidates: Dict[str, Any]) -> str:
    """Generate visualization of expected speedups"""
    try:
        plt.figure(figsize=(10, 6))
        
        functions = list(candidates.keys())
        speedups = [c['estimated_speedup'] for c in candidates.values()]
        
        plt.bar(functions, speedups, color='lightcoral')
        plt.xlabel('Functions')
        plt.ylabel('Estimated Speedup (x)')
        plt.title('Expected Performance Speedup')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
        
    except Exception as e:
        logger.error(f"Error generating speedup analysis chart: {str(e)}")
        return ""

class AcceleratorFunctionMapper:
    """Maps functions to accelerators based on system configuration and CFG analysis"""
    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.function_analyzers = {}
        self.accelerator_mappings = {}
        
    def analyze_workflow(self, workflow_file: str):
        """Analyze workflow file and map functions to accelerators"""
        try:
            # Create CFG analyzer for the workflow
            analyzer = CFGFunctionAnalyzer(workflow_file)
            if not analyzer.analyze_functions():
                logger.error("Failed to analyze workflow functions")
                return
            
            # Store analyzer for future reference
            self.function_analyzers[workflow_file] = analyzer
            
            # Map accelerator-specific functions
            self._map_accelerator_functions(analyzer)
            
        except Exception as e:
            logger.error(f"Error analyzing workflow: {str(e)}")
    
    def _map_accelerator_functions(self, analyzer: CFGFunctionAnalyzer):
        """Map functions to accelerators based on configuration and characteristics"""
        try:
            # Process each chip's accelerator configuration
            for chip_idx, chip in enumerate(self.system_config.chips):
                if 'accelerators' not in chip:
                    continue
                
                for acc_name, acc_config in chip['accelerators'].items():
                    # Get target functions for this accelerator
                    target_functions = acc_config.get('target_functions', [])
                    
                    # Find matching functions in the workflow
                    for func_name, func_data in analyzer.function_cfgs.items():
                        # Check if function matches accelerator targets
                        if self._is_function_match(func_name, func_data, target_functions, acc_config):
                            # Calculate offloading score
                            score = analyzer._calculate_offload_score(
                                func_data['characteristics'],
                                self.system_config
                            )
                            
                            # If score is good enough, map function to accelerator
                            if score > 0.6:  # 60% threshold
                                if func_name not in self.accelerator_mappings:
                                    self.accelerator_mappings[func_name] = {
                                        'accelerator': acc_name,
                                        'chip_index': chip_idx,
                                        'score': score,
                                        'characteristics': func_data['characteristics'],
                                        'estimated_speedup': self._estimate_acc_speedup(
                                            func_data['characteristics'],
                                            acc_config
                                        )
                                    }
                                elif score > self.accelerator_mappings[func_name]['score']:
                                    # Update mapping if better score found
                                    self.accelerator_mappings[func_name].update({
                                        'accelerator': acc_name,
                                        'chip_index': chip_idx,
                                        'score': score
                                    })
            
        except Exception as e:
            logger.error(f"Error mapping accelerator functions: {str(e)}")
    
    def _is_function_match(self, func_name: str, func_data: Dict, target_functions: List[str], acc_config: Dict) -> bool:
        """Check if function matches accelerator requirements"""
        try:
            # Direct name match
            if func_name in target_functions:
                return True
            
            # Pattern match
            if 'function_patterns' in acc_config:
                for pattern in acc_config['function_patterns']:
                    if re.match(pattern, func_name):
                        return True
            
            # Characteristic match
            if 'required_characteristics' in acc_config:
                chars = func_data['characteristics']
                reqs = acc_config['required_characteristics']
                
                # Check compute intensity
                if 'min_compute_intensity' in reqs:
                    if chars['compute_intensity'] < reqs['min_compute_intensity']:
                        return False
                
                # Check parallelizability
                if 'require_parallel' in reqs and reqs['require_parallel']:
                    if not chars['parallelizable']:
                        return False
                
                # Check data size constraints
                if 'max_data_size' in reqs:
                    if chars['data_transfer_size'] > reqs['max_data_size']:
                        return False
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking function match: {str(e)}")
            return False
    
    def _estimate_acc_speedup(self, characteristics: Dict[str, Any], acc_config: Dict[str, Any]) -> float:
        """Estimate speedup for function on specific accelerator"""
        try:
            base_speedup = acc_config.get('base_speedup', 1.0)
            
            # Adjust based on characteristics
            if characteristics['parallelizable']:
                parallel_factor = acc_config.get('parallel_speedup_factor', 1.5)
                base_speedup *= parallel_factor
            
            # Adjust for compute intensity
            compute_factor = 1.0 + (characteristics['compute_intensity'] * 
                                  acc_config.get('compute_scaling_factor', 0.5))
            base_speedup *= compute_factor
            
            # Cap maximum speedup
            max_speedup = acc_config.get('max_speedup', 20.0)
            return min(max_speedup, base_speedup)
            
        except Exception as e:
            logger.error(f"Error estimating accelerator speedup: {str(e)}")
            return 1.0
    
    def get_offload_plan(self) -> Dict[str, Any]:
        """Get the complete offloading plan for all mapped functions"""
        try:
            plan = {
                'mappings': self.accelerator_mappings,
                'statistics': {
                    'total_functions': len(self.accelerator_mappings),
                    'accelerator_distribution': self._get_accelerator_distribution(),
                    'estimated_total_speedup': self._calculate_total_speedup()
                },
                'visualizations': {
                    'distribution': self._generate_distribution_chart(),
                    'speedup': self._generate_speedup_chart()
                }
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating offload plan: {str(e)}")
            return {}
    
    def _get_accelerator_distribution(self) -> Dict[str, int]:
        """Get distribution of functions across accelerators"""
        distribution = {}
        for mapping in self.accelerator_mappings.values():
            acc_name = mapping['accelerator']
            distribution[acc_name] = distribution.get(acc_name, 0) + 1
        return distribution
    
    def _calculate_total_speedup(self) -> float:
        """Calculate estimated total speedup from all offloaded functions"""
        try:
            # Simple geometric mean of speedups
            speedups = [m['estimated_speedup'] for m in self.accelerator_mappings.values()]
            if not speedups:
                return 1.0
            return math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        except Exception as e:
            logger.error(f"Error calculating total speedup: {str(e)}")
            return 1.0
    
    def _generate_distribution_chart(self) -> str:
        """Generate visualization of function distribution across accelerators"""
        try:
            plt.figure(figsize=(10, 6))
            
            distribution = self._get_accelerator_distribution()
            plt.bar(distribution.keys(), distribution.values(), color='lightblue')
            
            plt.xlabel('Accelerators')
            plt.ylabel('Number of Functions')
            plt.title('Function Distribution Across Accelerators')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Error generating distribution chart: {str(e)}")
            return ""
    
    def _generate_speedup_chart(self) -> str:
        """Generate visualization of expected speedups per accelerator"""
        try:
            plt.figure(figsize=(10, 6))
            
            acc_speedups = {}
            for mapping in self.accelerator_mappings.values():
                acc_name = mapping['accelerator']
                if acc_name not in acc_speedups:
                    acc_speedups[acc_name] = []
                acc_speedups[acc_name].append(mapping['estimated_speedup'])
            
            # Calculate average speedup per accelerator
            avg_speedups = {
                acc: sum(speedups) / len(speedups)
                for acc, speedups in acc_speedups.items()
            }
            
            plt.bar(avg_speedups.keys(), avg_speedups.values(), color='lightcoral')
            plt.xlabel('Accelerators')
            plt.ylabel('Average Speedup (x)')
            plt.title('Expected Speedup per Accelerator')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Error generating speedup chart: {str(e)}")
            return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000) 