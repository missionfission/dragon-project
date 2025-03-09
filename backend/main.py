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
from datetime import datetime
from uuid import uuid4
import yaml

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

# First define all the base config models
class TechnologyConfig(BaseModel):
    wire_cap: float
    sense_amp_time: float
    plogic_node: int
    logic_node: int

class MemoryConfig(BaseModel):
    class_type: str = Field(..., alias='class')
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
    class_type: str = Field(..., alias='class')
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

class NetworkConfig(BaseModel):
    type: str
    bandwidth: float
    latency: float
    ports: int

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
                    "class": "SRAM",
                    "frequency": 1000,
                    "banks": 16,
                    "read_ports": 2,
                    "write_ports": 2,
                    "width": 32,
                    "size": 1048576,
                    "leakage_power": 0.1
                },
                "level1": {
                    "class": "DRAM",
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
                    "class": "systolic_array",
                    "frequency": 1000,
                    "size": 256,
                    "N_PE": 256,
                    "area": 2.0,
                    "per_op_energy": 0.1
                },
                "type2": {
                    "class": "mac",
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
                "class": "vector",
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
        
        # Convert workload names to graph objects
        self.graph_set = self._prepare_graph_set(requirements.selectedWorkloads)
        
    def _prepare_graph_set(self, workload_types: List[str]) -> List[Any]:
        """
        Convert workload types to corresponding graph objects
        This is a placeholder - you'll need to implement the actual graph creation
        based on your workload types
        """
        # TODO: Implement actual graph creation based on workload types
        graphs = []
        for workload in workload_types:
            if workload == "Machine Learning":
                # Create ML graph
                pass
            elif workload == "Image Processing":
                # Create image processing graph
                pass
            elif workload == "Network Processing":
                # Create network processing graph
                pass
            # Add more workload types as needed
        return graphs
    
    def optimize(self, iterations=10):
        """Run optimization using design_runner"""
        try:
            # Configure optimization parameters
            backprop = self.requirements.optimizationPriority == "performance"
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
        # Ensure system config has at least one chip
        if requirements.systemConfig:
            if not requirements.systemConfig.chips:
                requirements.systemConfig.chips = [ChipConfig(**ChipConfig.get_default_config())]
        else:
            # Create default system config with one chip
            requirements.systemConfig = SystemConfig(
                chips=[ChipConfig(**ChipConfig.get_default_config())],
                processors=defaultProcessors,
                networks=defaultNetworks,
                topology="mesh"
            )

        # Initialize system-level optimizer
        optimizer = SystemOptimizer(requirements)
        best_design = optimizer.optimize(iterations=10)
        
        # Get system-level performance estimates
        perf_results = estimate_system_performance(
            requirements.systemConfig,
            requirements.selectedWorkloads
        )
        
        optimization_data = {
            "graph": optimizer.generate_optimization_graph(),
            "animation_frames": optimizer.generate_animation_frames(),
            "performance_estimation": {
                "system": perf_results,
                "visualization": generate_system_visualization(perf_results)
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
        
        app.state.last_optimization = optimization_data
        return best_design
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        Convert workload types to corresponding graph objects
        This is a placeholder - you'll need to implement the actual graph creation
        based on your workload types
        """
        # TODO: Implement actual graph creation based on workload types
        graphs = []
        for workload in workload_types:
            if workload == "Machine Learning":
                # Create ML graph
                pass
            elif workload == "Image Processing":
                # Create image processing graph
                pass
            elif workload == "Network Processing":
                # Create network processing graph
                pass
            # Add more workload types as needed
        return graphs
    
    def optimize(self, iterations=10):
        """Run optimization using design_runner"""
        try:
            # Configure optimization parameters
            backprop = self.requirements.optimizationPriority == "performance"
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
        system_config = request.get("systemConfig")
        workloads = request.get("workloads", [])
        
        # Calculate total system metrics
        total_throughput = 0
        total_power = 0
        
        # Track chip utilizations
        chip_utils = []
        interconnect_bw = []
        
        # Analyze each chip's performance
        for i, chip in enumerate(system_config['chips']):
            # Create mapper instance for this chip
            mapper = Mapper(hwfile="default.yaml")
            mapper.complete_config(chip)
            
            # Get chip performance metrics
            time, energy, _, _, _ = mapper.save_stats(
                mapper,
                backprop=False,
                memory=get_backprop_memory([]),
                print_stats=False
            )
            
            throughput = 1/time[0] if time[0] > 0 else 0
            total_throughput += throughput
            total_power += energy[0]
            
            # Calculate realistic chip utilization
            utilization = calculate_chip_utilization(mapper, chip, workloads)
            chip_utils.append({
                "chipId": f"chip_{i}",
                "utilization": utilization
            })

        # Calculate network metrics with realistic utilization
        for i in range(len(system_config['networks'])):
            for j in range(i + 1, len(system_config['chips'])):
                utilization = calculate_network_utilization(
                    system_config['networks'][i],
                    system_config['chips'][i],
                    system_config['chips'][j],
                    workloads
                )
                interconnect_bw.append({
                    "source": f"chip_{i}",
                    "destination": f"chip_{j}", 
                    "bandwidth": system_config['networks'][i]['bandwidth'],
                    "utilization": utilization
                })

        # Calculate system-wide metrics
        system_latency = sum(net['latency'] for net in system_config['networks']) / len(system_config['networks'])
        network_utilization = sum(bw['utilization'] for bw in interconnect_bw) / len(interconnect_bw)

        return {
            "totalThroughput": total_throughput,
            "systemLatency": system_latency,
            "powerConsumption": total_power,
            "networkUtilization": network_utilization,
            "chipUtilizations": chip_utils,
            "interconnectBandwidth": interconnect_bw
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate system performance: {str(e)}"
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

def get_workload_characteristics(workload_name):
    """Get compute and memory characteristics for specific workloads"""
    characteristics = {
        # AI/ML Workloads
        "ResNet-50": {
            "compute_intensity": 0.85,  # Heavy CNN computations
            "memory_intensity": 0.6,    # Regular weight/activation access
            "data_transfer": 400,       # MB per batch (weights + activations)
            "parallel_friendly": True
        },
        "BERT": {
            "compute_intensity": 0.75,  # Transformer computations
            "memory_intensity": 0.8,    # Heavy attention matrix operations
            "data_transfer": 600,       # MB per batch (large attention matrices)
            "parallel_friendly": True
        },
        "GPT-4": {
            "compute_intensity": 0.9,   # Very compute intensive
            "memory_intensity": 0.85,   # Large model parameters
            "data_transfer": 800,       # MB per batch (massive model size)
            "parallel_friendly": True
        },
        "DLRM": {
            "compute_intensity": 0.6,   # Embedding lookups + MLPs
            "memory_intensity": 0.9,    # Heavy embedding table access
            "data_transfer": 700,       # MB per batch (large embedding tables)
            "parallel_friendly": True
        },
        "SSD": {
            "compute_intensity": 0.8,   # Detection + classification
            "memory_intensity": 0.7,    # Feature map processing
            "data_transfer": 300,       # MB per batch
            "parallel_friendly": True
        },
        
        # HPC Workloads
        "HPCG": {
            "compute_intensity": 0.7,   # Sparse matrix operations
            "memory_intensity": 0.85,   # Irregular memory access
            "data_transfer": 250,       # MB per iteration
            "parallel_friendly": True
        },
        "LINPACK": {
            "compute_intensity": 0.95,  # Dense matrix operations
            "memory_intensity": 0.7,    # Regular memory access
            "data_transfer": 400,       # MB per iteration
            "parallel_friendly": True
        },
        "STREAM": {
            "compute_intensity": 0.3,   # Memory benchmark
            "memory_intensity": 0.95,   # Memory bandwidth bound
            "data_transfer": 600,       # MB (large arrays)
            "parallel_friendly": True
        },
        
        # Graph Processing
        "BFS": {
            "compute_intensity": 0.4,   # Simple operations
            "memory_intensity": 0.9,    # Random memory access
            "data_transfer": 200,       # MB (graph structure)
            "parallel_friendly": False
        },
        "PageRank": {
            "compute_intensity": 0.5,   # Iterative calculations
            "memory_intensity": 0.85,   # Graph structure access
            "data_transfer": 300,       # MB (graph + ranks)
            "parallel_friendly": True
        },
        "Connected Components": {
            "compute_intensity": 0.45,  # Graph traversal
            "memory_intensity": 0.8,    # Graph structure access
            "data_transfer": 250,       # MB (graph structure)
            "parallel_friendly": False
        },
        
        # Cryptography
        "AES-256": {
            "compute_intensity": 0.9,   # Heavy encryption rounds
            "memory_intensity": 0.4,    # Small state size
            "data_transfer": 100,       # MB (block cipher)
            "parallel_friendly": True
        },
        "SHA-3": {
            "compute_intensity": 0.85,  # Hash computations
            "memory_intensity": 0.3,    # Small state
            "data_transfer": 80,        # MB (hash state)
            "parallel_friendly": True
        },
        "RSA": {
            "compute_intensity": 0.95,  # Heavy modular arithmetic
            "memory_intensity": 0.2,    # Small key size
            "data_transfer": 50,        # MB (keys + data)
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
    
    return characteristics.get(workload_name, default_chars)

def calculate_chip_utilization(mapper, chip_config, workloads):
    """Calculate realistic chip utilization based on workload characteristics and hardware capabilities"""
    try:
        # Get hardware capabilities
        compute_capacity = chip_config['mm_compute']['type1']['N_PE'] * chip_config['mm_compute']['type1']['frequency']
        memory_bandwidth = (chip_config['memory']['level0']['banks'] * 
                          chip_config['memory']['level0']['width'] * 
                          chip_config['memory']['level0']['frequency'] / 8)  # Convert to bytes/s
        
        # Calculate actual resource usage
        compute_usage = 0
        memory_usage = 0
        
        for workload in workloads:
            # Get specific workload characteristics
            chars = get_workload_characteristics(workload)
            
            # Account for data dependencies and parallel execution
            dependency_factor = min(1.0, mapper.bandwidth_idle_time / mapper.total_cycles)
            
            # Adjust parallel factor based on workload characteristics
            base_parallel_factor = min(1.0, len(workloads) / chip_config['mm_compute']['type1']['N_PE'])
            parallel_factor = base_parallel_factor if chars['parallel_friendly'] else base_parallel_factor * 0.5
            
            # Calculate resource usage considering workload characteristics
            compute_usage += chars['compute_intensity'] * parallel_factor * (1 - dependency_factor)
            memory_usage += chars['memory_intensity'] * parallel_factor
            
            # Add extra utilization for complementary operations
            # (e.g., memory ops during compute, prefetching during memory ops)
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
        print(f"Error calculating chip utilization: {e}")
        return 0.5  # Return moderate utilization on error

def calculate_network_utilization(network_config, source_chip, dest_chip, workloads):
    """Calculate realistic network utilization between two chips"""
    try:
        # Get network capabilities
        bandwidth = network_config['bandwidth']  # GB/s
        ports = network_config['ports']
        
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
        utilization = total_data / max_bandwidth
        
        # Account for protocol overhead and network contention
        protocol_overhead = {
            'pcie': 0.15,    # PCIe has higher overhead
            'nvlink': 0.08,  # NVLink is more efficient
            'ethernet': 0.2, # Ethernet has highest overhead
            'infinity-fabric': 0.1  # AMD Infinity Fabric
        }.get(network_config['type'].lower(), 0.1)
        
        # Calculate contention based on parallel transfers and available ports
        contention_factor = min(1.0, max_parallel_transfers / ports)
        
        # Apply network-specific adjustments
        utilization = utilization * (1 + protocol_overhead) * (1 + contention_factor)
        
        # Add realistic variation based on network type
        base_variation = 0.05  # Base 5% variation
        if network_config['type'].lower() == 'ethernet':
            base_variation = 0.1  # More variation for Ethernet
        
        variation = random.uniform(-base_variation, base_variation)
        utilization = max(0.05, min(0.95, utilization + variation))
        
        return utilization
        
    except Exception as e:
        print(f"Error calculating network utilization: {e}")
        return 0.3  # Return moderate utilization on error

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000) 