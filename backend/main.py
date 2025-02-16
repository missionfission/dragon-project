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
from src.src_main import design_runner, visualize_performance_estimation
from src.src_main import Mapper

app = FastAPI(
    title="Chip Designer API",
    description="API for generating and managing chip designs",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChipRequirements(BaseModel):
    powerBudget: float = Field(..., gt=0, description="Power budget in Watts")
    areaConstraint: float = Field(..., gt=0, description="Area constraint in mm²")
    performanceTarget: float = Field(..., gt=0, description="Performance target in MIPS")
    selectedWorkloads: List[str] = Field(..., description="List of selected workload types")
    optimizationPriority: Optional[str] = Field("balanced", description="Priority for optimization: 'power', 'performance', or 'balanced'")

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

@app.post("/api/generate-chip", response_model=ChipDesign)
async def generate_chip(requirements: ChipRequirements):
    try:
        optimizer = DesignOptimizer(requirements)
        best_design = optimizer.optimize(iterations=10)
        
        # Store optimization results
        optimization_data = {
            "graph": optimizer.generate_optimization_graph(),
            "animation_frames": optimizer.generate_animation_frames(),
            "performance_estimation_frames": optimizer.get_performance_estimation_animation()
        }
        
        # You might want to store this in a database or cache
        # For now, we'll store it in memory (not recommended for production)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 