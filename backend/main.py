from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import math
import random

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
        blocks = []
        total_power = 0
        total_area = 0
        
        # Scale factors based on optimization priority
        power_scale = 0.8 if requirements.optimizationPriority == "power" else 1.0
        perf_scale = 1.2 if requirements.optimizationPriority == "performance" else 1.0
        
        # Add CPU block (always present)
        cpu_metrics = calculate_block_metrics("Computing", requirements.powerBudget, requirements.performanceTarget)
        cpu_block = {
            "id": "cpu",
            "type": "Computing",
            "size": {"width": 100, "height": 100},
            "position": {"x": 50, "y": 50},
            "powerConsumption": cpu_metrics["power"] * power_scale,
            "performance": cpu_metrics["performance"] * perf_scale,
            "utilization": cpu_metrics["utilization"]
        }
        blocks.append(cpu_block)
        total_power += cpu_block["powerConsumption"]
        total_area += cpu_block["size"]["width"] * cpu_block["size"]["height"]
        
        # Calculate memory size based on performance target and workloads
        memory_scale = 1.0
        if "Machine Learning" in requirements.selectedWorkloads:
            memory_scale *= 1.5
        if "Image Processing" in requirements.selectedWorkloads:
            memory_scale *= 1.3
            
        memory_size = (requirements.performanceTarget / 10) * memory_scale
        
        # Add Memory blocks based on workloads
        if "Machine Learning" in requirements.selectedWorkloads:
            mem_metrics = calculate_block_metrics("Memory", requirements.powerBudget, requirements.performanceTarget)
            ml_mem_block = {
                "id": "memory-ml",
                "type": "Memory",
                "size": {"width": memory_size, "height": 60},
                "position": {"x": 170, "y": 50},
                "powerConsumption": mem_metrics["power"] * power_scale,
                "performance": mem_metrics["performance"] * perf_scale,
                "utilization": mem_metrics["utilization"]
            }
            blocks.append(ml_mem_block)
            total_power += ml_mem_block["powerConsumption"]
            total_area += ml_mem_block["size"]["width"] * ml_mem_block["size"]["height"]
            
        # Add Network block if needed
        if "Network Processing" in requirements.selectedWorkloads:
            network_metrics = calculate_block_metrics("Network", requirements.powerBudget, requirements.performanceTarget)
            network_size = requirements.powerBudget / 5
            network_block = {
                "id": "network",
                "type": "Network",
                "size": {"width": network_size, "height": network_size},
                "position": {"x": 50, "y": 170},
                "powerConsumption": network_metrics["power"] * power_scale,
                "performance": network_metrics["performance"] * perf_scale,
                "utilization": network_metrics["utilization"]
            }
            blocks.append(network_block)
            total_power += network_block["powerConsumption"]
            total_area += network_block["size"]["width"] * network_block["size"]["height"]
            
        # Add Cryptography block if needed
        if "Cryptography" in requirements.selectedWorkloads:
            crypto_metrics = calculate_block_metrics("Security", requirements.powerBudget, requirements.performanceTarget)
            crypto_block = {
                "id": "crypto",
                "type": "Security",
                "size": {"width": 60, "height": 60},
                "position": {"x": 170, "y": 170},
                "powerConsumption": crypto_metrics["power"] * power_scale,
                "performance": crypto_metrics["performance"] * perf_scale,
                "utilization": crypto_metrics["utilization"]
            }
            blocks.append(crypto_block)
            total_power += crypto_block["powerConsumption"]
            total_area += crypto_block["size"]["width"] * crypto_block["size"]["height"]
            
        # Calculate overall metrics
        estimated_performance = sum(block["performance"] for block in blocks)
        power_efficiency = estimated_performance / total_power if total_power > 0 else 0
            
        return ChipDesign(
            blocks=blocks,
            totalPower=total_power,
            totalArea=total_area,
            estimatedPerformance=estimated_performance,
            powerEfficiency=power_efficiency
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 