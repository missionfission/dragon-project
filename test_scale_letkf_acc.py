import os
import json
import base64
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from scale_letkf_hls import obs_select_hls, matrix_multiply_hls, svd_block_hls

@dataclass
class CustomWorkload:
    """Class to define a custom workload for analysis"""
    name: str
    kernels: List[Dict[str, Any]]
    target_latency_ms: Optional[float] = None
    power_constraint: Optional[float] = None

class SystemConfig:
    """System configuration class"""
    def __init__(self, chips: List[Dict], processors: List[Dict], networks: List[Dict], topology: str):
        self.chips = chips
        self.processors = processors
        self.networks = networks
        self.topology = topology

class ChipRequirements:
    """Chip requirements class"""
    def __init__(self, powerBudget: float, areaConstraint: float, performanceTarget: float,
                 selectedWorkloads: List[str], optimizationPriority: str = "balanced",
                 tech_node: str = "7nm", systemConfig: Optional[SystemConfig] = None):
        self.powerBudget = powerBudget
        self.areaConstraint = areaConstraint
        self.performanceTarget = performanceTarget
        self.selectedWorkloads = selectedWorkloads
        self.optimizationPriority = optimizationPriority
        self.tech_node = tech_node
        self.systemConfig = systemConfig

def analyze_function_offloading(code_path: str, system_config: SystemConfig) -> Dict[str, Any]:
    """Analyze functions for offloading potential"""
    # Create a custom workload for the SCALE-LETKF implementation
    workload = CustomWorkload(
        name="SCALE-LETKF",
        kernels=[
            {
                "name": "obs_select_hls",
                "code_path": code_path,
                "type": "spatial",
                "parallelizable": True,
                "compute_intensity": 0.7
            },
            {
                "name": "matrix_multiply_hls",
                "code_path": code_path,
                "type": "systolic",
                "parallelizable": True,
                "compute_intensity": 0.9
            },
            {
                "name": "svd_block_hls",
                "code_path": code_path,
                "type": "vector",
                "parallelizable": True,
                "compute_intensity": 0.8
            }
        ],
        target_latency_ms=1.0
    )
    
    # Analyze each kernel for offloading
    offload_candidates = {}
    for kernel in workload.kernels:
        # Match kernel to accelerator based on type and characteristics
        best_acc = None
        best_score = 0
        
        for chip in system_config.chips:
            for acc_name, acc_config in chip.get('accelerators', {}).items():
                if kernel['name'] in acc_config.get('target_functions', []):
                    # Calculate offloading score
                    compute_score = min(1.0, kernel['compute_intensity'])
                    parallel_score = 1.0 if kernel['parallelizable'] else 0.0
                    score = 0.6 * compute_score + 0.4 * parallel_score
                    
                    if score > best_score:
                        best_score = score
                        best_acc = {
                            'type': acc_config['type'],
                            'config': acc_config
                        }
        
        if best_acc and best_score > 0.6:
            # Calculate estimated benefits
            compute_units = best_acc['config'].get('compute_units', 1)
            frequency = best_acc['config'].get('frequency', 1000)
            estimated_speedup = min(20.0, (compute_units / 32) * (frequency / 1000))
            energy_savings = min(80.0, kernel['compute_intensity'] * 100)
            
            offload_candidates[kernel['name']] = {
                'score': best_score,
                'recommended_processor': best_acc,
                'estimated_speedup': estimated_speedup,
                'estimated_energy_savings': energy_savings
            }
    
    # Generate visualizations
    visualizations = {
        'offload_benefits': _generate_visualization(offload_candidates, 'benefits'),
        'processor_distribution': _generate_visualization(offload_candidates, 'distribution'),
        'speedup_analysis': _generate_visualization(offload_candidates, 'speedup')
    }
    
    return {
        'offload_candidates': offload_candidates,
        'visualizations': visualizations
    }

def _generate_visualization(data: Dict[str, Any], viz_type: str) -> str:
    """Generate visualization based on offloading data"""
    import matplotlib.pyplot as plt
    import io
    import base64
    
    plt.figure(figsize=(10, 6))
    
    if viz_type == 'benefits':
        functions = list(data.keys())
        scores = [c['score'] * 100 for c in data.values()]  # Convert to percentage
        energy_savings = [c['estimated_energy_savings'] for c in data.values()]
        
        x = range(len(functions))
        plt.bar([i - 0.2 for i in x], scores, width=0.4, label='Offload Score (%)', color='skyblue')
        plt.bar([i + 0.2 for i in x], energy_savings, width=0.4, label='Energy Savings (%)', color='lightgreen')
        plt.xticks(x, functions, rotation=45)
        plt.legend()
        plt.title('Offloading Benefits Analysis')
        
    elif viz_type == 'distribution':
        acc_types = {}
        for info in data.values():
            acc_type = info['recommended_processor']['type']
            acc_types[acc_type] = acc_types.get(acc_type, 0) + 1
        
        plt.pie(acc_types.values(), labels=acc_types.keys(), autopct='%1.1f%%')
        plt.title('Accelerator Distribution')
        
    else:  # speedup
        functions = list(data.keys())
        speedups = [c['estimated_speedup'] for c in data.values()]
        
        plt.bar(functions, speedups, color='lightcoral')
        plt.xticks(rotation=45)
        plt.ylabel('Estimated Speedup (x)')
        plt.title('Performance Speedup Analysis')
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

# Define system configuration with three accelerators
def create_system_config():
    return SystemConfig(
        chips=[{
            'accelerators': {
                'obs_select_acc': {
                    'type': 'spatial',
                    'target_functions': ['obs_select_hls'],
                    'compute_units': 128,
                    'frequency': 800,
                    'local_memory': 32768,
                    'required_characteristics': {
                        'min_compute_intensity': 0.3,
                        'require_parallel': True
                    }
                }
            }
        },
        {
            'accelerators': {
                'matrix_multiply_acc': {
                    'type': 'systolic',
                    'target_functions': ['matrix_multiply_hls'],
                    'compute_units': 256,
                    'frequency': 1000,
                    'local_memory': 65536,
                    'required_characteristics': {
                        'min_compute_intensity': 0.6,
                        'require_parallel': True
                    }
                }
            }
        },
        {
            'accelerators': {
                'svd_acc': {
                    'type': 'vector',
                    'target_functions': ['svd_block_hls'],
                    'compute_units': 192,
                    'frequency': 900,
                    'local_memory': 49152,
                    'required_characteristics': {
                        'min_compute_intensity': 0.5,
                        'require_parallel': True
                    }
                }
            }
        }],
        processors=[{
            'type': 'cpu',
            'name': 'Host CPU',
            'cores': 32,
            'frequency': 2500.0,
            'memory': 128.0,
            'tdp': 125.0
        }],
        networks=[{
            'type': 'pcie',
            'bandwidth': 32.0,
            'latency': 500.0,
            'ports': 32
        }],
        topology="mesh"
    )

def test_accelerator_offloading():
    """Test automatic function offloading to accelerators"""
    # Create system configuration
    system_config = create_system_config()
    
    # Create requirements
    requirements = ChipRequirements(
        powerBudget=300,
        areaConstraint=400,
        performanceTarget=2000,
        selectedWorkloads=["SCALE-LETKF"],
        optimizationPriority="performance",
        tech_node="7nm",
        systemConfig=system_config
    )
    
    # Analyze function offloading
    result = analyze_function_offloading(
        code_path="scale_letkf_hls.py",
        system_config=system_config
    )
    
    # Print offloading results
    print("\nFunction Offloading Analysis:")
    print("-----------------------------")
    for func_name, data in result['offload_candidates'].items():
        print(f"\nFunction: {func_name}")
        print(f"Recommended Accelerator: {data['recommended_processor']['type']}")
        print(f"Estimated Speedup: {data['estimated_speedup']}x")
        print(f"Energy Savings: {data['estimated_energy_savings']}%")
        print(f"Offload Score: {data['score']}")
    
    # Save visualizations
    print("\nSaving visualizations...")
    for name, img_data in result['visualizations'].items():
        if img_data:
            with open(f"{name}.png", "wb") as f:
                f.write(base64.b64decode(img_data))
            print(f"Saved {name}.png")

if __name__ == "__main__":
    test_accelerator_offloading() 