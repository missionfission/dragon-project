#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import json
from typing import Dict, Any

class CACTIValidator:
    """Memory hierarchy validator using CACTI"""
    
    def __init__(self, cacti_path='./cacti'):
        self.cacti_path = cacti_path
        self.results = {}
        
    def generate_cacti_config(self, level: str, size: int, associativity: int,
                            block_size: int, tech_node: int) -> str:
        """Generate CACTI configuration file for a memory level"""
        config = f"""
-size (bytes) {size}
-block size (bytes) {block_size}
-associativity {associativity}
-read-write port {1}
-exclusive read port {0}
-exclusive write port {0}
-single ended read ports {0}
-UCA bank count {1}
-technology (u) {tech_node/1000.0}
-cache type "ram"
-tag size (b) "default"
-access mode (normal, sequential, fast) "normal"
-output/input bus width {block_size*8}
-operating temperature (K) 350
-cache model (NUCA, UCA)  "UCA"
-design objective (weight delay, dynamic power, leakage power, cycle time, area) 0:0:0:100:0
"""
        config_path = f'cacti_configs/{level}_cache.cfg'
        os.makedirs('cacti_configs', exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(config)
        
        return config_path
    
    def run_cacti(self, config_path: str) -> Dict[str, Any]:
        """Run CACTI with given configuration"""
        result = subprocess.run([self.cacti_path, '-infile', config_path],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise RuntimeError(f"CACTI failed: {result.stderr.decode()}")
        
        return self._parse_cacti_output(result.stdout.decode())
    
    def _parse_cacti_output(self, output: str) -> Dict[str, Any]:
        """Parse CACTI output to extract relevant metrics"""
        metrics = {
            'access_time': 0.0,  # ns
            'cycle_time': 0.0,   # ns
            'dynamic_power': 0.0, # W
            'leakage_power': 0.0, # W
            'area': 0.0,         # mm^2
            'read_energy': 0.0,  # nJ
            'write_energy': 0.0  # nJ
        }
        
        for line in output.split('\n'):
            if 'Access time' in line:
                metrics['access_time'] = float(line.split(':')[1].split()[0])
            elif 'Cycle time' in line:
                metrics['cycle_time'] = float(line.split(':')[1].split()[0])
            elif 'Dynamic power' in line:
                metrics['dynamic_power'] = float(line.split(':')[1].split()[0])
            elif 'Leakage power' in line:
                metrics['leakage_power'] = float(line.split(':')[1].split()[0])
            elif 'Area' in line and 'mm2' in line:
                metrics['area'] = float(line.split(':')[1].split()[0])
            elif 'Read Energy' in line:
                metrics['read_energy'] = float(line.split(':')[1].split()[0])
            elif 'Write Energy' in line:
                metrics['write_energy'] = float(line.split(':')[1].split()[0])
        
        return metrics
    
    def validate_memory_hierarchy(self, hierarchy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entire memory hierarchy"""
        results = {}
        
        for level, config in hierarchy_config.items():
            # Generate CACTI config
            cacti_config = self.generate_cacti_config(
                level=level,
                size=config['size'],
                associativity=config.get('associativity', 1),
                block_size=config.get('block_size', 64),
                tech_node=config.get('tech_node', 45)
            )
            
            # Run CACTI
            level_results = self.run_cacti(cacti_config)
            results[level] = level_results
        
        self.results = results
        return results
    
    def save_results(self, output_path: str):
        """Save validation results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Validate memory hierarchy using CACTI')
    parser.add_argument('config', help='Memory hierarchy configuration file (JSON)')
    parser.add_argument('--cacti', default='./cacti', help='Path to CACTI executable')
    parser.add_argument('--output', default='cacti_results/validation.json',
                       help='Path to save validation results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found")
        sys.exit(1)
    
    # Load memory hierarchy configuration
    with open(args.config, 'r') as f:
        hierarchy_config = json.load(f)
    
    # Initialize validator
    validator = CACTIValidator(args.cacti)
    
    try:
        # Run validation
        print("Validating memory hierarchy...")
        results = validator.validate_memory_hierarchy(hierarchy_config)
        
        # Save results
        print(f"Saving results to {args.output}...")
        validator.save_results(args.output)
        
        print("Validation completed successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 