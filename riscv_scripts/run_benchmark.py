#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import json
from typing import Dict, Any

class RISCVSimulator:
    """RISC-V simulator wrapper for running benchmarks"""
    
    def __init__(self, spike_path='spike', pk_path='pk'):
        self.spike_path = spike_path
        self.pk_path = pk_path
        self.stats = {}
    
    def compile_benchmark(self, source_path: str) -> str:
        """Compile benchmark to RISC-V binary"""
        binary_path = source_path.replace('.py', '.riscv')
        
        # Use RISC-V GCC to compile
        compile_cmd = [
            'riscv64-unknown-elf-gcc',
            '-O2',
            '-march=rv64gc',
            '-mabi=lp64d',
            '-o', binary_path,
            source_path
        ]
        
        result = subprocess.run(compile_cmd, 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile {source_path}: {result.stderr.decode()}")
        
        return binary_path
    
    def run_simulation(self, binary_path: str) -> Dict[str, Any]:
        """Run binary in Spike RISC-V simulator"""
        # Run with instruction counting enabled
        sim_cmd = [
            self.spike_path,
            '--isa=RV64GC',
            '--ic',  # Enable instruction counting
            '--instret',  # Enable retired instruction counting
            self.pk_path,
            binary_path
        ]
        
        result = subprocess.run(sim_cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed: {result.stderr.decode()}")
        
        # Parse simulation output
        self.stats = self._parse_simulation_output(result.stdout.decode())
        
        return self.stats
    
    def _parse_simulation_output(self, output: str) -> Dict[str, Any]:
        """Parse simulator output to extract statistics"""
        stats = {
            'integer': 0,
            'floating': 0,
            'memory': 0,
            'branch': 0,
            'total_cycles': 0,
            'total_instructions': 0
        }
        
        for line in output.split('\n'):
            if 'integer' in line.lower():
                stats['integer'] = int(line.split()[-1])
            elif 'floating' in line.lower():
                stats['floating'] = int(line.split()[-1])
            elif 'memory' in line.lower():
                stats['memory'] = int(line.split()[-1])
            elif 'branch' in line.lower():
                stats['branch'] = int(line.split()[-1])
            elif 'total cycles' in line.lower():
                stats['total_cycles'] = int(line.split()[-1])
            elif 'total instructions' in line.lower():
                stats['total_instructions'] = int(line.split()[-1])
        
        return stats
    
    def save_stats(self, output_path: str):
        """Save simulation statistics to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.stats, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Run benchmark in RISC-V simulator')
    parser.add_argument('benchmark', help='Path to benchmark source file')
    parser.add_argument('--spike', default='spike', help='Path to Spike simulator')
    parser.add_argument('--pk', default='pk', help='Path to proxy kernel')
    parser.add_argument('--output', default='riscv_results/stats.json',
                       help='Path to save statistics')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.benchmark):
        print(f"Error: Benchmark file {args.benchmark} not found")
        sys.exit(1)
    
    # Initialize simulator
    simulator = RISCVSimulator(args.spike, args.pk)
    
    try:
        # Compile benchmark
        print(f"Compiling {args.benchmark}...")
        binary = simulator.compile_benchmark(args.benchmark)
        
        # Run simulation
        print(f"Running simulation...")
        stats = simulator.run_simulation(binary)
        
        # Save results
        print(f"Saving results to {args.output}...")
        simulator.save_stats(args.output)
        
        print("Simulation completed successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 