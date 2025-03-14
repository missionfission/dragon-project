#!/usr/bin/env python3

import sys
import os
import ast
import argparse
import numpy as np
import time
from tinyfive.machine import machine

# Import our Python-to-RISC-V compiler
try:
    from py2riscv import compile_and_run, RISCVCompiler
except ImportError:
    print("Error: py2riscv.py not found. Make sure it's in the current directory.")
    sys.exit(1)

class PowerAnalyzer:
    """Analyzes power consumption of RISC-V code execution"""
    
    def __init__(self, technology_node=7, frequency=1000):
        """
        Initialize power analyzer
        
        Args:
            technology_node: Process technology in nm (7, 14, 22, 45)
            frequency: Operating frequency in MHz
        """
        self.technology_node = technology_node
        self.frequency = frequency
        
        # Technology scaling factors (simplified)
        self.tech_scaling = {
            7: 1.0,    # 7nm as baseline
            14: 2.0,   # 14nm uses ~2x power of 7nm
            22: 3.5,   # 22nm uses ~3.5x power of 7nm
            45: 6.0,   # 45nm uses ~6x power of 7nm
        }
        
        # Base power parameters (mW) at 7nm, 1GHz
        self.base_dynamic_power = 0.1  # per instruction
        self.base_leakage_power = 5.0  # static leakage
        
        # Instruction energy costs (relative to base ALU op)
        self.instruction_energy = {
            'alu': 1.0,      # Base ALU operations (ADD, SUB, etc.)
            'mul': 3.0,      # Multiplication
            'div': 10.0,     # Division
            'load': 2.0,     # Memory load
            'store': 2.0,    # Memory store
            'branch': 1.5,   # Branch operations
            'jump': 1.5,     # Jump operations
        }
        
        # Simple CPI (Cycles Per Instruction) model
        self.cpi_model = {
            'ADD': 1, 'ADDI': 1, 'SUB': 1, 'AND': 1, 'OR': 1, 'XOR': 1, 'SLT': 1,
            'MUL': 3, 'MULH': 3, 'DIV': 10, 'REM': 10,
            'LW': 2, 'LH': 2, 'LB': 2,
            'SW': 1, 'SH': 1, 'SB': 1,
            'BEQ': 2, 'BNE': 2, 'BLT': 2, 'BGE': 2,
            'JAL': 2, 'JALR': 2
        }
    
    def analyze(self, instruction_counts):
        """
        Analyze power consumption based on instruction counts
        
        Args:
            instruction_counts: Dictionary with counts of different instruction types
        
        Returns:
            Dictionary with power and performance metrics
        """
        # Calculate total cycles
        total_cycles = 0
        for instr, count in instruction_counts.items():
            total_cycles += count * self.cpi_model.get(instr, 1)
        
        # Calculate dynamic power
        dynamic_power = 0
        total_instructions = sum(instruction_counts.values())
        
        # Map instructions to power model categories
        for instr_type, count in instruction_counts.items():
            if instr_type in ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'SLT', 'ADDI']:
                energy_factor = self.instruction_energy['alu']
            elif instr_type in ['MUL', 'MULH']:
                energy_factor = self.instruction_energy['mul']
            elif instr_type in ['DIV', 'REM']:
                energy_factor = self.instruction_energy['div']
            elif instr_type in ['LW', 'LH', 'LB']:
                energy_factor = self.instruction_energy['load']
            elif instr_type in ['SW', 'SH', 'SB']:
                energy_factor = self.instruction_energy['store']
            elif instr_type in ['BEQ', 'BNE', 'BLT', 'BGE']:
                energy_factor = self.instruction_energy['branch']
            elif instr_type in ['JAL', 'JALR']:
                energy_factor = self.instruction_energy['jump']
            else:
                energy_factor = self.instruction_energy['alu']  # Default to ALU
            
            dynamic_power += count * energy_factor * self.base_dynamic_power
        
        # Apply technology scaling
        tech_factor = self.tech_scaling.get(self.technology_node, 1.0)
        dynamic_power *= tech_factor
        
        # Calculate leakage power
        leakage_power = self.base_leakage_power * tech_factor
        
        # Calculate total power
        total_power = dynamic_power + leakage_power
        
        # Calculate execution time and energy
        execution_time_s = total_cycles / (self.frequency * 1e6)  # seconds
        energy_joules = total_power * 1e-3 * execution_time_s  # convert mW to W
        
        return {
            'dynamic_power_mW': dynamic_power,
            'leakage_power_mW': leakage_power,
            'total_power_mW': total_power,
            'total_cycles': total_cycles,
            'total_instructions': total_instructions,
            'execution_time_s': execution_time_s,
            'energy_joules': energy_joules,
            'instructions_per_cycle': total_instructions / max(1, total_cycles),
            'energy_per_instruction_nJ': (energy_joules * 1e9) / max(1, total_instructions)
        }

def analyze_python_file(file_path, mem_size=100000, tech_nodes=None):
    """
    Analyze a Python file by compiling it to RISC-V and measuring performance
    
    Args:
        file_path: Path to the Python file to analyze
        mem_size: Memory size for the RISC-V machine
        tech_nodes: List of technology nodes to analyze (default: [7, 14, 22, 45])
    
    Returns:
        Dictionary with analysis results
    """
    if tech_nodes is None:
        tech_nodes = [7, 14, 22, 45]
    
    print(f"Analyzing Python file: {file_path}")
    
    # Read the Python file
    try:
        with open(file_path, 'r') as f:
            source_code = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    
    # Parse the Python file to get an AST
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return None
    
    # Compile and run the Python code on RISC-V
    print("Compiling and running on RISC-V...")
    try:
        start_time = time.time()
        result = compile_and_run(source_code, mem_size=mem_size)
        end_time = time.time()
        
        # Extract instruction counts
        instruction_counts = result.get('instruction_counts', {})
        total_instructions = sum(instruction_counts.values())
        
        print(f"Compilation and execution completed in {end_time - start_time:.2f} seconds")
        print(f"Total instructions executed: {total_instructions}")
        
        # Print instruction mix
        print("\nInstruction mix:")
        for instr, count in sorted(instruction_counts.items()):
            percentage = (count / total_instructions) * 100 if total_instructions > 0 else 0
            print(f"{instr}: {count} ({percentage:.1f}%)")
        
        # Analyze power consumption for different technology nodes
        power_results = {}
        
        for node in tech_nodes:
            analyzer = PowerAnalyzer(technology_node=node)
            metrics = analyzer.analyze(instruction_counts)
            power_results[node] = metrics
        
        # Print power analysis results
        print("\nPower Analysis Results:")
        for node, metrics in power_results.items():
            print(f"\nTechnology Node: {node}nm")
            print(f"Total Instructions: {metrics['total_instructions']}")
            print(f"Total Cycles: {metrics['total_cycles']}")
            print(f"Execution Time: {metrics['execution_time_s']*1000:.6f} ms")
            print(f"Dynamic Power: {metrics['dynamic_power_mW']:.2f} mW")
            print(f"Leakage Power: {metrics['leakage_power_mW']:.2f} mW")
            print(f"Total Power: {metrics['total_power_mW']:.2f} mW")
            print(f"Energy: {metrics['energy_joules']*1e6:.6f} µJ")
            print(f"Instructions per Cycle: {metrics['instructions_per_cycle']:.2f}")
            print(f"Energy per Instruction: {metrics['energy_per_instruction_nJ']:.2f} nJ")
        
        # Return analysis results
        return {
            'file_path': file_path,
            'execution_time_s': end_time - start_time,
            'instruction_counts': instruction_counts,
            'total_instructions': total_instructions,
            'power_results': power_results,
            'variables': result.get('variables', {})
        }
    
    except Exception as e:
        print(f"Error compiling and running {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_report(analysis_results, output_file=None):
    """
    Generate a detailed report from analysis results
    
    Args:
        analysis_results: Dictionary with analysis results
        output_file: Path to save the report (default: None, print to stdout)
    """
    if analysis_results is None:
        print("No analysis results to report.")
        return
    
    report = []
    report.append("=" * 80)
    report.append(f"RISC-V Performance Analysis Report")
    report.append("=" * 80)
    report.append(f"File: {analysis_results['file_path']}")
    report.append(f"Total Instructions: {analysis_results['total_instructions']}")
    report.append(f"Execution Time: {analysis_results['execution_time_s']:.6f} seconds")
    report.append("")
    
    # Instruction mix
    report.append("Instruction Mix:")
    report.append("-" * 40)
    for instr, count in sorted(analysis_results['instruction_counts'].items()):
        percentage = (count / analysis_results['total_instructions']) * 100 if analysis_results['total_instructions'] > 0 else 0
        report.append(f"{instr}: {count} ({percentage:.1f}%)")
    report.append("")
    
    # Power analysis
    report.append("Power Analysis:")
    report.append("-" * 40)
    report.append(f"{'Technology Node':15} {'Power (mW)':15} {'Energy (µJ)':15} {'Energy/Instr (nJ)':15}")
    report.append("-" * 60)
    
    for node, metrics in sorted(analysis_results['power_results'].items()):
        power = metrics['total_power_mW']
        energy = metrics['energy_joules'] * 1e6
        energy_per_instr = metrics['energy_per_instruction_nJ']
        report.append(f"{node}nm{':':12} {power:15.2f} {energy:15.6f} {energy_per_instr:15.2f}")
    
    report.append("")
    
    # Variables
    if analysis_results.get('variables'):
        report.append("Final Variable Values:")
        report.append("-" * 40)
        for name, value in sorted(analysis_results['variables'].items()):
            report.append(f"{name}: {value}")
    
    # Join report lines
    report_text = "\n".join(report)
    
    # Output report
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        except Exception as e:
            print(f"Error saving report to {output_file}: {e}")
            print(report_text)
    else:
        print(report_text)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze Python code on RISC-V')
    parser.add_argument('file', help='Python file to analyze')
    parser.add_argument('--mem-size', type=int, default=100000, help='Memory size for RISC-V machine')
    parser.add_argument('--tech-nodes', type=int, nargs='+', default=[7, 14, 22, 45], 
                        help='Technology nodes to analyze (in nm)')
    parser.add_argument('--output', help='Output file for the report')
    args = parser.parse_args()
    
    # Analyze the Python file
    results = analyze_python_file(args.file, mem_size=args.mem_size, tech_nodes=args.tech_nodes)
    
    # Generate report
    if results:
        generate_report(results, args.output)

if __name__ == "__main__":
    main() 