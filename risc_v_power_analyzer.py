#!/usr/bin/env python3

import numpy as np
import time
import matplotlib.pyplot as plt
from tinyfive.machine import machine

class PowerModel:
    """Simple power model for RISC-V processors"""
    
    def __init__(self, technology_node=7, base_frequency=1000):
        """
        Initialize power model with technology parameters
        
        Args:
            technology_node: Process technology in nm (7, 14, 22, etc.)
            base_frequency: Base frequency in MHz
        """
        self.technology_node = technology_node
        self.base_frequency = base_frequency
        
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
    
    def estimate_power(self, instruction_counts, cycles, frequency=None):
        """
        Estimate power consumption based on instruction mix and frequency
        
        Args:
            instruction_counts: Dictionary with counts of different instruction types
            cycles: Total execution cycles
            frequency: Operating frequency in MHz (defaults to base_frequency)
        
        Returns:
            Dictionary with power estimates
        """
        if frequency is None:
            frequency = self.base_frequency
        
        # Apply frequency scaling (dynamic power scales with frequency)
        freq_scaling = frequency / self.base_frequency
        
        # Apply technology node scaling
        tech_factor = self.tech_scaling.get(self.technology_node, 1.0)
        
        # Calculate dynamic power from instruction mix
        dynamic_power = 0
        total_instructions = sum(instruction_counts.values())
        
        # Map TinyFive instructions to power model categories
        for instr_type, count in instruction_counts.items():
            if instr_type in ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'SLT']:
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
        
        # Scale dynamic power by frequency and technology
        dynamic_power *= freq_scaling * tech_factor
        
        # Calculate leakage power (scaled by technology but not frequency)
        leakage_power = self.base_leakage_power * tech_factor
        
        # Calculate total power
        total_power = dynamic_power + leakage_power
        
        # Calculate energy (power * time)
        execution_time_s = cycles / (frequency * 1e6)  # seconds
        energy_joules = total_power * 1e-3 * execution_time_s  # convert mW to W
        
        return {
            'dynamic_power_mW': dynamic_power,
            'leakage_power_mW': leakage_power,
            'total_power_mW': total_power,
            'execution_time_s': execution_time_s,
            'energy_joules': energy_joules,
            'instructions_per_cycle': total_instructions / max(1, cycles),
            'energy_per_instruction_nJ': (total_power * 1e-3 * execution_time_s * 1e9) / max(1, total_instructions)
        }

def run_matrix_multiply(mem_size=10000, size=4):
    """
    Run matrix multiplication on TinyFive RISC-V emulator
    
    Args:
        mem_size: Memory size for the TinyFive machine
        size: Size of square matrices to multiply
    
    Returns:
        Tuple of (result, stats)
    """
    # Create a new machine instance
    m = machine(mem_size=mem_size)
    
    # Create two random matrices
    A = np.random.randint(0, 10, (size, size)).astype(np.int32)
    B = np.random.randint(0, 10, (size, size)).astype(np.int32)
    
    # Store matrices in memory
    A_addr = 100  # Starting address for matrix A
    B_addr = A_addr + (size * size * 4)  # Starting address for matrix B
    C_addr = B_addr + (size * size * 4)  # Starting address for result matrix C
    
    # Store matrix A in memory using write_i32
    for i in range(size):
        for j in range(size):
            addr = A_addr + (i * size + j) * 4
            m.write_i32(addr, A[i, j])
    
    # Store matrix B in memory using write_i32
    for i in range(size):
        for j in range(size):
            addr = B_addr + (i * size + j) * 4
            m.write_i32(addr, B[i, j])
    
    # Initialize instruction counter
    m.instr_counter = {}
    
    # Matrix multiplication in RISC-V assembly
    # Using upper-case instructions for better performance
    
    # Initialize registers
    m.x[5] = size       # t0 = size (x5 is t0)
    m.x[6] = 0          # t1 = i = 0 (x6 is t1)
    
    # Outer loop (i)
    i_loop_start = len(m.instr)
    m.BGE(6, 5, 7 + size * (7 + size * 7))  # if i >= size, exit i loop
    
    m.x[7] = 0          # t2 = j = 0 (x7 is t2)
    
    # Middle loop (j)
    j_loop_start = len(m.instr)
    m.BGE(7, 5, 5 + size * 7)  # if j >= size, exit j loop
    
    m.x[28] = 0          # t3 = k = 0 (x28 is t3)
    m.x[31] = 0          # t6 = sum = 0 (x31 is t6)
    
    # Inner loop (k)
    k_loop_start = len(m.instr)
    m.BGE(28, 5, 7)   # if k >= size, exit k loop
    
    # Calculate address of A[i][k]
    m.MUL(29, 6, 5)     # t4 = i * size (x29 is t4)
    m.ADD(29, 29, 28)   # t4 = i * size + k
    m.MUL(29, 29, 4)    # t4 = (i * size + k) * 4
    m.ADDI(29, 29, A_addr)  # t4 = A_addr + (i * size + k) * 4
    m.LW(8, 29, 0)      # s0 = A[i][k] (x8 is s0)
    
    # Calculate address of B[k][j]
    m.MUL(30, 28, 5)    # t5 = k * size (x30 is t5)
    m.ADD(30, 30, 7)    # t5 = k * size + j
    m.MUL(30, 30, 4)    # t5 = (k * size + j) * 4
    m.ADDI(30, 30, B_addr)  # t5 = B_addr + (k * size + j) * 4
    m.LW(9, 30, 0)      # s1 = B[k][j] (x9 is s1)
    
    # Multiply and accumulate
    m.MUL(18, 8, 9)     # s2 = A[i][k] * B[k][j] (x18 is s2)
    m.ADD(31, 31, 18)   # sum += A[i][k] * B[k][j]
    
    # Increment k
    m.ADDI(28, 28, 1)   # k++
    m.JAL(0, k_loop_start)  # Jump back to start of k loop (x0 is zero)
    
    # Store result in C[i][j]
    m.MUL(29, 6, 5)     # t4 = i * size
    m.ADD(29, 29, 7)    # t4 = i * size + j
    m.MUL(29, 29, 4)    # t4 = (i * size + j) * 4
    m.ADDI(29, 29, C_addr)  # t4 = C_addr + (i * size + j) * 4
    m.SW(29, 31, 0)     # C[i][j] = sum
    
    # Increment j
    m.ADDI(7, 7, 1)     # j++
    m.JAL(0, j_loop_start)  # Jump back to start of j loop
    
    # Increment i
    m.ADDI(6, 6, 1)     # i++
    m.JAL(0, i_loop_start)  # Jump back to start of i loop
    
    # Execute the program
    start_time = time.time()
    m.exe()
    end_time = time.time()
    
    # Read result matrix C using read_i32
    C = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            addr = C_addr + (i * size + j) * 4
            C[i, j] = m.read_i32(addr)
    
    # Verify result
    expected = A @ B
    is_correct = np.array_equal(C, expected)
    
    # Collect statistics
    stats = {
        'instruction_counts': dict(m.instr_counter),
        'total_instructions': sum(m.instr_counter.values()),
        'execution_time_s': end_time - start_time,
        'is_correct': is_correct
    }
    
    return C, stats

def run_vector_add(mem_size=10000, size=100):
    """
    Run vector addition on TinyFive RISC-V emulator
    
    Args:
        mem_size: Memory size for the TinyFive machine
        size: Size of vectors to add
    
    Returns:
        Tuple of (result, stats)
    """
    # Create a new machine instance
    m = machine(mem_size=mem_size)
    
    # Create two random vectors
    A = np.random.randint(0, 100, size).astype(np.int32)
    B = np.random.randint(0, 100, size).astype(np.int32)
    
    # Store vectors in memory
    A_addr = 100  # Starting address for vector A
    B_addr = A_addr + (size * 4)  # Starting address for vector B
    C_addr = B_addr + (size * 4)  # Starting address for result vector C
    
    # Store vector A in memory using write_i32
    for i in range(size):
        addr = A_addr + i * 4
        m.write_i32(addr, A[i])
    
    # Store vector B in memory using write_i32
    for i in range(size):
        addr = B_addr + i * 4
        m.write_i32(addr, B[i])
    
    # Initialize instruction counter
    m.instr_counter = {}
    
    # Vector addition in RISC-V assembly
    # Using upper-case instructions for better performance
    
    # Initialize registers
    m.x[5] = size       # t0 = size (x5 is t0)
    m.x[6] = 0          # t1 = i = 0 (x6 is t1)
    
    # Loop
    loop_start = len(m.instr)
    m.BGE(6, 5, 10)  # if i >= size, exit loop
    
    # Calculate address of A[i]
    m.MUL(7, 6, 4)      # t2 = i * 4 (x7 is t2)
    m.ADDI(7, 7, A_addr)  # t2 = A_addr + i * 4
    m.LW(8, 7, 0)       # s0 = A[i] (x8 is s0)
    
    # Calculate address of B[i]
    m.MUL(28, 6, 4)     # t3 = i * 4 (x28 is t3)
    m.ADDI(28, 28, B_addr)  # t3 = B_addr + i * 4
    m.LW(9, 28, 0)      # s1 = B[i] (x9 is s1)
    
    # Add and store in C[i]
    m.ADD(18, 8, 9)     # s2 = A[i] + B[i] (x18 is s2)
    m.MUL(29, 6, 4)     # t4 = i * 4 (x29 is t4)
    m.ADDI(29, 29, C_addr)  # t4 = C_addr + i * 4
    m.SW(29, 18, 0)     # C[i] = A[i] + B[i]
    
    # Increment i
    m.ADDI(6, 6, 1)     # i++
    m.JAL(0, loop_start)  # Jump back to start of loop (x0 is zero)
    
    # Execute the program
    start_time = time.time()
    m.exe()
    end_time = time.time()
    
    # Read result vector C using read_i32
    C = np.zeros(size, dtype=np.int32)
    for i in range(size):
        addr = C_addr + i * 4
        C[i] = m.read_i32(addr)
    
    # Verify result
    expected = A + B
    is_correct = np.array_equal(C, expected)
    
    # Collect statistics
    stats = {
        'instruction_counts': dict(m.instr_counter),
        'total_instructions': sum(m.instr_counter.values()),
        'execution_time_s': end_time - start_time,
        'is_correct': is_correct
    }
    
    return C, stats

def analyze_workload(workload_name, workload_func, power_model, **kwargs):
    """
    Analyze a workload using TinyFive and power model
    
    Args:
        workload_name: Name of the workload
        workload_func: Function that runs the workload
        power_model: PowerModel instance
        **kwargs: Additional arguments to pass to workload_func
    
    Returns:
        Dictionary with analysis results
    """
    print(f"Running {workload_name}...")
    
    # Run workload
    result, stats = workload_func(**kwargs)
    
    # Calculate cycles (estimate based on instruction count)
    # In a real processor, this would be more complex due to pipelining, etc.
    instruction_counts = stats['instruction_counts']
    total_instructions = stats['total_instructions']
    
    # Simple CPI (Cycles Per Instruction) model
    cpi_model = {
        'ADD': 1, 'ADDI': 1, 'SUB': 1, 'AND': 1, 'OR': 1, 'XOR': 1, 'SLT': 1,
        'MUL': 3, 'MULH': 3, 'DIV': 10, 'REM': 10,
        'LW': 2, 'LH': 2, 'LB': 2,
        'SW': 1, 'SH': 1, 'SB': 1,
        'BEQ': 2, 'BNE': 2, 'BLT': 2, 'BGE': 2,
        'JAL': 2, 'JALR': 2
    }
    
    # Calculate total cycles
    total_cycles = 0
    for instr, count in instruction_counts.items():
        total_cycles += count * cpi_model.get(instr, 1)
    
    # Estimate power consumption
    power_stats = power_model.estimate_power(instruction_counts, total_cycles)
    
    # Combine stats
    analysis = {
        'workload': workload_name,
        'instruction_counts': instruction_counts,
        'total_instructions': total_instructions,
        'total_cycles': total_cycles,
        'execution_time_s': stats['execution_time_s'],
        'is_correct': stats['is_correct'],
        'power_stats': power_stats
    }
    
    print(f"Analysis complete for {workload_name}")
    print(f"Total instructions: {total_instructions}")
    print(f"Total cycles: {total_cycles}")
    print(f"Execution time: {stats['execution_time_s']:.6f} seconds")
    print(f"Power consumption: {power_stats['total_power_mW']:.2f} mW")
    print(f"Energy usage: {power_stats['energy_joules']*1000:.6f} mJ")
    print(f"Energy per instruction: {power_stats['energy_per_instruction_nJ']:.2f} nJ")
    print(f"Correct result: {stats['is_correct']}")
    print()
    
    return analysis

def plot_instruction_mix(analyses):
    """Plot instruction mix for different workloads"""
    workloads = [a['workload'] for a in analyses]
    
    # Get all unique instruction types
    instr_types = set()
    for a in analyses:
        instr_types.update(a['instruction_counts'].keys())
    instr_types = sorted(list(instr_types))
    
    # Create data for plotting
    data = []
    for a in analyses:
        counts = []
        for instr in instr_types:
            counts.append(a['instruction_counts'].get(instr, 0))
        data.append(counts)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(workloads))
    width = 0.8 / len(instr_types)
    
    for i, instr in enumerate(instr_types):
        counts = [a['instruction_counts'].get(instr, 0) for a in analyses]
        ax.bar(x + i * width - 0.4, counts, width, label=instr)
    
    ax.set_xlabel('Workload')
    ax.set_ylabel('Instruction Count')
    ax.set_title('Instruction Mix by Workload')
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('instruction_mix.png')
    plt.close()

def plot_power_comparison(analyses, tech_nodes):
    """Plot power comparison across technology nodes"""
    workloads = [a['workload'] for a in analyses]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(workloads))
    width = 0.8 / len(tech_nodes)
    
    for i, node in enumerate(tech_nodes):
        powers = [a['power_stats']['total_power_mW'] for a in analyses]
        ax.bar(x + i * width - 0.4, powers, width, label=f"{node}nm")
    
    ax.set_xlabel('Workload')
    ax.set_ylabel('Power (mW)')
    ax.set_title('Power Consumption by Technology Node')
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'power_comparison_{workloads[0].replace(" ", "_")}.png')
    plt.close()

def main():
    # Memory size for TinyFive machine
    mem_size = 10000  # 10KB of memory
    
    # Technology nodes to analyze
    tech_nodes = [7, 14, 22, 45]
    
    # Workloads to run
    workloads = [
        ('Matrix Multiplication 4x4', run_matrix_multiply, {'mem_size': mem_size, 'size': 4}),
        ('Matrix Multiplication 8x8', run_matrix_multiply, {'mem_size': mem_size, 'size': 8}),
        ('Vector Addition (100)', run_vector_add, {'mem_size': mem_size, 'size': 100}),
        ('Vector Addition (1000)', run_vector_add, {'mem_size': mem_size, 'size': 1000})
    ]
    
    # Run analyses for each technology node
    all_analyses = []
    
    for node in tech_nodes:
        print(f"\n=== Technology Node: {node}nm ===\n")
        
        # Initialize power model for this technology node
        power_model = PowerModel(technology_node=node, base_frequency=1000)
        
        # Run all workloads
        analyses = []
        for name, func, kwargs in workloads:
            analysis = analyze_workload(name, func, power_model, **kwargs)
            analysis['tech_node'] = node
            analyses.append(analysis)
        
        all_analyses.extend(analyses)
    
    # Plot results
    plot_instruction_mix([a for a in all_analyses if a['tech_node'] == 7])
    
    # Group analyses by workload for power comparison
    workload_names = set(a['workload'] for a in all_analyses)
    for workload in workload_names:
        workload_analyses = [a for a in all_analyses if a['workload'] == workload]
        plot_power_comparison(workload_analyses, tech_nodes)
    
    print("Analysis complete. Results saved to instruction_mix.png and power_comparison_*.png")

if __name__ == "__main__":
    main() 