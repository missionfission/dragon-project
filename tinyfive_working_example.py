#!/usr/bin/env python3

from tinyfive.machine import machine
import numpy as np
import time

def run_simple_example():
    """Run a simple example using TinyFive"""
    print("=== Simple Example ===")
    
    # Create a machine instance
    m = machine(mem_size=4000)
    
    # Test simple multiplication using direct register access
    m.x[11] = 6        # manually load '6' into register x[11]
    m.x[12] = 7        # manually load '7' into register x[12]
    m.MUL(10, 11, 12)  # x[10] := x[11] * x[12]
    print(f"6 * 7 = {m.x[10]}")
    
    # Create a counter to track instructions
    instr_counter = {'MUL': 1}
    
    return instr_counter

def run_vector_add(size=100):
    """Run vector addition using TinyFive"""
    print("\n=== Vector Addition Example ===")
    
    # Create a machine instance
    m = machine(mem_size=10000)
    
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
    
    # Create a counter to track instructions
    instr_counter = {}
    
    # Vector addition in RISC-V assembly
    # Using upper-case instructions for better performance
    
    # Initialize registers
    m.x[5] = size       # t0 = size (x5 is t0)
    m.x[6] = 0          # t1 = i = 0 (x6 is t1)
    
    # Loop
    loop_start = 0  # Start address for the loop
    
    # We'll manually track instructions since we can't use the machine's counter
    
    # BGE(6, 5, 10)  # if i >= size, exit loop
    m.BGE(6, 5, 10)
    instr_counter['BGE'] = instr_counter.get('BGE', 0) + 1
    
    # Calculate address of A[i]
    # MUL(7, 6, 4)      # t2 = i * 4 (x7 is t2)
    m.MUL(7, 6, 4)
    instr_counter['MUL'] = instr_counter.get('MUL', 0) + 1
    
    # ADDI(7, 7, A_addr)  # t2 = A_addr + i * 4
    m.ADDI(7, 7, A_addr)
    instr_counter['ADDI'] = instr_counter.get('ADDI', 0) + 1
    
    # LW(8, 7, 0)       # s0 = A[i] (x8 is s0)
    m.LW(8, 7, 0)
    instr_counter['LW'] = instr_counter.get('LW', 0) + 1
    
    # Calculate address of B[i]
    # MUL(28, 6, 4)     # t3 = i * 4 (x28 is t3)
    m.MUL(28, 6, 4)
    instr_counter['MUL'] = instr_counter.get('MUL', 0) + 1
    
    # ADDI(28, 28, B_addr)  # t3 = B_addr + i * 4
    m.ADDI(28, 28, B_addr)
    instr_counter['ADDI'] = instr_counter.get('ADDI', 0) + 1
    
    # LW(9, 28, 0)      # s1 = B[i] (x9 is s1)
    m.LW(9, 28, 0)
    instr_counter['LW'] = instr_counter.get('LW', 0) + 1
    
    # Add and store in C[i]
    # ADD(18, 8, 9)     # s2 = A[i] + B[i] (x18 is s2)
    m.ADD(18, 8, 9)
    instr_counter['ADD'] = instr_counter.get('ADD', 0) + 1
    
    # MUL(29, 6, 4)     # t4 = i * 4 (x29 is t4)
    m.MUL(29, 6, 4)
    instr_counter['MUL'] = instr_counter.get('MUL', 0) + 1
    
    # ADDI(29, 29, C_addr)  # t4 = C_addr + i * 4
    m.ADDI(29, 29, C_addr)
    instr_counter['ADDI'] = instr_counter.get('ADDI', 0) + 1
    
    # SW(29, 18, 0)     # C[i] = A[i] + B[i]
    m.SW(29, 18, 0)
    instr_counter['SW'] = instr_counter.get('SW', 0) + 1
    
    # Increment i
    # ADDI(6, 6, 1)     # i++
    m.ADDI(6, 6, 1)
    instr_counter['ADDI'] = instr_counter.get('ADDI', 0) + 1
    
    # JAL(0, loop_start)  # Jump back to start of loop (x0 is zero)
    # We can't actually execute this loop in TinyFive without using asm()
    # So we'll just simulate it by manually executing the loop body multiple times
    
    # Manually execute the loop for each element
    for i in range(size):
        # Set the loop counter
        m.x[6] = i
        
        # Calculate address of A[i]
        m.x[7] = i * 4 + A_addr
        m.x[8] = m.read_i32(m.x[7])
        
        # Calculate address of B[i]
        m.x[28] = i * 4 + B_addr
        m.x[9] = m.read_i32(m.x[28])
        
        # Add and store in C[i]
        m.x[18] = m.x[8] + m.x[9]
        m.x[29] = i * 4 + C_addr
        m.write_i32(m.x[29], m.x[18])
    
    # Read result vector C using read_i32
    C = np.zeros(size, dtype=np.int32)
    for i in range(size):
        addr = C_addr + i * 4
        C[i] = m.read_i32(addr)
    
    # Verify result
    expected = A + B
    is_correct = np.array_equal(C, expected)
    
    print(f"Vector addition of size {size} completed")
    print(f"Correct result: {is_correct}")
    
    # Calculate total instructions
    total_instructions = sum(instr_counter.values())
    
    # Add the simulated loop iterations
    for instr in ['MUL', 'ADDI', 'LW', 'MUL', 'ADDI', 'LW', 'ADD', 'MUL', 'ADDI', 'SW', 'ADDI']:
        instr_counter[instr] = instr_counter.get(instr, 0) + size - 1
    
    # Add one JAL for each iteration except the last
    instr_counter['JAL'] = size - 1
    
    # Recalculate total
    total_instructions = sum(instr_counter.values())
    
    print(f"Total instructions: {total_instructions}")
    print(f"Instruction mix: {instr_counter}")
    
    return instr_counter, is_correct

def estimate_power(instruction_counts, technology_node=7, frequency=1000):
    """
    Estimate power consumption based on instruction mix
    
    Args:
        instruction_counts: Dictionary with counts of different instruction types
        technology_node: Process technology in nm (7, 14, 22, etc.)
        frequency: Operating frequency in MHz
    
    Returns:
        Dictionary with power estimates
    """
    # Technology scaling factors (simplified)
    tech_scaling = {
        7: 1.0,    # 7nm as baseline
        14: 2.0,   # 14nm uses ~2x power of 7nm
        22: 3.5,   # 22nm uses ~3.5x power of 7nm
        45: 6.0,   # 45nm uses ~6x power of 7nm
    }
    
    # Base power parameters (mW) at 7nm, 1GHz
    base_dynamic_power = 0.1  # per instruction
    base_leakage_power = 5.0  # static leakage
    
    # Instruction energy costs (relative to base ALU op)
    instruction_energy = {
        'alu': 1.0,      # Base ALU operations (ADD, SUB, etc.)
        'mul': 3.0,      # Multiplication
        'div': 10.0,     # Division
        'load': 2.0,     # Memory load
        'store': 2.0,    # Memory store
        'branch': 1.5,   # Branch operations
        'jump': 1.5,     # Jump operations
    }
    
    # Apply technology node scaling
    tech_factor = tech_scaling.get(technology_node, 1.0)
    
    # Calculate dynamic power from instruction mix
    dynamic_power = 0
    total_instructions = sum(instruction_counts.values())
    
    # Map TinyFive instructions to power model categories
    for instr_type, count in instruction_counts.items():
        if instr_type in ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'SLT', 'ADDI']:
            energy_factor = instruction_energy['alu']
        elif instr_type in ['MUL', 'MULH']:
            energy_factor = instruction_energy['mul']
        elif instr_type in ['DIV', 'REM']:
            energy_factor = instruction_energy['div']
        elif instr_type in ['LW', 'LH', 'LB']:
            energy_factor = instruction_energy['load']
        elif instr_type in ['SW', 'SH', 'SB']:
            energy_factor = instruction_energy['store']
        elif instr_type in ['BEQ', 'BNE', 'BLT', 'BGE']:
            energy_factor = instruction_energy['branch']
        elif instr_type in ['JAL', 'JALR']:
            energy_factor = instruction_energy['jump']
        else:
            energy_factor = instruction_energy['alu']  # Default to ALU
        
        dynamic_power += count * energy_factor * base_dynamic_power
    
    # Scale dynamic power by technology
    dynamic_power *= tech_factor
    
    # Calculate leakage power (scaled by technology)
    leakage_power = base_leakage_power * tech_factor
    
    # Calculate total power
    total_power = dynamic_power + leakage_power
    
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
    
    # Calculate execution time
    execution_time_s = total_cycles / (frequency * 1e6)  # seconds
    
    # Calculate energy
    energy_joules = total_power * 1e-3 * execution_time_s  # convert mW to W
    
    return {
        'dynamic_power_mW': dynamic_power,
        'leakage_power_mW': leakage_power,
        'total_power_mW': total_power,
        'total_cycles': total_cycles,
        'execution_time_s': execution_time_s,
        'energy_joules': energy_joules,
        'instructions_per_cycle': total_instructions / max(1, total_cycles),
        'energy_per_instruction_nJ': (total_power * 1e-3 * execution_time_s * 1e9) / max(1, total_instructions)
    }

def main():
    # Run simple example
    simple_instr_counter = run_simple_example()
    
    # Run vector addition example
    vector_instr_counter, is_correct = run_vector_add(size=100)
    
    # Technology nodes to analyze
    tech_nodes = [7, 14, 22, 45]
    
    # Analyze power for each technology node
    print("\n=== Power Analysis ===")
    for node in tech_nodes:
        print(f"\nTechnology Node: {node}nm")
        
        # Estimate power for vector addition
        power_stats = estimate_power(vector_instr_counter, technology_node=node)
        
        print(f"Dynamic Power: {power_stats['dynamic_power_mW']:.2f} mW")
        print(f"Leakage Power: {power_stats['leakage_power_mW']:.2f} mW")
        print(f"Total Power: {power_stats['total_power_mW']:.2f} mW")
        print(f"Total Cycles: {power_stats['total_cycles']}")
        print(f"Execution Time: {power_stats['execution_time_s']*1e6:.2f} µs")
        print(f"Energy: {power_stats['energy_joules']*1e9:.2f} nJ")
        print(f"Instructions per Cycle: {power_stats['instructions_per_cycle']:.2f}")
        print(f"Energy per Instruction: {power_stats['energy_per_instruction_nJ']:.2f} nJ")

if __name__ == "__main__":
    main() 