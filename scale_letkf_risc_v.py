#!/usr/bin/env python3

from tinyfive.machine import machine
import numpy as np
import time

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

def matrix_multiply_risc_v(m, A, B, A_addr, B_addr, C_addr, M, K, N):
    """
    Perform matrix multiplication C = A * B using RISC-V
    
    Args:
        m: TinyFive machine instance
        A: First matrix (M x K)
        B: Second matrix (K x N)
        A_addr: Memory address for matrix A
        B_addr: Memory address for matrix B
        C_addr: Memory address for result matrix C
        M, K, N: Matrix dimensions
    
    Returns:
        Instruction counts dictionary
    """
    # Store matrices in memory
    m.write_i32_vec(A.flatten(), A_addr)
    m.write_i32_vec(B.flatten(), B_addr)
    
    # Reset instruction counter
    instr_counter = {}
    
    # Initialize registers
    m.x[5] = M  # M dimension
    m.x[6] = K  # K dimension
    m.x[7] = N  # N dimension
    m.x[8] = 0  # i = 0
    
    # Outer loop (i)
    while m.x[8] < m.x[5]:  # i < M
        m.x[9] = 0  # j = 0
        
        # Middle loop (j)
        while m.x[9] < m.x[7]:  # j < N
            m.x[10] = 0  # k = 0
            m.x[11] = 0  # sum = 0
            
            # Inner loop (k)
            while m.x[10] < m.x[6]:  # k < K
                # Calculate address of A[i][k]
                m.MUL(12, 8, 6)     # t = i * K
                m.ADD(12, 12, 10)   # t = i * K + k
                m.MUL(12, 12, 4)    # t = (i * K + k) * 4
                m.ADDI(12, 12, A_addr)  # t = A_addr + (i * K + k) * 4
                m.LW(13, 12, 0)     # s0 = A[i][k]
                
                # Calculate address of B[k][j]
                m.MUL(14, 10, 7)    # t = k * N
                m.ADD(14, 14, 9)    # t = k * N + j
                m.MUL(14, 14, 4)    # t = (k * N + j) * 4
                m.ADDI(14, 14, B_addr)  # t = B_addr + (k * N + j) * 4
                m.LW(15, 14, 0)     # s1 = B[k][j]
                
                # Multiply and accumulate
                m.MUL(16, 13, 15)   # s2 = A[i][k] * B[k][j]
                m.ADD(11, 11, 16)   # sum += A[i][k] * B[k][j]
                
                # Increment k
                m.ADDI(10, 10, 1)   # k++
                
                # Update instruction counts
                instr_counter['MUL'] = instr_counter.get('MUL', 0) + 3
                instr_counter['ADD'] = instr_counter.get('ADD', 0) + 3
                instr_counter['ADDI'] = instr_counter.get('ADDI', 0) + 2
                instr_counter['LW'] = instr_counter.get('LW', 0) + 2
            
            # Store result in C[i][j]
            m.MUL(12, 8, 7)     # t = i * N
            m.ADD(12, 12, 9)    # t = i * N + j
            m.MUL(12, 12, 4)    # t = (i * N + j) * 4
            m.ADDI(12, 12, C_addr)  # t = C_addr + (i * N + j) * 4
            m.SW(12, 11, 0)     # C[i][j] = sum
            
            # Increment j
            m.ADDI(9, 9, 1)     # j++
            
            # Update instruction counts
            instr_counter['MUL'] = instr_counter.get('MUL', 0) + 2
            instr_counter['ADD'] = instr_counter.get('ADD', 0) + 1
            instr_counter['ADDI'] = instr_counter.get('ADDI', 0) + 2
            instr_counter['SW'] = instr_counter.get('SW', 0) + 1
        
        # Increment i
        m.ADDI(8, 8, 1)     # i++
        
        # Update instruction counts
        instr_counter['ADDI'] = instr_counter.get('ADDI', 0) + 1
    
    # Read result matrix
    C = m.read_i32_vec(C_addr, M * N).reshape(M, N)
    
    return C, instr_counter

def matrix_transpose_risc_v(m, A, A_addr, AT_addr, rows, cols):
    """
    Perform matrix transpose AT = A^T using RISC-V
    
    Args:
        m: TinyFive machine instance
        A: Input matrix (rows x cols)
        A_addr: Memory address for matrix A
        AT_addr: Memory address for result matrix AT
        rows, cols: Matrix dimensions
    
    Returns:
        Instruction counts dictionary
    """
    # Store matrix in memory
    m.write_i32_vec(A.flatten(), A_addr)
    
    # Reset instruction counter
    instr_counter = {}
    
    # Initialize registers
    m.x[5] = rows  # rows
    m.x[6] = cols  # cols
    m.x[7] = 0     # i = 0
    
    # Outer loop (i)
    while m.x[7] < m.x[5]:  # i < rows
        m.x[8] = 0  # j = 0
        
        # Inner loop (j)
        while m.x[8] < m.x[6]:  # j < cols
            # Calculate address of A[i][j]
            m.MUL(9, 7, 6)      # t = i * cols
            m.ADD(9, 9, 8)      # t = i * cols + j
            m.MUL(9, 9, 4)      # t = (i * cols + j) * 4
            m.ADDI(9, 9, A_addr)  # t = A_addr + (i * cols + j) * 4
            m.LW(10, 9, 0)      # s0 = A[i][j]
            
            # Calculate address of AT[j][i]
            m.MUL(11, 8, 5)     # t = j * rows
            m.ADD(11, 11, 7)    # t = j * rows + i
            m.MUL(11, 11, 4)    # t = (j * rows + i) * 4
            m.ADDI(11, 11, AT_addr)  # t = AT_addr + (j * rows + i) * 4
            m.SW(11, 10, 0)     # AT[j][i] = A[i][j]
            
            # Increment j
            m.ADDI(8, 8, 1)     # j++
            
            # Update instruction counts
            instr_counter['MUL'] = instr_counter.get('MUL', 0) + 4
            instr_counter['ADD'] = instr_counter.get('ADD', 0) + 2
            instr_counter['ADDI'] = instr_counter.get('ADDI', 0) + 3
            instr_counter['LW'] = instr_counter.get('LW', 0) + 1
            instr_counter['SW'] = instr_counter.get('SW', 0) + 1
        
        # Increment i
        m.ADDI(7, 7, 1)     # i++
        
        # Update instruction counts
        instr_counter['ADDI'] = instr_counter.get('ADDI', 0) + 1
    
    # Read result matrix
    AT = m.read_i32_vec(AT_addr, rows * cols).reshape(cols, rows)
    
    return AT, instr_counter

def run_scale_letkf(ensemble_size=4, state_dim=6, obs_dim=3):
    """
    Run SCALE-LETKF algorithm on RISC-V and analyze performance
    
    Args:
        ensemble_size: Number of ensemble members
        state_dim: State dimension
        obs_dim: Observation dimension
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"Running SCALE-LETKF with ensemble_size={ensemble_size}, state_dim={state_dim}, obs_dim={obs_dim}")
    
    # Create a machine instance
    mem_size = 1000000  # 1MB of memory
    m = machine(mem_size=mem_size)
    
    # Create test data
    ensemble_states = np.random.randint(1, 10, (ensemble_size, state_dim)).astype(np.int32)
    H = np.eye(obs_dim, state_dim, dtype=np.int32)  # Simple observation operator
    
    # Memory addresses
    ensemble_addr = 1000
    H_addr = ensemble_addr + ensemble_size * state_dim * 4
    ensemble_T_addr = H_addr + obs_dim * state_dim * 4
    HX_addr = ensemble_T_addr + state_dim * ensemble_size * 4
    result_addr = HX_addr + obs_dim * ensemble_size * 4
    
    # Track total instruction counts
    total_instr_counter = {}
    
    # Step 1: Transpose ensemble states
    print("Step 1: Transpose ensemble states")
    ensemble_T, instr_counter1 = matrix_transpose_risc_v(
        m, ensemble_states, ensemble_addr, ensemble_T_addr, ensemble_size, state_dim)
    
    # Update instruction counts
    for instr, count in instr_counter1.items():
        total_instr_counter[instr] = total_instr_counter.get(instr, 0) + count
    
    # Step 2: Transform to observation space (HX = H * ensemble_T)
    print("Step 2: Transform to observation space")
    HX, instr_counter2 = matrix_multiply_risc_v(
        m, H, ensemble_T, H_addr, ensemble_T_addr, HX_addr, obs_dim, state_dim, ensemble_size)
    
    # Update instruction counts
    for instr, count in instr_counter2.items():
        total_instr_counter[instr] = total_instr_counter.get(instr, 0) + count
    
    # Print results
    print("\nEnsemble states:")
    print(ensemble_states)
    print("\nTransposed ensemble:")
    print(ensemble_T)
    print("\nObservation operator (H):")
    print(H)
    print("\nTransformed ensemble (HX):")
    print(HX)
    
    # Verify results
    expected_HX = np.matmul(H, ensemble_T)
    is_correct = np.array_equal(HX, expected_HX)
    print(f"\nCorrect result: {is_correct}")
    
    # Print instruction counts
    print("\nInstruction counts:")
    for instr, count in sorted(total_instr_counter.items()):
        print(f"{instr}: {count}")
    
    # Analyze power consumption for different technology nodes
    tech_nodes = [7, 14, 22, 45]
    power_results = {}
    
    for node in tech_nodes:
        analyzer = PowerAnalyzer(technology_node=node)
        metrics = analyzer.analyze(total_instr_counter)
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
    
    return {
        'ensemble_size': ensemble_size,
        'state_dim': state_dim,
        'obs_dim': obs_dim,
        'instruction_counts': total_instr_counter,
        'power_results': power_results,
        'is_correct': is_correct
    }

if __name__ == "__main__":
    # Run with different problem sizes
    results_small = run_scale_letkf(ensemble_size=4, state_dim=6, obs_dim=3)
    print("\n" + "="*80 + "\n")
    results_medium = run_scale_letkf(ensemble_size=10, state_dim=20, obs_dim=10)
    print("\n" + "="*80 + "\n")
    results_large = run_scale_letkf(ensemble_size=20, state_dim=50, obs_dim=25) 