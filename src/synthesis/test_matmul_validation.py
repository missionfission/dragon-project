import numpy as np
import sys
import os
import ast
from typing import Dict, Tuple

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir.cfg.staticfg.builder import CFGBuilder
from synthesis.hls import parse_graph

def print_ast_graph(node, level=0, prefix=""):
    """Print AST graph in a tree format"""
    indent = "  " * level
    node_type = type(node).__name__
    
    if isinstance(node, ast.Name):
        print(f"{indent}{prefix}{node_type}: {node.id}")
    elif isinstance(node, ast.Constant):
        print(f"{indent}{prefix}{node_type}: {node.value}")
    elif isinstance(node, ast.BinOp):
        print(f"{indent}{prefix}{node_type} ({type(node.op).__name__})")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            print(f"{indent}{prefix}{node_type}: {node.func.id}()")
        else:
            print(f"{indent}{prefix}{node_type}")
    else:
        print(f"{indent}{prefix}{node_type}")
    
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    print_ast_graph(item, level + 1, f"{field}: ")
        elif isinstance(value, ast.AST):
            print_ast_graph(value, level + 1, f"{field}: ")

def analyze_ast(src_code: str, name: str):
    """Analyze AST for the given source code"""
    print(f"\nAST Analysis for {name}:")
    print("=" * 50)
    tree = ast.parse(src_code)
    print_ast_graph(tree)
    
    node_counts = {}
    for node in ast.walk(tree):
        node_type = type(node).__name__
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    
    print("\nNode Type Statistics:")
    print("-" * 30)
    for node_type, count in sorted(node_counts.items()):
        print(f"{node_type:15} : {count}")
    print("-" * 30)

def basic_matmul_code() -> str:
    return """
def basic_matmul(A, B, C, N):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    """

def systolic_matmul_code() -> str:
    return """
def systolic_matmul(A, B, C, N, P):
    # P x P systolic array
    for i in range(0, N, P):
        for j in range(0, N, P):
            for k in range(0, N, P):
                # Process P x P tile
                for ii in range(P):
                    for jj in range(P):
                        for kk in range(P):
                            C[i+ii][j+jj] += A[i+ii][k+kk] * B[k+kk][j+jj]
    """

def calculate_theoretical_metrics(matrix_size: int, tech_node: str = '45nm') -> Dict:
    """Calculate theoretical hardware metrics for matrix multiplication
    
    Args:
        matrix_size: Size of square matrix (N x N)
        tech_node: Technology node for scaling
        
    Returns:
        Dictionary containing theoretical metrics
    """
    # Basic operation counts for N x N matrix multiplication
    num_multiplications = matrix_size ** 3  # N^3 multiplications
    num_additions = matrix_size ** 2 * (matrix_size - 1)  # N^2 * (N-1) additions
    
    # Theoretical cycles (based on latency values from hls.py)
    mult_latency = 5  # From latency dict
    add_latency = 4   # From latency dict
    theoretical_cycles = (num_multiplications * mult_latency + 
                        num_additions * add_latency)
    
    # Minimum required registers
    min_registers = 3 * matrix_size  # Input matrices + output matrix elements
    
    # Theoretical memory bandwidth required (bytes/sec)
    bytes_per_element = 4  # Assuming 32-bit floats
    total_memory = 2 * matrix_size * matrix_size * bytes_per_element  # Input matrices
    
    return {
        "theoretical_cycles": theoretical_cycles,
        "min_registers": min_registers,
        "num_multiplications": num_multiplications,
        "num_additions": num_additions,
        "memory_requirements": total_memory
    }

def calculate_systolic_metrics(matrix_size: int, pe_array_size: int) -> Dict:
    """Calculate theoretical metrics for systolic array implementation
    
    Args:
        matrix_size: Size of square matrix (N x N)
        pe_array_size: Size of processing element array (P x P)
        
    Returns:
        Dictionary containing theoretical metrics
    """
    # Number of processing elements
    num_pes = pe_array_size * pe_array_size
    
    # Cycles for systolic array
    pipeline_depth = 2 * pe_array_size - 1  # Initial fill + drain
    compute_cycles = matrix_size * matrix_size / num_pes  # Parallel computation
    
    # Memory bandwidth and data movement
    bytes_per_element = 4  # Assuming 32-bit floats
    total_data = 2 * matrix_size * matrix_size * bytes_per_element
    reuse_factor = pe_array_size  # Data reuse in systolic array
    memory_bandwidth = total_data / reuse_factor
    
    # Memory cycles considering bandwidth
    memory_cycles = total_data / (pe_array_size * 2)  # Input + output bandwidth
    
    # Total cycles including overhead
    theoretical_cycles = pipeline_depth + max(compute_cycles, memory_cycles)
    sync_overhead = pe_array_size * 0.1  # 10% overhead per PE row
    theoretical_cycles *= (1 + sync_overhead)
    
    # Required registers per PE
    regs_per_pe = 3  # 2 inputs + 1 accumulator
    total_registers = num_pes * regs_per_pe
    
    # Operation counts (same as basic matrix multiplication)
    num_multiplications = matrix_size ** 3  # N^3 multiplications
    num_additions = matrix_size ** 2 * (matrix_size - 1)  # N^2 * (N-1) additions
    
    return {
        "theoretical_cycles": theoretical_cycles,
        "min_registers": total_registers,
        "num_pes": num_pes,
        "pipeline_depth": pipeline_depth,
        "memory_bandwidth": memory_bandwidth,
        "num_multiplications": num_multiplications,
        "num_additions": num_additions
    }

def validate_synthesis_results(hw_allocated: Dict, cycles: float, 
                            theoretical: Dict, margin: float = 0.2) -> Tuple[bool, str]:
    """Validate synthesis results against theoretical calculations
    
    Args:
        hw_allocated: Dictionary of allocated hardware resources
        cycles: Number of cycles from synthesis
        theoretical: Dictionary of theoretical metrics
        margin: Acceptable margin of error (default 20%)
        
    Returns:
        Tuple of (is_valid, explanation)
    """
    validation_messages = []
    is_valid = True
    
    # Check cycles
    cycle_diff = abs(cycles - theoretical['theoretical_cycles']) / theoretical['theoretical_cycles']
    if cycle_diff > margin:
        is_valid = False
        validation_messages.append(
            f"Cycle count differs by {cycle_diff*100:.1f}% from theoretical"
        )
    
    # Check register count
    if hw_allocated['Regs'] < theoretical['min_registers']:
        is_valid = False
        validation_messages.append(
            f"Insufficient registers: {hw_allocated['Regs']} < {theoretical['min_registers']}"
        )
    
    # Check multiplication units
    required_mults = max(1, theoretical['num_multiplications'] // theoretical['theoretical_cycles'])
    if hw_allocated['Mult'] < required_mults:
        is_valid = False
        validation_messages.append(
            f"Insufficient multipliers: {hw_allocated['Mult']} < {required_mults}"
        )
    
    # Check addition units
    required_adds = max(1, theoretical['num_additions'] // theoretical['theoretical_cycles'])
    if hw_allocated['Add'] < required_adds:
        is_valid = False
        validation_messages.append(
            f"Insufficient adders: {hw_allocated['Add']} < {required_adds}"
        )
    
    return is_valid, "\n".join(validation_messages) if validation_messages else "All checks passed"

def test_matmul_synthesis_with_validation():
    """Test matrix multiplication synthesis with validation"""
    # Matrix size parameters
    MATRIX_SIZE = 16
    PE_ARRAY_SIZE = 8
    
    # Calculate theoretical metrics
    theoretical_basic = calculate_theoretical_metrics(MATRIX_SIZE)
    theoretical_systolic = calculate_systolic_metrics(MATRIX_SIZE, PE_ARRAY_SIZE)
    
    # Create CFG builder
    cfg_builder = CFGBuilder(tech_node='45nm')
    
    # Test basic implementation
    src_code_basic = """
def matrix_multiply(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0 for _ in range(p)] for _ in range(n)]
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C
"""
    
    cfg_basic = cfg_builder.build_from_src("matmul", src_code_basic)
    cycles_basic, hw_basic, mem_basic = parse_graph(
        cfg_basic,
        dse_given=True,
        dse_input={"loop1": [16, 4]},
        given_bandwidth=1000000,
        tech_node='45nm'
    )
    
    # Test systolic implementation
    src_code_systolic = """
def matrix_multiply_systolic(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0 for _ in range(p)] for _ in range(n)]
    
    for wave in range(n + m - 1):
        for i in range(max(0, wave - m + 1), min(n, wave + 1)):
            j = wave - i
            if j < p:
                for k in range(m):
                    if k <= wave:
                        C[i][j] += A[i][k] * B[k][j]
    return C
"""
    
    cfg_systolic = cfg_builder.build_from_src("matmul_systolic", src_code_systolic)
    cycles_systolic, hw_systolic, mem_systolic = parse_graph(
        cfg_systolic,
        dse_given=True,
        dse_input={
            "loop1": [16, 8],
            "systolic": True,
        },
        given_bandwidth=2000000,
        tech_node='45nm'
    )
    
    # Print and validate results
    print("\nBasic Matrix Multiplication Validation:")
    print("=" * 50)
    print("Theoretical Metrics:")
    for key, value in theoretical_basic.items():
        print(f"  {key}: {value}")
    
    print("\nSynthesis Results:")
    print(f"  Cycles: {cycles_basic}")
    print("\nHardware Resources:")
    for op, count in hw_basic.items():
        print(f"  {op}: {count}")
    
    is_valid, message = validate_synthesis_results(hw_basic, cycles_basic, theoretical_basic)
    print("\nValidation Result:", "PASS" if is_valid else "FAIL")
    print(message)
    
    print("\nSystolic Array Implementation Validation:")
    print("=" * 50)
    print("Theoretical Metrics:")
    for key, value in theoretical_systolic.items():
        print(f"  {key}: {value}")
    
    print("\nSynthesis Results:")
    print(f"  Cycles: {cycles_systolic}")
    print("\nHardware Resources:")
    for op, count in hw_systolic.items():
        print(f"  {op}: {count}")
    
    is_valid, message = validate_synthesis_results(hw_systolic, cycles_systolic, theoretical_systolic)
    print("\nValidation Result:", "PASS" if is_valid else "FAIL")
    print(message)
    
    # Print performance comparison
    print("\nPerformance Comparison:")
    print("=" * 50)
    speedup = cycles_basic / cycles_systolic
    print(f"Speedup with systolic array: {speedup:.2f}x")
    
    efficiency = (theoretical_systolic['num_pes'] * cycles_systolic) / (MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE)
    print(f"PE Array Efficiency: {efficiency:.2f}")
    
    bandwidth_reduction = theoretical_basic['memory_requirements'] / theoretical_systolic['memory_bandwidth']
    print(f"Memory Bandwidth Reduction: {bandwidth_reduction:.2f}x")

def main():
    # Matrix size and PE array parameters
    N = 32  # Matrix size (N x N)
    P = 8   # PE array size (P x P)
    
    print("\nAnalyzing Matrix Multiplication Implementations:")
    print("=" * 50)
    
    # Analyze basic matrix multiplication
    analyze_ast(basic_matmul_code(), "Basic Matrix Multiplication")
    
    # Analyze systolic array implementation
    analyze_ast(systolic_matmul_code(), "Systolic Array Matrix Multiplication")
    
    # ... rest of the validation code ...
    
    # Matrix size parameters
    MATRIX_SIZE = 16
    PE_ARRAY_SIZE = 8
    
    # Calculate theoretical metrics
    theoretical_basic = calculate_theoretical_metrics(MATRIX_SIZE)
    theoretical_systolic = calculate_systolic_metrics(MATRIX_SIZE, PE_ARRAY_SIZE)
    
    # Create CFG builder
    cfg_builder = CFGBuilder(tech_node='45nm')
    
    # Test basic implementation
    src_code_basic = """
def matrix_multiply(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0 for _ in range(p)] for _ in range(n)]
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C
"""
    
    cfg_basic = cfg_builder.build_from_src("matmul", src_code_basic)
    cycles_basic, hw_basic, mem_basic = parse_graph(
        cfg_basic,
        dse_given=True,
        dse_input={"loop1": [16, 4]},
        given_bandwidth=1000000,
        tech_node='45nm'
    )
    
    # Test systolic implementation
    src_code_systolic = """
def matrix_multiply_systolic(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0 for _ in range(p)] for _ in range(n)]
    
    for wave in range(n + m - 1):
        for i in range(max(0, wave - m + 1), min(n, wave + 1)):
            j = wave - i
            if j < p:
                for k in range(m):
                    if k <= wave:
                        C[i][j] += A[i][k] * B[k][j]
    return C
"""
    
    cfg_systolic = cfg_builder.build_from_src("matmul_systolic", src_code_systolic)
    cycles_systolic, hw_systolic, mem_systolic = parse_graph(
        cfg_systolic,
        dse_given=True,
        dse_input={
            "loop1": [16, 8],
            "systolic": True,
        },
        given_bandwidth=2000000,
        tech_node='45nm'
    )
    
    # Print and validate results
    print("\nBasic Matrix Multiplication Validation:")
    print("=" * 50)
    print("Theoretical Metrics:")
    for key, value in theoretical_basic.items():
        print(f"  {key}: {value}")
    
    print("\nSynthesis Results:")
    print(f"  Cycles: {cycles_basic}")
    print("\nHardware Resources:")
    for op, count in hw_basic.items():
        print(f"  {op}: {count}")
    
    is_valid, message = validate_synthesis_results(hw_basic, cycles_basic, theoretical_basic)
    print("\nValidation Result:", "PASS" if is_valid else "FAIL")
    print(message)
    
    print("\nSystolic Array Implementation Validation:")
    print("=" * 50)
    print("Theoretical Metrics:")
    for key, value in theoretical_systolic.items():
        print(f"  {key}: {value}")
    
    print("\nSynthesis Results:")
    print(f"  Cycles: {cycles_systolic}")
    print("\nHardware Resources:")
    for op, count in hw_systolic.items():
        print(f"  {op}: {count}")
    
    is_valid, message = validate_synthesis_results(hw_systolic, cycles_systolic, theoretical_systolic)
    print("\nValidation Result:", "PASS" if is_valid else "FAIL")
    print(message)
    
    # Print performance comparison
    print("\nPerformance Comparison:")
    print("=" * 50)
    speedup = cycles_basic / cycles_systolic
    print(f"Speedup with systolic array: {speedup:.2f}x")
    
    efficiency = (theoretical_systolic['num_pes'] * cycles_systolic) / (MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE)
    print(f"PE Array Efficiency: {efficiency:.2f}")
    
    bandwidth_reduction = theoretical_basic['memory_requirements'] / theoretical_systolic['memory_bandwidth']
    print(f"Memory Bandwidth Reduction: {bandwidth_reduction:.2f}x")

if __name__ == "__main__":
    main() 