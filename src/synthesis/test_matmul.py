import numpy as np
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir.cfg.staticfg.builder import CFGBuilder
from synthesis.hls import parse_graph

def matrix_multiply(A, B):
    """Basic matrix multiplication implementation"""
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0 for _ in range(p)] for _ in range(n)]
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_multiply_systolic(A, B):
    """Matrix multiplication optimized for systolic array hardware"""
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0 for _ in range(p)] for _ in range(n)]
    
    # Systolic array implementation
    # Each PE computes one element of the output matrix
    for wave in range(n + m - 1):  # Wavefront parallelism
        for i in range(max(0, wave - m + 1), min(n, wave + 1)):
            j = wave - i
            if j < p:
                for k in range(m):
                    if k <= wave:
                        C[i][j] += A[i][k] * B[k][j]
    return C

def test_matmul_synthesis():
    # Create CFG builder
    cfg_builder = CFGBuilder(tech_node='45nm')
    
    # Get the source code of the matrix_multiply function
    src_code = """
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
    
    # Build CFG
    cfg = cfg_builder.build_from_src("matmul", src_code)
    
    # Parse graph and get hardware synthesis results
    cycles, hw_allocated, memory_cfgs = parse_graph(
        cfg,
        dse_given=True,
        dse_input={"loop1": [16, 4]},  # Assuming 16x16 matrices with unroll factor 4
        given_bandwidth=1000000,  # 1 MB/s bandwidth
        tech_node='45nm'
    )
    
    # Print synthesis results
    print("\nMatrix Multiplication Hardware Synthesis Results:")
    print("=" * 50)
    print(f"Total Cycles: {cycles}")
    print("\nHardware Resources Allocated:")
    for op, count in hw_allocated.items():
        print(f"  {op}: {count}")
    print("\nMemory Configurations:")
    for var, size in memory_cfgs.items():
        print(f"  {var}: {size} bytes")

    # Get the source code of the systolic array implementation
    src_code = """
def matrix_multiply_systolic(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0 for _ in range(p)] for _ in range(n)]
    
    # Systolic array implementation
    for wave in range(n + m - 1):  # Wavefront parallelism
        for i in range(max(0, wave - m + 1), min(n, wave + 1)):
            j = wave - i
            if j < p:
                for k in range(m):
                    if k <= wave:
                        C[i][j] += A[i][k] * B[k][j]
    return C
"""
    
    # Build CFG
    cfg_systolic = cfg_builder.build_from_src("matmul_systolic", src_code)
    
    # Parse graph and get hardware synthesis results
    # Using larger unroll factor for systolic array
    cycles_systolic, hw_allocated_systolic, memory_cfgs_systolic = parse_graph(
        cfg_systolic,
        dse_given=True,
        dse_input={
            "loop1": [16, 8],  # Increased unroll factor for wavefront parallelism
            "systolic": True,  # Enable systolic array optimizations
        },
        given_bandwidth=2000000,  # Increased bandwidth for parallel access
        tech_node='45nm'
    )
    
    # Print synthesis results
    print("\nMatrix Multiplication (Systolic Array) Hardware Synthesis Results:")
    print("=" * 60)
    print(f"Total Cycles: {cycles_systolic}")
    print("\nHardware Resources Allocated:")
    for op, count in hw_allocated_systolic.items():
        print(f"  {op}: {count}")
    print("\nMemory Configurations:")
    for var, size in memory_cfgs_systolic.items():
        print(f"  {var}: {size} bytes")
    
    # Print systolic array specific details
    print("\nSystolic Array Configuration:")
    print(f"  Array Size: 8x8 Processing Elements")
    print(f"  Wavefront Parallelism: Enabled")
    print(f"  Input Bandwidth: 2 MB/s")
    print(f"  Technology Node: 45nm")

if __name__ == "__main__":
    test_matmul_synthesis() 