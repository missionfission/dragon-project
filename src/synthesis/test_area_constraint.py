import sys
import os
import ast
import astor

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthesis.hls import allocate_resources_with_area_constraint, build_dfg_from_ast
from ir.cfg.staticfg import CFGBuilder

def load_matmul_code():
    """Load the matrix multiplication code"""
    matmul_code = """
def matmul(A, B, C, N):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    """
    return matmul_code

def load_systolic_code():
    """Load the systolic array matrix multiplication code"""
    systolic_code = """
def matmul_systolic(A, B, C, N, P):
    for i in range(0, N, P):
        for j in range(0, N, P):
            for k in range(0, N, P):
                for ii in range(P):
                    for jj in range(P):
                        for kk in range(P):
                            C[i+ii][j+jj] += A[i+ii][k+kk] * B[k+kk][j+jj]
    """
    return systolic_code

def test_with_area_constraint(area_constraint):
    """Test matrix multiplication with a specific area constraint"""
    print(f"\nTesting with area constraint: {area_constraint} mm^2")
    print("=" * 50)
    
    # Parse basic matrix multiplication
    matmul_code = load_matmul_code()
    matmul_ast = ast.parse(matmul_code)
    matmul_cfg = CFGBuilder().build_from_src('matmul', matmul_code)
    
    print("\nBasic Matrix Multiplication:")
    print("-" * 40)
    matmul_dfg = build_dfg_from_ast(matmul_cfg)
    hw_allocated = allocate_resources_with_area_constraint(
        matmul_dfg, 
        area_constraint, 
        algorithm_type='matmul', 
        algorithm_params={'matrix_size': 16}
    )
    
    print(f"Hardware allocation: {hw_allocated}")
    
    # Parse systolic array matrix multiplication
    systolic_code = load_systolic_code()
    systolic_ast = ast.parse(systolic_code)
    systolic_cfg = CFGBuilder().build_from_src('matmul_systolic', systolic_code)
    
    print("\nSystolic Array Matrix Multiplication:")
    print("-" * 40)
    systolic_dfg = build_dfg_from_ast(systolic_cfg)
    hw_allocated = allocate_resources_with_area_constraint(
        systolic_dfg, 
        area_constraint, 
        algorithm_type='systolic', 
        algorithm_params={'matrix_size': 16, 'tile_size': 8}
    )
    
    print(f"Hardware allocation: {hw_allocated}")

def main():
    """Main function to test area constraints"""
    print("Testing Matrix Multiplication with Different Area Constraints")
    print("=" * 70)
    
    # Test with different area constraints
    test_with_area_constraint(1.0)  # Small area constraint
    test_with_area_constraint(5.0)  # Medium area constraint
    test_with_area_constraint(10.0)  # Large area constraint

if __name__ == "__main__":
    main() 