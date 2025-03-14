import numpy as np
import sys
import os
import ast
from typing import Dict, Tuple

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir.cfg.staticfg.builder import CFGBuilder
from synthesis.hls import parse_graph

def calculate_theoretical_metrics(ensemble_size=4, state_dim=6, obs_dim=3, block_size=2):
    """Calculate theoretical hardware metrics for LETKF
    
    Args:
        ensemble_size: Number of ensemble members
        state_dim: State dimension
        obs_dim: Observation dimension
        block_size: Block size for matrix operations
        
    Returns:
        Dictionary containing theoretical metrics
    """
    # Operation counts for LETKF
    # 1. Compute ensemble mean: state_dim * ensemble_size additions, state_dim divisions
    mean_ops = {
        'Add': state_dim * (ensemble_size - 1),
        'Div': state_dim
    }
    
    # 2. Compute perturbations: state_dim * ensemble_size subtractions
    pert_ops = {
        'Sub': state_dim * ensemble_size
    }
    
    # 3. Transform to observation space: obs_dim * state_dim * ensemble_size multiplications and additions
    transform_ops = {
        'Mult': obs_dim * state_dim * ensemble_size,
        'Add': obs_dim * state_dim * (ensemble_size - 1)
    }
    
    # 4. SVD computation (power iteration): ~10 iterations * (obs_dim * ensemble_size^2) operations
    svd_ops = {
        'Mult': 10 * obs_dim * ensemble_size**2,
        'Add': 10 * obs_dim * ensemble_size**2,
        'Div': 10 * ensemble_size
    }
    
    # 5. Analysis ensemble computation: obs_dim * ensemble_size * state_dim operations
    analysis_ops = {
        'Mult': obs_dim * ensemble_size * state_dim,
        'Add': obs_dim * (ensemble_size - 1) * state_dim
    }
    
    # Total operations
    total_ops = {}
    for op_type in set(list(mean_ops.keys()) + list(pert_ops.keys()) + list(transform_ops.keys()) + 
                      list(svd_ops.keys()) + list(analysis_ops.keys())):
        total_ops[op_type] = (mean_ops.get(op_type, 0) + pert_ops.get(op_type, 0) + 
                             transform_ops.get(op_type, 0) + svd_ops.get(op_type, 0) + 
                             analysis_ops.get(op_type, 0))
    
    # Minimum registers needed
    # - Ensemble states: ensemble_size * state_dim
    # - Ensemble mean: state_dim
    # - Perturbations: ensemble_size * state_dim
    # - Observation space perturbations: obs_dim * ensemble_size
    # - SVD components: obs_dim + ensemble_size + 1
    # - Analysis weights: obs_dim * ensemble_size
    min_registers = (ensemble_size * state_dim + state_dim + ensemble_size * state_dim + 
                    obs_dim * ensemble_size + obs_dim + ensemble_size + 1 + obs_dim * ensemble_size)
    
    # Calculate theoretical cycles
    # Assuming:
    # - 4 cycles per multiplication
    # - 2 cycles per addition/subtraction
    # - 10 cycles per division
    theoretical_cycles = (total_ops.get('Mult', 0) * 4 + 
                         (total_ops.get('Add', 0) + total_ops.get('Sub', 0)) * 2 + 
                         total_ops.get('Div', 0) * 10)
    
    # Adjust for parallelism with block size
    parallelism_factor = min(block_size**2, min(state_dim, ensemble_size))
    theoretical_cycles = theoretical_cycles / max(1, parallelism_factor)
    
    # Minimum hardware resources
    min_multipliers = max(1, total_ops.get('Mult', 0) // theoretical_cycles)
    min_adders = max(1, (total_ops.get('Add', 0) + total_ops.get('Sub', 0)) // theoretical_cycles)
    min_dividers = max(1, total_ops.get('Div', 0) // theoretical_cycles)
    
    return {
        'theoretical_cycles': theoretical_cycles,
        'min_registers': min_registers,
        'min_multipliers': min_multipliers,
        'min_adders': min_adders,
        'min_dividers': min_dividers,
        'total_operations': sum(total_ops.values()),
        'operation_counts': total_ops,
        'ensemble_size': ensemble_size,
        'state_dim': state_dim,
        'obs_dim': obs_dim,
        'block_size': block_size
    }

def validate_synthesis_results(hw_allocated, cycles, theoretical, margin=0.3):
    """Validate synthesis results against theoretical calculations
    
    Args:
        hw_allocated: Dictionary of allocated hardware resources
        cycles: Number of cycles from synthesis
        theoretical: Dictionary of theoretical metrics
        margin: Acceptable margin of error (default 30%)
        
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
    if hw_allocated.get('Regs', 0) < theoretical['min_registers']:
        is_valid = False
        validation_messages.append(
            f"Insufficient registers: {hw_allocated.get('Regs', 0)} < {theoretical['min_registers']}"
        )
    
    # Check multiplication units
    if hw_allocated.get('Mult', 0) < theoretical['min_multipliers']:
        is_valid = False
        validation_messages.append(
            f"Insufficient multipliers: {hw_allocated.get('Mult', 0)} < {theoretical['min_multipliers']}"
        )
    
    # Check addition units
    if hw_allocated.get('Add', 0) < theoretical['min_adders']:
        is_valid = False
        validation_messages.append(
            f"Insufficient adders: {hw_allocated.get('Add', 0)} < {theoretical['min_adders']}"
        )
    
    # Check division units
    if hw_allocated.get('Div', 0) < theoretical['min_dividers']:
        is_valid = False
        validation_messages.append(
            f"Insufficient dividers: {hw_allocated.get('Div', 0)} < {theoretical['min_dividers']}"
        )
    
    return is_valid, "\n".join(validation_messages) if validation_messages else "All checks passed"

def debug_ast_graph(graph, name="LETKF Graph"):
    """Debug the AST graph structure to understand how parse_graph processes it
    
    Args:
        graph: The control flow graph to debug
        name: Name for the debug output
    """
    print(f"\n{name} Analysis:")
    print("=" * 50)
    
    # Count nodes and statements
    node_count = 0
    stmt_count = 0
    loop_count = 0
    call_count = 0
    assign_count = 0
    binop_count = 0
    
    # Track operation types
    operations = {
        'Add': 0,
        'Sub': 0,
        'Mult': 0,
        'Div': 0,
        'Load': 0,
        'Store': 0,
        'Call': 0
    }
    
    # Track function calls
    function_calls = {}
    
    # Analyze graph structure
    for node in graph:
        node_count += 1
        
        # Print node info
        print(f"\nNode {node_count}: {node}")
        
        for stmt in node.statements:
            stmt_count += 1
            stmt_type = type(stmt).__name__
            
            # Count statement types
            if isinstance(stmt, ast.For):
                loop_count += 1
                print(f"  Loop {loop_count}: {ast.unparse(stmt)[:50]}...")
                
                # Analyze loop body
                loop_body_ops = count_operations(stmt.body)
                for op, count in loop_body_ops.items():
                    operations[op] = operations.get(op, 0) + count
                
                # Get loop iterations if available
                if hasattr(stmt, 'iter') and hasattr(stmt.iter, 'args') and len(stmt.iter.args) > 0:
                    if isinstance(stmt.iter.args[0], ast.Constant):
                        print(f"    Loop iterations: {stmt.iter.args[0].value}")
                    else:
                        print(f"    Loop iterations: variable")
                
            elif isinstance(stmt, ast.Call):
                call_count += 1
                if hasattr(stmt, 'func') and hasattr(stmt.func, 'id'):
                    func_name = stmt.func.id
                    function_calls[func_name] = function_calls.get(func_name, 0) + 1
                    print(f"  Function call: {func_name}")
                
            elif isinstance(stmt, ast.Assign):
                assign_count += 1
                
                # Check for operations in the assignment
                if hasattr(stmt, 'value'):
                    if isinstance(stmt.value, ast.BinOp):
                        binop_count += 1
                        op_type = type(stmt.value.op).__name__
                        operations[op_type] = operations.get(op_type, 0) + 1
                        print(f"  Assignment with {op_type}: {ast.unparse(stmt)[:50]}...")
    
    # Print summary
    print("\nGraph Summary:")
    print(f"Total nodes: {node_count}")
    print(f"Total statements: {stmt_count}")
    print(f"Loops: {loop_count}")
    print(f"Function calls: {call_count}")
    print(f"Assignments: {assign_count}")
    print(f"Binary operations: {binop_count}")
    
    print("\nOperation Counts:")
    for op, count in operations.items():
        if count > 0:
            print(f"  {op}: {count}")
    
    print("\nFunction Calls:")
    for func, count in function_calls.items():
        print(f"  {func}: {count}")
    
    return operations, function_calls

def count_operations(statements):
    """Count operations in a list of statements
    
    Args:
        statements: List of AST statements
        
    Returns:
        Dictionary of operation counts
    """
    operations = {}
    
    for stmt in statements:
        if isinstance(stmt, ast.BinOp):
            op_type = type(stmt.op).__name__
            operations[op_type] = operations.get(op_type, 0) + 1
        
        # Recursively process nested statements
        if hasattr(stmt, 'body'):
            nested_ops = count_operations(stmt.body)
            for op, count in nested_ops.items():
                operations[op] = operations.get(op, 0) + count
    
    return operations

def get_letkf_implementation():
    """Get the LETKF implementation code"""
    src_code = """
def matrix_multiply_hls(A, B, M, K, N):
    '''
    HLS-friendly matrix multiplication
    Uses basic loops instead of NumPy operations
    '''
    C = [[0.0 for _ in range(N)] for _ in range(M)]
    
    for i in range(M):
        for j in range(N):
            sum_val = 0.0
            for k in range(K):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
    
    return C

def matrix_multiply_block_hls(A, B, M, K, N, block_size):
    '''
    HLS-friendly blocked matrix multiplication
    Designed for hardware implementation with configurable block size
    '''
    C = [[0.0 for _ in range(N)] for _ in range(M)]
    
    # Block sizes adjusted to matrix dimensions
    block_size_m = min(block_size, M)
    block_size_n = min(block_size, N)
    block_size_k = min(block_size, K)
    
    # Iterate over blocks
    for i0 in range(0, M, block_size_m):
        i_end = min(i0 + block_size_m, M)
        for j0 in range(0, N, block_size_n):
            j_end = min(j0 + block_size_n, N)
            for k0 in range(0, K, block_size_k):
                k_end = min(k0 + block_size_k, K)
                
                # Compute block multiplication
                for i in range(i0, i_end):
                    for j in range(j0, j_end):
                        sum_val = C[i][j]  # Load existing value
                        for k in range(k0, k_end):
                            sum_val += A[i][k] * B[k][j]
                        C[i][j] = sum_val  # Store result
    
    return C

def matrix_transpose_hls(A, rows, cols):
    '''
    HLS-friendly matrix transpose
    '''
    AT = [[0.0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            AT[j][i] = A[i][j]
    return AT

def power_iteration_hls(A, max_iter, M, N):
    '''
    HLS-friendly power iteration method for dominant eigenpair computation
    '''
    # Initialize random vector
    v = [1.0/N**0.5 for _ in range(N)]
    
    # Power iteration
    for _ in range(max_iter):
        # Matrix-vector multiplication: u = A*v
        u = [0.0] * M
        for i in range(M):
            for j in range(N):
                u[i] += A[i][j] * v[j]
        
        # Normalize u
        norm_u = 0.0
        for i in range(M):
            norm_u += u[i] * u[i]
        norm_u = norm_u ** 0.5
        if norm_u > 0:
            for i in range(M):
                u[i] /= norm_u
        
        # Matrix-vector multiplication: v = A^T*u
        v = [0.0] * N
        for i in range(N):
            for j in range(M):
                v[i] += A[j][i] * u[j]
        
        # Normalize v
        norm_v = 0.0
        for i in range(N):
            norm_v += v[i] * v[i]
        norm_v = norm_v ** 0.5
        if norm_v > 0:
            for i in range(N):
                v[i] /= norm_v
    
    # Compute eigenvalue
    lambda_val = 0.0
    for i in range(M):
        temp = 0.0
        for j in range(N):
            temp += A[i][j] * v[j]
        lambda_val += u[i] * temp
    
    return lambda_val, u, v

def svd_block_hls(X, max_iterations, M, N):
    '''
    HLS-friendly SVD computation using power iteration
    Returns only the dominant singular value and vectors
    '''
    # Compute X^T * X for eigendecomposition
    XtX = matrix_multiply_hls(matrix_transpose_hls(X, M, N), X, N, M, N)
    
    # Get dominant eigenpair using power iteration
    lambda_val, _, v = power_iteration_hls(XtX, max_iterations, N, N)
    
    # Compute corresponding left singular vector
    u = [0.0] * M
    for i in range(M):
        for j in range(N):
            u[i] += X[i][j] * v[j]
    
    # Normalize u
    norm_u = 0.0
    for i in range(M):
        norm_u += u[i] * u[i]
    norm_u = norm_u ** 0.5
    if norm_u > 0:
        for i in range(M):
            u[i] /= norm_u
    
    # Singular value is square root of eigenvalue
    s = lambda_val ** 0.5
    
    return s, u, v

class SCALELETKF_HLS:
    def __init__(self, ensemble_size, state_dim, obs_dim, block_size=32):
        self.k = ensemble_size
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.block_size = block_size
    
    def compute_mean_hls(self, ensemble_states):
        '''
        HLS-friendly ensemble mean computation
        '''
        mean = [0.0] * self.state_dim
        for i in range(self.state_dim):
            for j in range(self.k):
                mean[i] += ensemble_states[j][i]
            mean[i] /= self.k
        return mean
    
    def compute_perturbations_hls(self, ensemble_states, mean):
        '''
        HLS-friendly perturbation computation
        '''
        X = [[0.0 for _ in range(self.state_dim)] for _ in range(self.k)]
        for i in range(self.k):
            for j in range(self.state_dim):
                X[i][j] = ensemble_states[i][j] - mean[j]
        return X
    
    def compute_letkf_step_hls(self, ensemble_states, observations, obs_error_cov, H):
        '''
        HLS-friendly LETKF computation step
        Breaks down the algorithm into basic operations
        '''
        # Compute ensemble mean
        x_mean = self.compute_mean_hls(ensemble_states)
        
        # Compute perturbations
        X = self.compute_perturbations_hls(ensemble_states, x_mean)
        
        # Transform to observation space using blocked operations
        HX = matrix_multiply_block_hls(H, matrix_transpose_hls(X, self.k, self.state_dim),
                                     self.obs_dim, self.state_dim, self.k,
                                     self.block_size)
        
        # Compute analysis weights using power iteration
        s, u, v = svd_block_hls(HX, 30, self.obs_dim, self.k)
        
        # Compute analysis ensemble using blocked operations
        W = matrix_multiply_block_hls(u, v, self.obs_dim, 1, self.k,
                                    self.block_size)
        
        return matrix_multiply_block_hls(W, X, self.obs_dim, self.k, self.state_dim,
                                       self.block_size)
"""
    return src_code

def test_letkf_synthesis_with_validation():
    """Test LETKF synthesis with validation"""
    # Parameters
    ensemble_size = 4
    state_dim = 6
    obs_dim = 3
    block_size = 2
    
    # Create CFG builder
    cfg_builder = CFGBuilder(tech_node='45nm')
    
    # Get LETKF implementation
    src_code = get_letkf_implementation()
    
    # Build CFG for the LETKF compute step
    cfg = cfg_builder.build_from_src("compute_letkf_step_hls", src_code)
    
    # Debug AST graph
    operations, function_calls = debug_ast_graph(cfg, "LETKF AST Graph")
    
    # Calculate theoretical metrics
    theoretical = calculate_theoretical_metrics(
        ensemble_size=ensemble_size,
        state_dim=state_dim,
        obs_dim=obs_dim,
        block_size=block_size
    )
    
    # Print theoretical metrics
    print("\nTheoretical Metrics:")
    print("=" * 50)
    print(f"Ensemble size: {ensemble_size}")
    print(f"State dimension: {state_dim}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Block size: {block_size}")
    print(f"Theoretical cycles: {theoretical['theoretical_cycles']:.1f}")
    print(f"Minimum registers: {theoretical['min_registers']}")
    print(f"Minimum multipliers: {theoretical['min_multipliers']}")
    print(f"Minimum adders: {theoretical['min_adders']}")
    print(f"Minimum dividers: {theoretical['min_dividers']}")
    print(f"Total operations: {theoretical['total_operations']}")
    print("\nOperation counts:")
    for op, count in theoretical['operation_counts'].items():
        print(f"  {op}: {count}")
    
    # Run parse_graph
    cycles, hw_allocated, memory_cfgs = parse_graph(
        cfg,
        dse_given=True,
        dse_input={"cycle_time": 2, "unrolling": 2},
        given_bandwidth=1000000,
        tech_node='45nm'
    )
    
    # Print synthesis results
    print("\nSynthesis Results:")
    print("=" * 50)
    print(f"Cycles: {cycles}")
    print(f"Hardware allocation:")
    for op, count in hw_allocated.items():
        print(f"  {op}: {count}")
    
    # Validate results
    is_valid, message = validate_synthesis_results(
        hw_allocated, 
        cycles, 
        theoretical,
        margin=0.3  # 30% margin
    )
    
    print("\nValidation Result:", "PASS" if is_valid else "FAIL")
    print(message)
    
    return is_valid, cycles, hw_allocated

def main():
    print("\nAnalyzing SCALE LETKF HLS Implementation:")
    print("=" * 50)
    
    is_valid, cycles, hw_allocated = test_letkf_synthesis_with_validation()
    
    if is_valid:
        print("\nSUCCESS: LETKF implementation validated successfully!")
        print(f"Cycles: {cycles}, Hardware: {hw_allocated}")
    else:
        print("\nWARNING: LETKF implementation validation failed.")
        print("Please check the implementation and HLS calculations.")

if __name__ == "__main__":
    main() 