import os
import sys
import time
import ast
import numpy as np

# Add the src directory to the path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from ir.cfg.staticfg import CFGBuilder
from synthesis.hls import parse_graph, improved_parse_graph

def get_matrix_multiply_implementation():
    """Get matrix multiplication implementation"""
    return """
def matrix_multiply(A, B, C, N):
    for i in range(N):
        for j in range(N):
            sum = 0
            for k in range(N):
                sum += A[i][k] * B[k][j]
            C[i][j] = sum
    return C
"""

def get_fir_filter_implementation():
    """Get FIR filter implementation"""
    return """
def fir_filter(input_signal, coeffs, output_signal, signal_len, filter_len):
    for n in range(signal_len):
        output_signal[n] = 0
        for k in range(filter_len):
            if n - k >= 0:
                output_signal[n] += coeffs[k] * input_signal[n - k]
    return output_signal
"""

def get_aes_implementation():
    """Get simplified AES implementation for testing"""
    return """
def aes_encrypt(plaintext, key, ciphertext, num_rounds):
    # Initialize state and round keys
    state = [0] * 16
    round_keys = [0] * (16 * (num_rounds + 1))
    sbox = [0] * 256  # Placeholder for S-box
    
    # Initial round key addition
    for i in range(16):
        state[i] = plaintext[i] ^ key[i]
    
    # Main rounds
    for round_num in range(1, num_rounds):
        # SubBytes
        for i in range(16):
            state[i] = sbox[state[i]]
        
        # ShiftRows
        for i in range(1, 4):
            for j in range(i):
                temp = state[i]
                state[i] = state[i + 4]
                state[i + 4] = state[i + 8]
                state[i + 8] = state[i + 12]
                state[i + 12] = temp
        
        # MixColumns (simplified)
        for col in range(4):
            a = state[col * 4]
            b = state[col * 4 + 1]
            c = state[col * 4 + 2]
            d = state[col * 4 + 3]
            
            state[col * 4] = (2 * a) ^ (3 * b) ^ c ^ d
            state[col * 4 + 1] = a ^ (2 * b) ^ (3 * c) ^ d
            state[col * 4 + 2] = a ^ b ^ (2 * c) ^ (3 * d)
            state[col * 4 + 3] = (3 * a) ^ b ^ c ^ (2 * d)
        
        # AddRoundKey
        for i in range(16):
            state[i] ^= round_keys[round_num * 16 + i]
    
    # Final round (no MixColumns)
    for i in range(16):
        state[i] = sbox[state[i]]
    
    # ShiftRows
    for i in range(1, 4):
        for j in range(i):
            temp = state[i]
            state[i] = state[i + 4]
            state[i + 4] = state[i + 8]
            state[i + 8] = state[i + 12]
            state[i + 12] = temp
    
    # AddRoundKey
    for i in range(16):
        ciphertext[i] = state[i] ^ round_keys[num_rounds * 16 + i]
    
    return ciphertext
"""

def test_benchmark(benchmark_name, src_code, function_name, dse_input=None):
    """Test a benchmark with both original and improved HLS implementations
    
    Args:
        benchmark_name: Name of the benchmark
        src_code: Source code of the benchmark
        function_name: Name of the function to analyze
        dse_input: Design space exploration input parameters
        
    Returns:
        dict: Benchmark results
    """
    print(f"\nTesting {benchmark_name} benchmark:")
    print("=" * 50)
    
    # Create CFG builder
    cfg_builder = CFGBuilder(tech_node='45nm')
    
    # Build CFG
    try:
        cfg = cfg_builder.build_from_src(function_name, src_code)
        print(f"Successfully built CFG for {function_name}")
    except Exception as e:
        print(f"Error building CFG for {function_name}: {e}")
        print(f"Source code:\n{src_code}")
        return {
            "benchmark": benchmark_name,
            "original_success": False,
            "improved_success": False,
            "original_cycles": 0,
            "improved_cycles": 0,
            "original_power": 0,
            "improved_power": 0,
            "original_time": 0,
            "improved_time": 0,
            "cycle_diff": 1.0,
            "power_diff": 1.0,
            "time_diff": 1.0
        }
    
    # Set DSE parameters
    dse_given = dse_input is not None
    if not dse_given:
        dse_input = {"cycle_time": 2}
    
    # Run original parse_graph
    start_time = time.time()
    try:
        original_cycles, original_hw, original_mem = parse_graph(
            cfg,
            dse_given=dse_given,
            dse_input=dse_input,
            given_bandwidth=1000000,
            tech_node='45nm'
        )
        original_time = time.time() - start_time
        original_success = True
    except Exception as e:
        print(f"Error in original parse_graph: {e}")
        import traceback
        traceback.print_exc()
        original_cycles = 0
        original_hw = {"power": 0}
        original_mem = {}
        original_time = 0
        original_success = False
    
    # Run improved parse_graph
    start_time = time.time()
    try:
        improved_cycles, improved_hw, improved_mem = improved_parse_graph(
            cfg,
            dse_given=dse_given,
            dse_input=dse_input,
            given_bandwidth=1000000,
            tech_node='45nm'
        )
        improved_time = time.time() - start_time
        improved_success = True
    except Exception as e:
        print(f"Error in improved parse_graph: {e}")
        import traceback
        traceback.print_exc()
        improved_cycles = 0
        improved_hw = {"power": 0}
        improved_mem = {}
        improved_time = 0
        improved_success = False
    
    # Print results
    if original_success:
        print("\nOriginal parse_graph results:")
        print(f"Cycles: {original_cycles}")
        print(f"Power: {original_hw.get('power', 0)} mW")
        print(f"Execution time: {original_time:.6f} seconds")
        print("\nHardware allocation:")
        for op, count in original_hw.items():
            if op != 'power':
                print(f"  {op}: {count}")
    else:
        print("\nOriginal parse_graph failed.")
    
    if improved_success:
        print("\nImproved parse_graph results:")
        print(f"Cycles: {improved_cycles}")
        print(f"Power: {improved_hw.get('power', 0)} mW")
        print(f"Execution time: {improved_time:.6f} seconds")
        print("\nHardware allocation:")
        for op, count in improved_hw.items():
            if op != 'power':
                print(f"  {op}: {count}")
    else:
        print("\nImproved parse_graph failed.")
    
    # Calculate differences if both succeeded
    if original_success and improved_success:
        cycle_diff = abs(improved_cycles - original_cycles) / max(1, original_cycles)
        power_diff = abs(improved_hw.get('power', 0) - original_hw.get('power', 0)) / max(1, original_hw.get('power', 0))
        time_diff = improved_time / max(0.001, original_time)
        
        print("\nComparison:")
        print(f"Cycle difference: {cycle_diff*100:.2f}%")
        print(f"Power difference: {power_diff*100:.2f}%")
        print(f"Execution time ratio: {time_diff:.2f}x")
    else:
        cycle_diff = 1.0
        power_diff = 1.0
        time_diff = 1.0
        
        print("\nCannot compare results due to errors.")
    
    # Return results
    return {
        "benchmark": benchmark_name,
        "original_success": original_success,
        "improved_success": improved_success,
        "original_cycles": original_cycles,
        "improved_cycles": improved_cycles,
        "original_power": original_hw.get('power', 0),
        "improved_power": improved_hw.get('power', 0),
        "original_time": original_time,
        "improved_time": improved_time,
        "cycle_diff": cycle_diff,
        "power_diff": power_diff,
        "time_diff": time_diff
    }

def main():
    """Run all benchmarks"""
    print("Testing HLS Benchmarks")
    print("=" * 50)
    
    # Enable verbose output
    os.environ["HLS_VERBOSE"] = "1"
    
    # Test matrix multiplication
    matrix_results = test_benchmark(
        "Matrix Multiplication",
        get_matrix_multiply_implementation(),
        "matrix_multiply",
        {"cycle_time": 2, "unrolling": 2}
    )
    
    # Test FIR filter
    fir_results = test_benchmark(
        "FIR Filter",
        get_fir_filter_implementation(),
        "fir_filter",
        {"cycle_time": 2, "unrolling": 4}
    )
    
    # Test AES
    aes_results = test_benchmark(
        "AES Encryption",
        get_aes_implementation(),
        "aes_encrypt",
        {"cycle_time": 2, "unrolling": 1}
    )
    
    # Summarize results for successful benchmarks
    print("\nBenchmark Summary:")
    print("=" * 50)
    print(f"{'Benchmark':<20} {'Status':<15} {'Cycle Diff':<15} {'Power Diff':<15} {'Time Ratio':<15}")
    print("-" * 80)
    
    def print_result(result):
        status = "SUCCESS" if result['original_success'] and result['improved_success'] else "FAILED"
        print(f"{result['benchmark']:<20} {status:<15} ", end="")
        if result['original_success'] and result['improved_success']:
            print(f"{result['cycle_diff']*100:>6.2f}% {result['power_diff']*100:>13.2f}% {result['time_diff']:>14.2f}x")
        else:
            print("N/A            N/A            N/A")
    
    print_result(matrix_results)
    print_result(fir_results)
    print_result(aes_results)
    
    # Calculate overall assessment for successful benchmarks
    successful_benchmarks = []
    if matrix_results['original_success'] and matrix_results['improved_success']:
        successful_benchmarks.append(matrix_results)
    if fir_results['original_success'] and fir_results['improved_success']:
        successful_benchmarks.append(fir_results)
    if aes_results['original_success'] and aes_results['improved_success']:
        successful_benchmarks.append(aes_results)
    
    if successful_benchmarks:
        avg_cycle_diff = sum(b['cycle_diff'] for b in successful_benchmarks) / len(successful_benchmarks)
        avg_power_diff = sum(b['power_diff'] for b in successful_benchmarks) / len(successful_benchmarks)
        avg_time_diff = sum(b['time_diff'] for b in successful_benchmarks) / len(successful_benchmarks)
        
        print("\nOverall Assessment:")
        print(f"Average cycle difference: {avg_cycle_diff*100:.2f}%")
        print(f"Average power difference: {avg_power_diff*100:.2f}%")
        print(f"Average execution time ratio: {avg_time_diff:.2f}x")
        
        if avg_cycle_diff < 0.2 and avg_power_diff < 0.2:
            print("\nCONCLUSION: The improved HLS implementation provides similar results to the original implementation.")
        else:
            print("\nCONCLUSION: The improved HLS implementation provides significantly different results from the original implementation.")
        
        if avg_time_diff < 1.5:
            print("The improved implementation has comparable performance to the original implementation.")
        else:
            print("The improved implementation is significantly slower than the original implementation.")
    else:
        print("\nNo successful benchmarks to assess.")

if __name__ == "__main__":
    main() 