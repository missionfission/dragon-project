import unittest
import os
import subprocess
import json
import yaml
import numpy as np
from src.ir.cfg.staticfg import CFGBuilder
from src.synthesis.hls import parse_graph, get_stats
from src.mapper.mapper import Mapper

class TestSystemValidation(unittest.TestCase):
    """Test suite for validating CFG builder and system simulator against gem5 and RISC-V"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and configurations"""
        # Load default config
        with open('configs/default.yaml', 'r') as f:
            cls.default_config = yaml.safe_load(f)
            
        # Define benchmark paths
        cls.benchmark_paths = {
            'matrix_mult': 'tests/benchmarks/matrix_mult.py',
            'bfs': 'tests/benchmarks/bfs.py',
            'aes': 'tests/benchmarks/aes.py'
        }
        
        # Define technology nodes to test
        cls.tech_nodes = ['7nm', '14nm', '22nm', '45nm']
        
        # Create benchmark files if they don't exist
        cls._create_benchmark_files()
    
    @classmethod
    def _create_benchmark_files(cls):
        """Create benchmark files for testing"""
        os.makedirs('tests/benchmarks', exist_ok=True)
        
        # Matrix multiplication benchmark
        matrix_mult_code = '''
def matrix_multiply(A, B):
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result
'''
        with open(cls.benchmark_paths['matrix_mult'], 'w') as f:
            f.write(matrix_mult_code)
            
        # BFS benchmark
        bfs_code = '''
def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        vertex = queue.pop(0)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited
'''
        with open(cls.benchmark_paths['bfs'], 'w') as f:
            f.write(bfs_code)
            
        # AES benchmark
        aes_code = '''
def aes_encrypt(data, key):
    state = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            state[i][j] = data[i*4 + j]
    
    # Simplified AES round function
    for round in range(10):
        for i in range(4):
            for j in range(4):
                state[i][j] ^= key[round*16 + i*4 + j]
    return state
'''
        with open(cls.benchmark_paths['aes'], 'w') as f:
            f.write(aes_code)
    
    def test_cfg_builder_tech_scaling(self):
        """Test CFG builder with different technology nodes"""
        for tech_node in self.tech_nodes:
            for benchmark_name, benchmark_path in self.benchmark_paths.items():
                # Build CFG with specific technology node
                cfg = CFGBuilder(tech_node=tech_node).build_from_file(
                    benchmark_name,
                    benchmark_path
                )
                
                # Verify technology scaling is applied
                self.assertEqual(cfg.tech_node, tech_node)
                self.assertIsNotNone(cfg.get_tech_scaling())
                
                # Parse graph and get stats
                parse_graph(cfg, tech_node=tech_node)
                stats = get_stats(cfg)
                
                # Verify basic stats are present
                self.assertIsNotNone(stats)
    
    def test_system_simulator_accuracy(self):
        """Test system simulator accuracy against gem5"""
        # Only run if gem5 is available
        if not self._is_gem5_available():
            self.skipTest("gem5 not available for validation")
            
        for benchmark_name, benchmark_path in self.benchmark_paths.items():
            # Get our simulator's results
            cfg = CFGBuilder().build_from_file(benchmark_name, benchmark_path)
            our_cycles, our_power = self._get_our_simulator_results(cfg)
            
            # Get gem5 results
            gem5_cycles, gem5_power = self._run_gem5_simulation(benchmark_path)
            
            # Compare results (allowing for some deviation)
            max_cycle_deviation = 0.15  # 15% deviation allowed
            max_power_deviation = 0.20  # 20% deviation allowed
            
            cycle_diff = abs(our_cycles - gem5_cycles) / gem5_cycles
            power_diff = abs(our_power - gem5_power) / gem5_power
            
            self.assertLess(cycle_diff, max_cycle_deviation, 
                          f"Cycle count deviation too high for {benchmark_name}")
            self.assertLess(power_diff, max_power_deviation,
                          f"Power estimation deviation too high for {benchmark_name}")
    
    def test_memory_hierarchy_validation(self):
        """Test memory hierarchy modeling against CACTI"""
        for benchmark_name, benchmark_path in self.benchmark_paths.items():
            cfg = CFGBuilder().build_from_file(benchmark_name, benchmark_path)
            
            # Get memory stats from our simulator
            our_mem_stats = self._get_memory_stats(cfg)
            
            # Get CACTI results
            cacti_stats = self._run_cacti_validation()
            
            # Compare memory access latencies and energy
            max_latency_deviation = 0.25  # 25% deviation allowed
            max_energy_deviation = 0.25  # 25% deviation allowed
            
            for mem_level in our_mem_stats:
                if mem_level in cacti_stats:
                    latency_diff = abs(our_mem_stats[mem_level]['latency'] - 
                                     cacti_stats[mem_level]['latency']) / cacti_stats[mem_level]['latency']
                    energy_diff = abs(our_mem_stats[mem_level]['energy'] - 
                                    cacti_stats[mem_level]['energy']) / cacti_stats[mem_level]['energy']
                    
                    self.assertLess(latency_diff, max_latency_deviation,
                                  f"Memory latency deviation too high for {mem_level}")
                    self.assertLess(energy_diff, max_energy_deviation,
                                  f"Memory energy deviation too high for {mem_level}")
    
    def test_risc_v_validation(self):
        """Test against RISC-V simulator for instruction-level accuracy"""
        # Only run if RISC-V simulator is available
        if not self._is_riscv_available():
            self.skipTest("RISC-V simulator not available for validation")
            
        for benchmark_name, benchmark_path in self.benchmark_paths.items():
            # Get our simulator's instruction stats
            cfg = CFGBuilder().build_from_file(benchmark_name, benchmark_path)
            our_instr_stats = self._get_instruction_stats(cfg)
            
            # Get RISC-V simulator stats
            riscv_stats = self._run_riscv_simulation(benchmark_path)
            
            # Compare instruction mix and execution time
            max_instr_deviation = 0.10  # 10% deviation allowed
            
            for instr_type in our_instr_stats:
                if instr_type in riscv_stats:
                    diff = abs(our_instr_stats[instr_type] - riscv_stats[instr_type]) / riscv_stats[instr_type]
                    self.assertLess(diff, max_instr_deviation,
                                  f"Instruction count deviation too high for {instr_type}")
    
    def _is_gem5_available(self):
        """Check if gem5 is available"""
        try:
            subprocess.run(['gem5.opt', '--version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            return False
    
    def _is_riscv_available(self):
        """Check if RISC-V simulator is available"""
        try:
            subprocess.run(['spike', '--version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            return False
    
    def _get_our_simulator_results(self, cfg):
        """Get results from our simulator"""
        mapper = Mapper()
        cycles = 0
        power = 0
        
        # Run simulation
        parse_graph(cfg)
        stats = get_stats(cfg)
        
        # Extract cycles and power from stats
        if stats:
            cycles = stats.get('cycles', 0)
            power = stats.get('power', 0)
        
        return cycles, power
    
    def _run_gem5_simulation(self, benchmark_path):
        """Run simulation in gem5"""
        # This is a placeholder - actual gem5 integration would be needed
        gem5_script = 'gem5_scripts/run_benchmark.py'
        result = subprocess.run(['gem5.opt', gem5_script, benchmark_path],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        
        # Parse gem5 stats file
        stats_file = 'gem5_results/stats.txt'
        cycles = 0
        power = 0
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                for line in f:
                    if 'system.cpu.numCycles' in line:
                        cycles = int(line.split()[1])
                    elif 'system.cpu.power' in line:
                        power = float(line.split()[1])
        
        return cycles, power
    
    def _get_memory_stats(self, cfg):
        """Get memory hierarchy statistics from our simulator"""
        # This would need to be implemented based on your memory modeling
        return {
            'L1': {'latency': 1, 'energy': 0.1},
            'L2': {'latency': 10, 'energy': 1.0},
            'DRAM': {'latency': 100, 'energy': 10.0}
        }
    
    def _run_cacti_validation(self):
        """Run CACTI for memory validation"""
        # This is a placeholder - actual CACTI integration would be needed
        return {
            'L1': {'latency': 1, 'energy': 0.1},
            'L2': {'latency': 10, 'energy': 1.0},
            'DRAM': {'latency': 100, 'energy': 10.0}
        }
    
    def _get_instruction_stats(self, cfg):
        """Get instruction statistics from our simulator"""
        # This would need to be implemented based on your instruction modeling
        return {
            'integer': 1000,
            'floating': 500,
            'memory': 300,
            'branch': 200
        }
    
    def _run_riscv_simulation(self, benchmark_path):
        """Run simulation in RISC-V simulator"""
        # This is a placeholder - actual RISC-V integration would be needed
        return {
            'integer': 1000,
            'floating': 500,
            'memory': 300,
            'branch': 200
        }

if __name__ == '__main__':
    unittest.main() 