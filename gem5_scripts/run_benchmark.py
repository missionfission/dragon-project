#!/usr/bin/env python3

import os
import sys
import argparse
import m5
from m5.objects import *
from m5.util import addToPath
from gem5.runtime import get_runtime_isa

def create_system():
    """Create a basic system configuration for gem5"""
    system = System()
    
    # Set up clock domain
    system.clk_domain = SrcClockDomain()
    system.clk_domain.clock = '2GHz'
    system.clk_domain.voltage_domain = VoltageDomain()
    
    # Set up memory mode
    system.mem_mode = 'timing'
    system.mem_ranges = [AddrRange('512MB')]
    
    # Create CPU
    system.cpu = TimingSimpleCPU()
    system.cpu.icache = L1ICache()
    system.cpu.dcache = L1DCache()
    
    # Create memory bus
    system.membus = SystemXBar()
    
    # Connect CPU caches to memory bus
    system.cpu.icache.connectCPU(system.cpu)
    system.cpu.dcache.connectCPU(system.cpu)
    
    system.cpu.icache.connectBus(system.membus)
    system.cpu.dcache.connectBus(system.membus)
    
    # Connect memory port
    system.cpu.createInterruptController()
    system.cpu.interrupts[0].pio = system.membus.mem_side_ports
    system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
    system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports
    
    # Create and connect system port
    system.system_port = system.membus.cpu_side_ports
    
    # Create memory controller and connect to memory bus
    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR3_1600_8x8()
    system.mem_ctrl.dram.range = system.mem_ranges[0]
    system.mem_ctrl.port = system.membus.mem_side_ports
    
    return system

def simulate_benchmark(benchmark_path):
    """Run a benchmark in gem5"""
    # Create system
    system = create_system()
    
    # Create process
    process = Process()
    process.cmd = [benchmark_path]
    
    # Set process for CPU
    system.cpu.workload = process
    system.cpu.createThreads()
    
    # Create root
    root = Root(full_system=False, system=system)
    
    # Instantiate simulation
    m5.instantiate()
    
    # Run simulation
    print(f"Starting simulation of {benchmark_path}")
    exit_event = m5.simulate()
    print(f"Exiting @ tick {m5.curTick()}\n")
    
    # Write stats
    m5.stats.dump()

def main():
    parser = argparse.ArgumentParser(description='Run benchmark in gem5')
    parser.add_argument('benchmark', help='Path to benchmark file')
    args = parser.parse_args()
    
    if not os.path.exists(args.benchmark):
        print(f"Error: Benchmark file {args.benchmark} not found")
        sys.exit(1)
    
    # Create results directory
    os.makedirs('gem5_results', exist_ok=True)
    
    # Run simulation
    simulate_benchmark(args.benchmark)

if __name__ == '__main__':
    main() 