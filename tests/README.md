# Testing Flow for CFG Builder and System Simulator

This directory contains the testing framework for validating the CFG builder and system simulator against established simulators like gem5 and RISC-V.

## Directory Structure

```
tests/
├── benchmarks/           # Benchmark programs
├── cfg_builder/          # CFG builder specific tests
├── system_sim/          # System simulator specific tests
├── validation/          # Cross-validation tests
├── gem5_scripts/        # gem5 simulation scripts
├── riscv_scripts/       # RISC-V simulation scripts
├── cacti_scripts/       # CACTI validation scripts
└── test_system_validation.py  # Main test suite
```

## Prerequisites

1. gem5 simulator
   - Installation: Follow instructions at http://www.gem5.org/documentation/
   - Required version: >= 21.0

2. RISC-V toolchain and Spike simulator
   - Installation: `apt-get install gcc-riscv64-unknown-elf`
   - Spike simulator: Build from https://github.com/riscv-software-src/riscv-isa-sim

3. CACTI
   - Download from: https://github.com/HewlettPackard/cacti
   - Build following repository instructions

## Test Suite Components

1. CFG Builder Validation
   - Technology node scaling tests
   - Control flow analysis accuracy
   - Memory access pattern detection

2. System Simulator Validation
   - Cycle count accuracy vs gem5
   - Power estimation validation
   - Memory hierarchy modeling

3. Memory Hierarchy Validation
   - CACTI-based validation
   - Access latency verification
   - Energy consumption accuracy

4. Instruction-Level Validation
   - RISC-V ISA compliance
   - Instruction mix analysis
   - Execution time correlation

## Running Tests

1. Run all tests:
   ```bash
   python -m unittest tests/test_system_validation.py
   ```

2. Run specific test category:
   ```bash
   python -m unittest tests/test_system_validation.py -k test_cfg_builder
   python -m unittest tests/test_system_validation.py -k test_system_simulator
   python -m unittest tests/test_system_validation.py -k test_memory_hierarchy
   python -m unittest tests/test_system_validation.py -k test_risc_v
   ```

3. Run with detailed output:
   ```bash
   python -m unittest -v tests/test_system_validation.py
   ```

## Benchmark Suite

The test suite includes the following benchmarks:

1. Matrix Multiplication
   - Dense matrix operations
   - Memory access patterns
   - Computational intensity

2. Breadth-First Search (BFS)
   - Graph traversal
   - Irregular memory access
   - Control flow complexity

3. AES Encryption
   - Bit manipulation
   - Regular computation patterns
   - Security workload characteristics

## Validation Metrics

1. Performance Metrics
   - Cycle count deviation (target: <15%)
   - Instruction count accuracy (target: <10%)
   - Memory access latency (target: <25%)

2. Power and Energy Metrics
   - Power estimation deviation (target: <20%)
   - Energy per operation (target: <25%)
   - Memory energy consumption (target: <25%)

3. Resource Utilization
   - Memory bandwidth utilization
   - Compute unit utilization
   - Cache hit rates

## Adding New Tests

1. Create benchmark:
   - Add source file to `tests/benchmarks/`
   - Implement in both Python and C (for RISC-V)
   - Document expected behavior

2. Add test cases:
   - Extend `TestSystemValidation` class
   - Define validation criteria
   - Add to appropriate test category

3. Update validation scripts:
   - Modify gem5/RISC-V/CACTI scripts as needed
   - Update parsing logic for new metrics
   - Document changes in this README

## Troubleshooting

1. gem5 Issues:
   - Check gem5 installation and paths
   - Verify benchmark compatibility
   - Check simulation parameters

2. RISC-V Issues:
   - Verify toolchain installation
   - Check binary compilation
   - Validate Spike simulator setup

3. CACTI Issues:
   - Verify installation and dependencies
   - Check configuration file syntax
   - Validate technology parameters

## Contributing

1. Follow test structure
2. Document validation criteria
3. Include benchmark characteristics
4. Update README as needed

## References

1. gem5 Documentation: http://www.gem5.org/documentation/
2. RISC-V Specifications: https://riscv.org/technical/specifications/
3. CACTI Documentation: https://github.com/HewlettPackard/cacti 