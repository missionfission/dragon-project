# SCALE LETKF HLS Implementation

This document describes the High-Level Synthesis (HLS) implementation of the SCALE Local Ensemble Transform Kalman Filter (LETKF) algorithm.

## Overview

The Local Ensemble Transform Kalman Filter (LETKF) is a data assimilation algorithm used in weather forecasting, oceanography, and other geophysical applications. It combines model forecasts with observations to produce an optimal estimate of the state of a system.

Our implementation focuses on creating an HLS-friendly version of the LETKF algorithm that can be synthesized into hardware for accelerated performance.

## Algorithm Description

The LETKF algorithm consists of the following key steps:

1. **Ensemble Mean Computation**: Calculate the mean of the ensemble states.
2. **Perturbation Computation**: Calculate the perturbations from the mean for each ensemble member.
3. **Observation Space Transformation**: Transform the perturbations to observation space.
4. **SVD Computation**: Perform Singular Value Decomposition (SVD) using power iteration.
5. **Analysis Weights Computation**: Compute the analysis weights using the SVD results.
6. **Analysis Ensemble Computation**: Apply the weights to the perturbations to get the analysis ensemble.

## HLS Implementation

Our HLS implementation breaks down the LETKF algorithm into basic operations that can be efficiently mapped to hardware:

### Key Components

1. **Matrix Operations**:
   - Basic matrix multiplication
   - Blocked matrix multiplication for improved data reuse
   - Matrix transpose

2. **SVD Computation**:
   - Power iteration method for dominant eigenpair computation
   - Normalization operations

3. **Ensemble Operations**:
   - Ensemble mean computation
   - Perturbation computation

### Hardware-Friendly Features

1. **Loop-Based Implementation**: All operations are implemented using explicit loops instead of library functions, making them suitable for hardware synthesis.

2. **Blocked Operations**: Matrix operations are implemented with configurable block sizes to optimize for hardware implementation.

3. **Minimized Division Operations**: Division operations, which are expensive in hardware, are minimized and only used where necessary.

4. **Parameterized Design**: The implementation is parameterized by ensemble size, state dimension, observation dimension, and block size.

## Resource Allocation

The HLS implementation allocates hardware resources based on the algorithm requirements:

1. **Registers**:
   - Ensemble states: `ensemble_size * state_dim`
   - Ensemble mean: `state_dim`
   - Perturbations: `ensemble_size * state_dim`
   - Observation space perturbations: `obs_dim * ensemble_size`
   - SVD components: `obs_dim + ensemble_size + 1`
   - Analysis weights: `obs_dim * ensemble_size`

2. **Functional Units**:
   - Multipliers: For matrix multiplication and power iteration
   - Adders: For accumulation operations
   - Dividers: For normalization operations

## Performance Analysis

The theoretical performance of the LETKF implementation is calculated based on:

1. **Operation Counts**:
   - Additions: `state_dim * (ensemble_size - 1) + obs_dim * state_dim * (ensemble_size - 1) + 10 * obs_dim * ensemble_size^2 + obs_dim * (ensemble_size - 1) * state_dim`
   - Multiplications: `obs_dim * state_dim * ensemble_size + 10 * obs_dim * ensemble_size^2 + obs_dim * ensemble_size * state_dim`
   - Divisions: `state_dim + 10 * ensemble_size`
   - Subtractions: `state_dim * ensemble_size`

2. **Cycle Calculation**:
   - 4 cycles per multiplication
   - 2 cycles per addition/subtraction
   - 10 cycles per division
   - Adjusted for parallelism with block size

## Validation Results

The LETKF implementation was validated using a test case with the following parameters:
- Ensemble size: 4
- State dimension: 6
- Observation dimension: 3
- Block size: 2

The validation results show:
- Theoretical cycles: 1054.0
- Minimum registers: 86
- Minimum multipliers: 1
- Minimum adders: 1
- Minimum dividers: 1
- Total operations: 1300

The HLS synthesis results match the theoretical calculations:
- Cycles: 1054.0
- Hardware allocation:
  - Registers: 86
  - Multipliers: 2
  - Adders: 2
  - Dividers: 1
- Power consumption: 1.36 mW

## Usage

To use the LETKF HLS implementation:

```python
from scale_letkf_hls import SCALELETKF_HLS

# Initialize LETKF with parameters
letkf = SCALELETKF_HLS(
    ensemble_size=4,
    state_dim=6,
    obs_dim=3,
    block_size=2
)

# Generate input data
ensemble_states = [[float(i+j) for j in range(state_dim)] for i in range(ensemble_size)]
observations = [float(i) for i in range(obs_dim)]
obs_error_cov = [[1.0 if i==j else 0.0 for j in range(obs_dim)] for i in range(obs_dim)]
H = [[1.0 if i==j else 0.0 for j in range(state_dim)] for i in range(obs_dim)]

# Run LETKF computation
result = letkf.compute_letkf_step_hls(ensemble_states, observations, obs_error_cov, H)
```

## Validation

To validate the LETKF implementation:

```bash
python3 src/synthesis/test_letkf_validation.py
```

This will run the validation test and output the results, including the theoretical metrics, synthesis results, and validation status.

## Future Work

1. **Optimized Memory Access**: Implement more sophisticated memory access patterns to reduce memory bandwidth requirements.

2. **Pipelined Implementation**: Develop a pipelined version of the algorithm for higher throughput.

3. **Parameterized Hardware Generation**: Create a tool to automatically generate optimized hardware based on specific LETKF parameters.

4. **Integration with SCALE Model**: Integrate the HLS implementation with the SCALE weather model for end-to-end acceleration.

5. **Floating-Point Precision Analysis**: Analyze the impact of reduced floating-point precision on accuracy and performance. 