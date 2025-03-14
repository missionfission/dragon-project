def matrix_multiply_hls(A, B, M, K, N):
    """HLS-friendly matrix multiplication"""
    C = [[0 for _ in range(N)] for _ in range(M)]
    
    for i in range(M):
        for j in range(N):
            sum_val = 0
            for k in range(K):
                sum_val = sum_val + A[i][k] * B[k][j]
            C[i][j] = sum_val
    
    return C

def matrix_transpose_hls(A, rows, cols):
    """HLS-friendly matrix transpose"""
    AT = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            AT[j][i] = A[i][j]
    return AT

def compute_mean_hls(ensemble_states, state_dim, ensemble_size):
    """HLS-friendly ensemble mean computation"""
    mean = [0 for _ in range(state_dim)]
    for i in range(state_dim):
        for j in range(ensemble_size):
            mean[i] = mean[i] + ensemble_states[j][i]
        mean[i] = mean[i] / ensemble_size
    return mean

# Test parameters
ensemble_size = 3
state_dim = 2
obs_dim = 2

# Create test data
ensemble_states = [
    [1, 2],
    [3, 4],
    [5, 6]
]

# Compute ensemble mean
mean = compute_mean_hls(ensemble_states, state_dim, ensemble_size)

# Create observation operator (H matrix)
H = [[1, 0], [0, 1]]  # Identity for simplicity

# Transform ensemble to observation space
HX = matrix_multiply_hls(H, matrix_transpose_hls(ensemble_states, ensemble_size, state_dim), 
                        obs_dim, state_dim, ensemble_size)

print_result = True  # Set to True to print results 