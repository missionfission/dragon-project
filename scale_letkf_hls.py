def obs_select_hls(state_point, obs_points, radius, obs_dim, coord_dim):
    """
    HLS-friendly observation selection function
    Uses basic loops and arithmetic instead of NumPy operations
    """
    mask = [False] * obs_dim
    
    # Compute distances using basic arithmetic
    for i in range(obs_dim):
        dist_sq = 0
        for j in range(coord_dim):
            diff = state_point[j] - obs_points[i][j]
            dist_sq += diff * diff
        
        # Compare with radius squared to avoid sqrt
        if dist_sq <= radius * radius:
            mask[i] = True
    
    return mask

def matrix_multiply_hls(A, B, M, K, N):
    """
    HLS-friendly matrix multiplication
    Uses basic loops instead of NumPy operations
    """
    C = [[0.0 for _ in range(N)] for _ in range(M)]
    
    for i in range(M):
        for j in range(N):
            sum_val = 0.0
            for k in range(K):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
    
    return C

def matrix_multiply_block_hls(A, B, M, K, N, block_size):
    """
    HLS-friendly blocked matrix multiplication
    Designed for hardware implementation with configurable block size
    """
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
    """
    HLS-friendly matrix transpose
    """
    AT = [[0.0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            AT[j][i] = A[i][j]
    return AT

def power_iteration_hls(A, max_iter, M, N):
    """
    HLS-friendly power iteration method for dominant eigenpair computation
    """
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
    """
    HLS-friendly SVD computation using power iteration
    Returns only the dominant singular value and vectors
    """
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
        """
        HLS-friendly ensemble mean computation
        """
        mean = [0.0] * self.state_dim
        for i in range(self.state_dim):
            for j in range(self.k):
                mean[i] += ensemble_states[j][i]
            mean[i] /= self.k
        return mean
    
    def compute_perturbations_hls(self, ensemble_states, mean):
        """
        HLS-friendly perturbation computation
        """
        X = [[0.0 for _ in range(self.state_dim)] for _ in range(self.k)]
        for i in range(self.k):
            for j in range(self.state_dim):
                X[i][j] = ensemble_states[i][j] - mean[j]
        return X
    
    def compute_letkf_step_hls(self, ensemble_states, observations, obs_error_cov, H):
        """
        HLS-friendly LETKF computation step
        Breaks down the algorithm into basic operations
        """
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

def example_usage_hls():
    """
    Example usage of HLS-friendly LETKF implementation
    """
    # Example parameters
    ensemble_size = 4  # Small size for demonstration
    state_dim = 6
    obs_dim = 3
    block_size = 2
    
    # Initialize system
    letkf = SCALELETKF_HLS(ensemble_size, state_dim, obs_dim, block_size)
    
    # Generate simple test data
    ensemble_states = [[float(i+j) for j in range(state_dim)] for i in range(ensemble_size)]
    observations = [float(i) for i in range(obs_dim)]
    obs_error_cov = [[1.0 if i==j else 0.0 for j in range(obs_dim)] for i in range(obs_dim)]
    H = [[1.0 if i==j else 0.0 for j in range(state_dim)] for i in range(obs_dim)]
    
    # Run HLS-friendly LETKF
    result = letkf.compute_letkf_step_hls(ensemble_states, observations, obs_error_cov, H)
    print("Analysis ensemble shape:", len(result), "x", len(result[0]))

if __name__ == "__main__":
    example_usage_hls() 