import numpy as np
from scipy import linalg
import time

def obs_select(state_point, obs_points, radius):
    """
    Select observations within a given radius of a state point.
    Designed for parallel processing - each state point can be processed independently.
    
    Args:
        state_point (np.ndarray): Single state point coordinates
        obs_points (np.ndarray): Array of observation points coordinates
        radius (float): Selection radius
        
    Returns:
        np.ndarray: Boolean mask of selected observations
    """
    # Compute distances - vectorized operation
    distances = np.sqrt(np.sum((obs_points - state_point)**2, axis=1))
    
    # Create selection mask
    mask = distances <= radius
    
    return mask

def matrix_multiply_block(A, B, block_size=32):
    """
    Block matrix multiplication optimized for HLS implementation.
    Breaks down matrix multiplication into smaller blocks for better parallelization.
    
    Args:
        A (np.ndarray): First matrix
        B (np.ndarray): Second matrix
        block_size (int): Size of blocks for tiled multiplication
        
    Returns:
        np.ndarray: Result of matrix multiplication
    """
    M, K = A.shape
    K, N = B.shape
    C = np.zeros((M, N))
    
    # Adjust block sizes if they're larger than matrix dimensions
    block_size_m = min(block_size, M)
    block_size_n = min(block_size, N)
    block_size_k = min(block_size, K)
    
    # Iterate over blocks
    for i in range(0, M, block_size_m):
        i_end = min(i + block_size_m, M)
        for j in range(0, N, block_size_n):
            j_end = min(j + block_size_n, N)
            # Initialize block result
            block_result = np.zeros((i_end - i, j_end - j))
            
            for k in range(0, K, block_size_k):
                k_end = min(k + block_size_k, K)
                # Compute block multiplication
                block_result += np.dot(A[i:i_end, k:k_end], B[k:k_end, j:j_end])
            
            C[i:i_end, j:j_end] = block_result
    
    return C

def svd_block(X, max_iterations=30):
    """
    SVD computation optimized for HLS implementation using power iteration method.
    This implementation is more suitable for hardware synthesis than traditional SVD.
    
    Args:
        X (np.ndarray): Input matrix
        max_iterations (int): Maximum number of power iterations
        
    Returns:
        tuple: (U, S, V) - Left singular vectors, Singular values, Right singular vectors
    """
    m, n = X.shape
    k = min(m, n)
    
    # Initialize matrices
    U = np.zeros((m, k))
    S = np.zeros(k)
    V = np.zeros((n, k))
    
    # Temporary matrix for deflation
    X_temp = X.copy()
    
    # Compute singular vectors/values one at a time
    for i in range(k):
        # Initialize random vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        # Power iteration
        for _ in range(max_iterations):
            # Matrix-vector multiplications
            u = X_temp @ v
            s = np.linalg.norm(u)
            if s > 0:
                u = u / s
            v = X_temp.T @ u
            s = np.linalg.norm(v)
            if s > 0:
                v = v / s
        
        # Compute singular value
        u = X_temp @ v
        s = np.linalg.norm(u)
        if s > 0:
            u = u / s
        
        # Store results
        U[:, i] = u
        S[i] = s
        V[:, i] = v
        
        # Deflation
        X_temp -= s * np.outer(u, v)
    
    return U, S, V

class SCALELETKF_Parallel:
    def __init__(self, ensemble_size, state_dim, obs_dim, localization_radius=None, block_size=32):
        """
        Initialize parallel SCALE-LETKF system
        
        Args:
            ensemble_size (int): Number of ensemble members (k)
            state_dim (int): Dimension of state vector
            obs_dim (int): Dimension of observation vector
            localization_radius (float, optional): Localization radius
            block_size (int): Block size for matrix operations
        """
        self.k = ensemble_size
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.loc_radius = localization_radius
        self.block_size = block_size
    
    def compute_letkf_parallel(self, ensemble_states, observations, obs_error_cov, H):
        """
        Parallel implementation of LETKF analysis
        
        Args:
            ensemble_states (np.ndarray): Shape (k, state_dim) ensemble state matrix
            observations (np.ndarray): Shape (obs_dim,) observation vector
            obs_error_cov (np.ndarray): Shape (obs_dim, obs_dim) observation error covariance
            H (np.ndarray): Shape (obs_dim, state_dim) observation operator
        
        Returns:
            np.ndarray: Updated ensemble states
        """
        start_time = time.time()
        
        # Compute ensemble mean and perturbations
        x_mean = np.mean(ensemble_states, axis=0)
        X = ensemble_states - x_mean
        
        # Transform ensemble to observation space using block matrix multiplication
        HX = matrix_multiply_block(H, X.T, self.block_size).T
        y_mean = matrix_multiply_block(H, x_mean.reshape(-1, 1), self.block_size).flatten()
        Y = HX - y_mean
        
        # Compute innovation
        d = observations - y_mean
        
        # Prepare matrices for LETKF computation
        R_inv = np.linalg.inv(obs_error_cov)
        
        # Compute covariance matrix using block operations
        C = matrix_multiply_block(Y.T, matrix_multiply_block(R_inv, Y, self.block_size), 
                                self.block_size) / (self.k - 1)
        
        # Compute SVD using block method
        U, S, V = svd_block(C)
        
        # Compute analysis weights
        gamma = 1.0
        W = matrix_multiply_block(
            U, 
            matrix_multiply_block(
                np.diag(1.0 / (S + gamma)),
                V.T,
                self.block_size
            ),
            self.block_size
        )
        
        # Compute final analysis
        w_mean = matrix_multiply_block(
            W,
            matrix_multiply_block(
                Y.T,
                matrix_multiply_block(R_inv, d.reshape(-1, 1), self.block_size),
                self.block_size
            ),
            self.block_size
        ).flatten() / (self.k - 1)
        
        # Compute analysis ensemble
        W_a = np.sqrt(self.k - 1) * linalg.sqrtm(W)
        X_a = matrix_multiply_block(W_a, X, self.block_size)
        x_a = x_mean + matrix_multiply_block(X.T, w_mean.reshape(-1, 1), self.block_size).flatten()
        
        # Combine results
        analysis_ensemble = X_a + x_a
        
        end_time = time.time()
        print(f"Parallel LETKF computation completed in {end_time - start_time:.3f} seconds")
        
        return analysis_ensemble

def example_usage_parallel():
    """
    Example usage of parallel SCALE-LETKF
    """
    # Example parameters
    ensemble_size = 40
    state_dim = 100
    obs_dim = 50
    block_size = 32
    
    # Initialize parallel LETKF system
    letkf = SCALELETKF_Parallel(
        ensemble_size=ensemble_size,
        state_dim=state_dim,
        obs_dim=obs_dim,
        localization_radius=2.0,
        block_size=block_size
    )
    
    # Generate synthetic data
    np.random.seed(42)
    ensemble_states = np.random.randn(ensemble_size, state_dim)
    observations = np.random.randn(obs_dim)
    obs_error_cov = np.eye(obs_dim)
    H = np.random.randn(obs_dim, state_dim)
    
    # Run parallel LETKF
    updated_ensemble = letkf.compute_letkf_parallel(
        ensemble_states,
        observations,
        obs_error_cov,
        H
    )
    print(f"Updated ensemble shape: {updated_ensemble.shape}")

if __name__ == "__main__":
    example_usage_parallel() 