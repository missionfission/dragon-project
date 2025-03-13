import numpy as np
from scipy import linalg
import time

class SCALELETKF:
    def __init__(self, ensemble_size, state_dim, obs_dim, localization_radius=None):
        """
        Initialize SCALE-LETKF system
        
        Args:
            ensemble_size (int): Number of ensemble members (k)
            state_dim (int): Dimension of state vector
            obs_dim (int): Dimension of observation vector
            localization_radius (float, optional): Localization radius for observation impact
        """
        self.k = ensemble_size
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.loc_radius = localization_radius
        
    def compute_letkf(self, ensemble_states, observations, obs_error_cov, H):
        """
        Compute LETKF analysis
        
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
        X = ensemble_states - x_mean  # Ensemble perturbations
        
        # Matrix-matrix product 1 (52.98% of computation)
        # Transform ensemble states to observation space
        HX = np.dot(H, X.T).T
        y_mean = np.dot(H, x_mean)
        Y = HX - y_mean
        
        # Compute innovation
        d = observations - y_mean
        
        # Prepare matrices for LETKF computation
        R_inv = np.linalg.inv(obs_error_cov)
        
        # Matrix-matrix product 2 (23% of computation)
        C = np.dot(Y.T, np.dot(R_inv, Y)) / (self.k - 1)
        
        # Eigenvalue and eigenvector computation (23.32% of computation)
        eig_vals, eig_vecs = linalg.eigh(C)
        
        # Compute analysis weights
        gamma = 1.0  # Inflation factor (can be adjusted)
        W = eig_vecs.dot(np.diag(1.0 / (eig_vals + gamma)))
        W = W.dot(eig_vecs.T)
        
        # Matrix-vector product 1 (0.5% of computation)
        w_mean = np.dot(W, np.dot(Y.T, np.dot(R_inv, d))) / (self.k - 1)
        
        # Compute analysis ensemble
        W_a = np.sqrt(self.k - 1) * linalg.sqrtm(W)
        X_a = x_mean + np.dot(W_a, X)
        x_a = x_mean + np.dot(X, w_mean)
        
        # Add the analysis mean to each ensemble member
        analysis_ensemble = X_a + x_a
        
        end_time = time.time()
        print(f"LETKF computation completed in {end_time - start_time:.3f} seconds")
        
        return analysis_ensemble
    
    def apply_localization(self, dist_matrix, obs_error_cov):
        """
        Apply Gaspari-Cohn localization to observation error covariance
        
        Args:
            dist_matrix (np.ndarray): Distance matrix between state and observation locations
            obs_error_cov (np.ndarray): Original observation error covariance
            
        Returns:
            np.ndarray: Localized observation error covariance
        """
        if self.loc_radius is None:
            return obs_error_cov
            
        # Gaspari-Cohn localization function
        r = dist_matrix / self.loc_radius
        loc = np.zeros_like(r)
        
        # Region 1: 0 ≤ r ≤ 1
        mask1 = r <= 1
        loc[mask1] = -0.25 * r[mask1]**5 + 0.5 * r[mask1]**4 + \
                     0.625 * r[mask1]**3 - (5/3) * r[mask1]**2 + 1
        
        # Region 2: 1 < r ≤ 2
        mask2 = (r > 1) & (r <= 2)
        loc[mask2] = (1/12) * r[mask2]**5 - 0.5 * r[mask2]**4 + \
                     0.625 * r[mask2]**3 + (5/3) * r[mask2]**2 - 5 * r[mask2] + 4 - \
                     (2/3) * r[mask2]**(-1)
        
        return obs_error_cov * loc

def example_usage():
    """
    Example usage of SCALE-LETKF
    """
    # Example parameters
    ensemble_size = 40
    state_dim = 100
    obs_dim = 50
    
    # Initialize LETKF system
    letkf = SCALELETKF(ensemble_size, state_dim, obs_dim, localization_radius=2.0)
    
    # Generate synthetic data
    np.random.seed(42)
    ensemble_states = np.random.randn(ensemble_size, state_dim)
    observations = np.random.randn(obs_dim)
    obs_error_cov = np.eye(obs_dim)  # Diagonal observation error covariance
    H = np.random.randn(obs_dim, state_dim)  # Random observation operator
    
    # Run LETKF
    updated_ensemble = letkf.compute_letkf(ensemble_states, observations, obs_error_cov, H)
    print(f"Updated ensemble shape: {updated_ensemble.shape}")

if __name__ == "__main__":
    example_usage() 