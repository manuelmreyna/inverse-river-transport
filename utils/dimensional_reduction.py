import numpy as np
import scipy as sp
def KL_decomposition(samples_field, n_lmbds, dt):

    # Compute the mean field across all samples
    field_mean = np.mean(samples_field, axis=0)

    # Compute the anomaly matrix (deviation from the mean)
    field_diff = samples_field - field_mean

    # Compute the covariance matrix of the anomalies
    cov_field = np.cov(field_diff.T)

    # Compute eigenvalues and eigenvectors of the covariance matrix (KL decomposition)
    lambdas, phis = sp.linalg.eigh(cov_field)

    # Sort eigenvalues and eigenvectors in descending order
    lambdas, phis = lambdas[::-1], phis[:, ::-1].T

    # Project the anomaly matrix onto the eigenvectors (KL coefficients)
    etas = (field_diff @ phis.T) / np.sqrt(lambdas)

    # Scale the coefficients by sqrt(lambda) (for reconstruction)
    Zs = etas * np.sqrt(lambdas)

    # Return the mean field, first n_lmbds KL modes, and corresponding KL coefficients

    return field_mean, phis[:n_lmbds]/np.sqrt(dt), lambdas[:n_lmbds]*dt , Zs[:,:n_lmbds]*np.sqrt(dt)