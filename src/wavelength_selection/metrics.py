import numpy as np
import scipy
from numpy.typing import NDArray


def nrmse(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute the Normalized Root Mean Squared Error (NRMSE) between the true and predicted concentration map.

    Formula from https://www.researchgate.net/publication/272755125_Optimal_wavelength_combinations_for_near-infrared_spectroscopic_monitoring_of_changes_in_brain_tissue_hemoglobin_and_cytochrome_c_oxidase_concentrations

    Args:
        y_true (NDArray): gt concentration map
        y_pred (NDArray): predicted concentration map

    Returns:
        float: NRMSE value
    """
    numerator = np.mean((y_pred - y_true) ** 2, axis=2)
    denominator = np.mean(y_true**2, axis=2)
    # TODO: test different kind of normalization?
    # Avoid division by zero
    denominator[denominator == 0] = 1e-12

    return np.mean(
        np.sqrt(
            numerator / denominator,
        ),
    )


def rmse(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute the Root Mean Squared Error (RMSE) between the true and predicted concentration/ attenuation map.

    Args:
        y_true (NDArray): gt concentration map
        y_pred (NDArray): predicted concentration map

    Returns:
        float: RMSE value
    """
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def optimum_index_factor(M: NDArray) -> float:
    """
    Compute the Optimum Index Factor (OIF) for a given matrix M (pixels x wavelength bands).

    Parameters:
    M (np.ndarray): A 2D array of shape (pixels, wavelength bands).

    Returns:
    float: The computed OIF value.
    """
    # Compute standard deviations of each spectral band
    variances = np.var(M, axis=0, ddof=1)  # ddof=1 for sample variance

    # Compute correlation matrix
    corr_matrix = np.corrcoef(M, rowvar=False)

    # Extract the upper triangle of the absolute correlation matrix, excluding the diagonal
    upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
    corr_sum = np.sum(np.abs(corr_matrix[upper_triangle_indices]))

    # Compute OIF
    oif = np.sum(variances) / corr_sum if corr_sum != 0 else np.inf
    return oif


def cond_num_metric(M: NDArray) -> float:
    """
    Compute the condition number of the matrix M.

    Args:
        M (NDArray): 2D matrix

    Returns:
        float: condition number of M
    """
    return np.linalg.cond(M)


def svdvals_prod_metric(M: NDArray) -> float:
    """
    Compute the product of the singular values of the matrix M.

    Args:
        M (NDArray): 2D matrix

    Returns:
        float: Product of the singular values
    """
    # return np.min(scipy.linalg.svdvals(M))
    return np.prod(scipy.linalg.svdvals(M))


def svdvals_sum_metric(M: NDArray) -> float:
    """
    Compute the sum of the singular values of the matrix M.

    Args:
        M (NDArray): 2D matrix

    Returns:
        float: Product of the singular values
    """
    # return np.min(scipy.linalg.svdvals(M))
    return np.sum(scipy.linalg.svdvals(M))


def min_svdvals_metric(M: NDArray) -> float:
    """
    Compute the minimum singular value of the matrix M.

    Args:
        M (NDArray): 2D matrix

    Returns:
        float: Minimum singular value
    """
    return np.min(scipy.linalg.svdvals(M))
