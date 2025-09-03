# Global imports
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.stats import binned_statistic

# If you want to use your GPU change here
which_device = "cpu"
jax.config.update("jax_default_device", jax.devices(which_device)[0])

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# H0/h = 100 km/s/Mpc expressed in meters
Hubble_over_h = 3.24e-18

# Hour
hour = 3600

# Day
day = 24 * hour

# Year in seconds
yr = 365.25 * day

# Frequency associated with 1yr
f_yr = 1 / yr

# Light speed in m/s
light_speed = 299792458.0

# Parsec in meters
parsec = 3.085677581491367e16

# Megaparsec in meters
Mpc = 1e6 * parsec

# Set the path to the default pulsar parameters
path_to_defaults = os.path.join(os.path.dirname(__file__), "defaults/")

# Set the path to the default pulsar parameters
path_to_default_pulsar_parameters = os.path.join(
    path_to_defaults, "default_pulsar_parameters.yaml"
)

# Set the path to the default pulsar catalog
path_to_default_pulsar_catalog = os.path.join(
    path_to_defaults, "default_catalog.txt"
)

# Set the path to the default pulsar catalog
path_to_default_NANOGrav_positions = os.path.join(
    os.path.dirname(__file__), "defaults/NANOGrav_positions.txt"
)

# Set the path to the default pulsar catalog
path_to_default_NANOGrav_positions = os.path.join(
    os.path.dirname(__file__), "defaults/NANOGrav_positions.txt"
)


def compare_versions(version1, version2):
    """
    Compare two versions of a software package.
    Returns True if version1 >= version2, False otherwise.

    Parameters:
    -----------
    version1 : str
        First version to compare.
    version2 : str
        Second version to compare.

    Returns:
    --------
    is_greater : bool
        True if version1 >= version2, False otherwise.
    """

    version1 = version1.split(".")
    version2 = version2.split(".")

    for i in range(3):
        if int(version1[i]) < int(version2[i]):
            return False
        elif int(version1[i]) > int(version2[i]):
            return True

    return True


def save_table(filename: str, data: dict, verbose: bool = False):
    """
    Save a dict of columns (arrays or lists) into a single .txt file with
    headers. Handles both numeric and string columns.
    """
    if verbose:
        print(f"Saving data to {filename}")

    keys = list(data.keys())
    cols = [np.array(data[k]) for k in keys]

    # Build object array (n_rows x n_cols) → no dtype promotion
    arr = np.empty((len(cols[0]), len(cols)), dtype=object)
    for j, col in enumerate(cols):
        arr[:, j] = col

    # Build format specifiers per column
    fmts = []
    for col in cols:
        if np.issubdtype(col.dtype, np.number):
            fmts.append("%.6g")  # numeric
        else:
            fmts.append("%s")  # string

    # Save
    np.savetxt(filename, arr, fmt=fmts, header=" ".join(keys), comments="# ")


def load_table(filename: str, verbose: bool = False):
    """
    Load a table from a text file into a dictionary.
    """
    if verbose:
        print(f"Loading data from {filename}")

    # Load with genfromtxt: dtype=None lets it infer per-column types
    arr = np.genfromtxt(filename, dtype=None, encoding=None, names=True)

    result = {}
    for name in arr.dtype.names:
        col = arr[name]

        # If column is object/string-like, keep as np.array of strings
        if np.issubdtype(col.dtype, np.number):
            result[name] = jnp.array(col)  # convert to JAX numeric array
        else:
            result[name] = np.array(col, dtype=str)  # keep as strings

    return result


@jax.jit
def dot_product(theta_1, phi_1, theta_2, phi_2):
    """
    Compute the dot product of two unit vectors given their spherical
    coordinates. Theta is the polar angle (co-latitude) and phi is the
    azimuthal angle (longitude). The input angles theta and phi should be given
    in radians.

    Parameters:
    -----------
    theta_1 : Array
        Array of angles in radians representing the polar angle (co-latitude)
        of the first vector.
    phi_1 : Array
        Array of angles in radians representing the azimuthal angle (longitude)
        of the first vector.
    theta_2 : Array
        Array of angles in radians representing the polar angle (co-latitude)
        of the second vector.
    phi_2 : Array
        Array of angles in radians representing the azimuthal angle (longitude)
        of the second vector.

    Returns:
    --------
    dot_product : Array
        Array of dot products computed for the given unit vectors.

    """

    # Sum of the product of x and y components
    term1 = jnp.sin(theta_1) * jnp.sin(theta_2) * jnp.cos(phi_1 - phi_2)

    # The product of the z components
    term2 = jnp.cos(theta_1) * jnp.cos(theta_2)

    return term1 + term2


def characteristic_strain_to_Omega(frequency):
    """
    Computes the dimensionless gravitational wave energy density parameter
    Omega_gw given the characteristic strain at certain frequencies.

    Parameters:
    -----------
    frequency : Array
        Frequency (in Hz) at which the characteristic strain is measured.

    Returns:
    --------
    Omega_gw : Array
        Dimensionless gravitational wave energy density parameter
        at the given frequencies.

    Notes:
    ------
    Hubble_over_h : float
        Constant representing the Hubble parameter divided by the Hubble
        constant.
    """

    return 2 * jnp.pi**2 * frequency**2 / 3 / Hubble_over_h**2


def strain_to_Omega(frequency):
    """
    Computes the dimensionless gravitational wave energy density parameter
    Omega_gw given the strain at certain frequencies.

    Parameters:
    -----------
    frequency : Array
        Frequency (in Hz) at which the characteristic strain is measured.

    Returns:
    --------
    Omega_gw : Array
        Dimensionless gravitational wave energy density parameter
        at the given frequencies.

    Notes:
    ------
    Hubble_over_h : float
        Constant representing the Hubble parameter divided by the Hubble
        constant.
    """

    return 2 * jnp.pi**2 * frequency**3 / 3 / Hubble_over_h**2


# To go from CP delays to characteristic Strain
def hc_from_CP(CP, frequency, T_obs_s):
    """
    Computes the characteristic strain given the Common Process (CP) delays.

    Parameters:
    -----------
    CP : Array
        Amplitude of the Common Process (in seconds^3)
    frequency : Array
        Frequency (in Hz) at which the characteristic strain is measured.
    T_obs_s : float
        Observation time (in seconds)

    Returns:
    --------
    hc : Array
        Characteristic strain at the given frequencies.

    Notes:
    ------
    Hubble_over_h : float
        Constant representing the Hubble parameter divided by the Hubble
        constant.
    """

    return 2 * jnp.sqrt(3) * CP * frequency**1.5 * jnp.pi * jnp.sqrt(T_obs_s)


def load_yaml(path_to_file):
    """
    Loads some inputs from a YAML file (located at the specified path) and
    returns the parsed data as a dictionary.

    Parameters:
    -----------
    path_to_file : str
        Path to the YAML file to load.

    Returns:
    --------
    data : dict
        YAML data loaded from the file.

    Notes:
    ------

    """

    with open(path_to_file, "r+") as stream:
        raw = "".join(stream.readlines())
    return yaml.load(raw, Loader=yaml.SafeLoader)


default_pulsar_parameters = load_yaml(path_to_default_pulsar_parameters)


@jax.jit
def compute_inverse(matrix):
    """
    A function to compute the inverse of a matrix. Applies rescaling to
    maintain numerical stability, especially for near-singular matrices.

    Parameters:
    -----------
    matrix : Array
        Input matrix to compute the inverse of. The shape is assumed to be
        (..., N, N) and the inverse is computed on the last 2 indexes

    Returns:
    --------
    c_inverse : Array
        Inverse of the input matrix.

    Notes:
    ------
    Assumes the input matrix is a square matrix on the last 2 axes.

    """

    # Defines the matrix for the rescaling using the elements on the diagonal
    rescaling_vec = jnp.sqrt(jnp.diagonal(matrix, axis1=-2, axis2=-1))
    rescaling = rescaling_vec[..., :, None] * rescaling_vec[..., None, :]

    return jnp.linalg.inv(matrix / rescaling) / rescaling


@jax.jit
def logdet_kronecker_product(A, B):
    """
    Computes the logarithm of the determinant of a Kronecker product of two
    matrices without explicitly computing the Kronecker product.

    For two matrices A and B, the determinant of their Kronecker product
    A ⊗ B has the property: det(A ⊗ B) = det(A)^dim(B) * det(B)^dim(A).
    Taking the logarithm:
    log(det(A ⊗ B)) = dim(B) * log(det(A)) + dim(A) * log(det(B)).
    This avoids the need to compute the potentially large Kronecker product.

    Parameters:
    -----------
    A : Array
        First matrix in the Kronecker product, with shape (m, m).
    B : Array
        Second matrix in the Kronecker product, with shape (J, J).

    Returns:
    --------
    logdet_kron : float
        The logarithm of the determinant of the Kronecker product A ⊗ B.
    """
    # Get dimensions of the matrices
    m, _ = A.shape
    _, J = B.shape

    # Compute the log determinants of each matrix separately
    _, logdet_A = jnp.linalg.slogdet(A)
    _, logdet_B = jnp.linalg.slogdet(B)

    # Compute the log determinant of the Kronecker product
    # log(det(A ⊗ B)) = dim(B) * log(det(A)) + dim(A) * log(det(B))
    logdet_kron = J * logdet_A + m * logdet_B

    return logdet_kron


def compute_D_IJ_mean(x, y, nbins):
    """
    Computes binned statistics for the correlation coefficients (D_IJ)
    as a function of angular separation between pulsars.

    This function bins the provided data (typically angular separations and
    correlation coefficients) and computes the mean and standard deviation
    in each bin. It is commonly used for analyzing pulsar correlation
    patterns like the Hellings-Downs curve.

    Parameters:
    -----------
    x : Array
        Array of angular separations between pulsars (in radians).
        Typically ranges from 0 to pi.
    y : Array
        Array of correlation coefficients corresponding to the angular
        separations in x.
    nbins : int
        Number of bins to divide the range [0, pi] into.

    Returns:
    --------
    bin_means : Array
        Mean value of y in each bin.
    bin_std : Array
        Standard deviation of y in each bin.
    bin_centers : Array
        Central value of each bin (average of bin edges).
    """
    # Compute mean values in each bin
    bin_means, bin_edges, _ = binned_statistic(
        x, y, statistic="mean", bins=np.linspace(0, np.pi, nbins + 1)
    )

    # Compute standard deviations in each bin
    bin_std, _, _ = binned_statistic(
        x, y, statistic="std", bins=np.linspace(0, np.pi, nbins + 1)
    )

    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_means, bin_std, bin_centers


def compute_pulsar_average_D_IJ(ang_list, D_IJ_list, nbins=20):
    """
    Computes the average pulsar correlation coefficients across multiple
    realizations, binned by angular separation.

    This function processes multiple realizations of pulsar correlation data,
    filters out very small angular separations, and bins the data for each
    realization. It's useful for analyzing the consistency of correlation
    patterns (like Hellings-Downs curves) across different simulations or
    data subsets.

    Parameters:
    -----------
    ang_list : Array
        Array of angular separations between pulsars for multiple realizations.
        Shape: (n_realizations, n_pulsars, n_pulsars) where each slice [i,:,:]
        is a matrix of angular separations between pulsar pairs.
    D_IJ_list : Array
        Array of correlation coefficients corresponding to the angular
        separations in ang_list, with the same shape.
    nbins : int, optional
        Number of bins to divide the range [0, pi] into. Default is 20.

    Returns:
    --------
    x_avg : Array
        Array of bin centers for each realization.
        Shape: (n_realizations, nbins)
    y_avg : Array
        Array of mean correlation values for each realization and bin.
        Shape: (n_realizations, nbins)
    """
    # Initialize lists to store binned data for each realization
    x_avg = []
    y_avg = []

    # Get the number of realizations from the shape of the input
    n_realizations = ang_list.shape[0]

    # Process each realization separately
    for i in range(n_realizations):
        # Flatten the angular separations and correlations for this realization
        x = ang_list[i].flatten()
        y = D_IJ_list[i].flatten()

        # Filter out very small angular separations (self-correlations)
        y = y[x > 1e-5]
        x = x[x > 1e-5]

        # Compute binned statistics for this realization
        bin_means, _, bin_centers = compute_D_IJ_mean(x, y, nbins)

        # Store the results
        x_avg.append(bin_centers)
        y_avg.append(bin_means)

    # Convert lists to arrays for return
    return np.array(x_avg), np.array(y_avg)


def get_R(samples):
    """
    Computes the Gelman-Rubin (GR) statistic for convergence assessment. The
    GR statistic is a convergence diagnostic used to assess whether multiple
    Markov chains have converged to the same distribution. Values close to 1
    indicate convergence. For details see
    https://en.wikipedia.org/wiki/Gelman-Rubin_statistic

    Parameters:
    -----------
    samples : numpy.ndarray
        Array containing MCMC samples with dimensions
        (N_steps, N_chains, N_parameters).

    Returns:
    --------
    R : numpy.ndarray
        Array containing the Gelman-Rubin statistics indicating convergence for
        the different parameters. Values close to 1 indicate convergence.

    """

    # Get the shapes
    N_steps, N_chains, N_parameters = samples.shape

    # Chain means
    chain_mean = np.mean(samples, axis=0)

    # Global mean
    global_mean = np.mean(chain_mean, axis=0)

    # Variance between the chain means
    variance_of_means = (
        N_steps
        / (N_chains - 1)
        * np.sum((chain_mean - global_mean[None, :]) ** 2, axis=0)
    )

    # Variance of the individual chain across all chains
    intra_chain_variance = np.std(samples, axis=0, ddof=1) ** 2

    # And its averaged value over the chains
    mean_intra_chain_variance = np.mean(intra_chain_variance, axis=0)

    # First term
    term_1 = (N_steps - 1) / N_steps

    # Second term
    term_2 = variance_of_means / mean_intra_chain_variance / N_steps

    # This is the R (as a vector running on the paramters)
    return term_1 + term_2
