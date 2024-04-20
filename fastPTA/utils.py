# Global
import os
import yaml
import numpy as np

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

# If you want to use your GPU change here
jax.config.update("jax_default_device", jax.devices("cpu")[0])


# Just some constants
log_A_curn_default = -13.94
log_gamma_curn_default = 2.71
integration_points = 10000


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

# Set the path to the default pulsar parameters
path_to_default_pulsar_parameters = os.path.join(
    os.path.dirname(__file__), "defaults/default_pulsar_parameters.yaml"
)

# Set the path to the default pulsar catalog
path_to_default_pulsar_catalog = os.path.join(
    os.path.dirname(__file__), "defaults/default_catalog.txt"
)


def characteristic_strain_to_Omega(frequency):
    """
    Computes the dimensionless gravitational wave energy density parameter
    Omega_gw given the characteristic strain at certain frequencies.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Frequency (in Hz) at which the characteristic strain is measured.

    Returns:
    --------
    Omega_gw : numpy.ndarray or jax.numpy.ndarray
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
    frequency : numpy.ndarray or jax.numpy.ndarray
        Frequency (in Hz) at which the characteristic strain is measured.

    Returns:
    --------
    Omega_gw : numpy.ndarray or jax.numpy.ndarray
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
    CP : numpy.ndarray or jax.numpy.ndarray
        Amplitude of the Common Process (in seconds^3)

    frequency : numpy.ndarray or jax.numpy.ndarray
        Frequency (in Hz) at which the characteristic strain is measured.

    T_obs_s : float
        Observation time (in seconds)

    Returns:
    --------
    hc : numpy.ndarray or jax.numpy.ndarray
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
    matrix : numpy.ndarray or jax.numpy.ndarray
        Input matrix to compute the inverse of. The shape is assumed to be
        (..., N, N) and the inverse is computed on the last 2 indexes

    Returns:
    --------
    c_inverse : numpy.ndarray or jax.numpy.ndarray
        Inverse of the input matrix.

    Notes:
    ------
    Assumes the input matrix is a square matrix on the last 2 axes.

    """

    # Defines the matrix for the rescaling using the elements on the diagonal
    rescaling_vec = jnp.sqrt(jnp.diagonal(matrix, axis1=-2, axis2=-1))
    rescaling = rescaling_vec[..., :, None] * rescaling_vec[..., None, :]

    return jnp.linalg.inv(matrix / rescaling) / rescaling


def get_R(samples):
    """
    Computes the Gelman-Rubin (GR) statistic for convergence assessment. The GR
    statistic is a convergence diagnostic used to assess whether multiple
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
