# Global
import os
import yaml
import numpy as np

import wigners

import jax
import jax.numpy as jnp
from scipy.special import sph_harm


jax.config.update("jax_enable_x64", True)


# If you want to use your GPU change here
which_device = "cpu"
jax.config.update("jax_default_device", jax.devices(which_device)[0])


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


def get_sort_indexes(l_max, indexing="xy"):
    """
    Given the maximum ell value, this function returns the indexes to sort the
    spherical harmonics coefficients as returned by map2alm of healpy (see
    https://healpy.readthedocs.io/en/latest/). The coefficients are sorted in
    the following way:

    - First all the negative m values for a given ell are sorted with
        decreasing order.
    - Then the m=0 values are sorted.
    - Finally all the positive m values for a given ell are sorted with
        increasing order.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.
    indexing : str
        Indexing scheme to use for lm. Default is 'xy' which assumes sorting
        according to m, l.

    Returns:
    --------
    l_grid : numpy.ndarray
        Array of l values.
    m_grid : numpy.ndarray
        Array of m values.
    ll : numpy.ndarray
        Array of l values corresponding to the sorted indexes.
    mm : numpy.ndarray
        Array of m values corresponding to the sorted indexes.
    sort_indexes : numpy.ndarray
        Array of indexes to sort the spherical harmonics coefficients.

    """

    # Create arrays for l and m
    l_values = np.arange(l_max + 1)
    m_values = np.arange(l_max + 1)

    # Create a grid of all possible (l, m) pairs
    l_grid, m_grid = np.meshgrid(l_values, m_values, indexing=indexing)  # type: ignore

    # Flatten the grid
    l_flat = l_grid.flatten()
    m_flat = m_grid.flatten()

    # Select only the m values that are allowed for a given ell\
    l_grid = l_flat[np.abs(m_flat) <= l_flat]
    m_grid = m_flat[np.abs(m_flat) <= l_flat]

    # Create a vector with all the m<0 and then all the m>=0
    mm = np.append(-np.flip(m_grid[m_grid > 0]), m_grid)

    # Create a vector with all the ls corresponding to mm
    ll = np.append(np.flip(l_grid[m_grid > 0]), l_grid)

    # Return the sorted indexes
    return l_grid, m_grid, ll, mm, np.lexsort((mm, ll))


def complex_to_real_conversion(
    spherical_harmonics, l_max, m_grid, m_positive, indexing="xy"
):
    """
    Converts the complex spherical harmonics (or the coefficients) to real
    spherical harmonics (or the coefficients).

    Parameters:
    -----------
    spherical_harmonics : numpy.ndarray
        2D (or 1D) array of complex spherical harmonics coefficients.
        If 2D, the shape is (lm, pp), where lm runs over l,m (with m > 0), and
        pp is the number of theta and phi values. If 1D, the shape is (lm,).
    l_max : int
        Maximum ell value.
    m_grid : numpy.ndarray
        Array of m values.
    m_positive : numpy.ndarray
        2D (or 1D) array of positive m values, this should have the same shape
        as spherical_harmonics
    indexing : str
        Indexing scheme to use for lm. Default is 'xy' which assumes sorting
        according to m, l.

    Returns:
    --------
    all_spherical_harmonics : numpy.ndarray
        2D (or 1D) array of real spherical harmonics coefficients. If 2D, the
        shape is (lm, pp), where lm runs over l,m (with -l <= m <= l), and pp
        is the number of theta and phi values. If 1D, the shape is (lm,).

    """

    # Create arrays the m_values and the indexes to sort
    inds = get_sort_indexes(l_max, indexing=indexing)

    # Pick only m = 0
    zero = spherical_harmonics[m_grid == 0.0].real

    # Build the positive m values
    positive = (
        np.sqrt(2.0)
        * (-1.0) ** m_positive
        * spherical_harmonics[m_grid > 0.0].real
    )

    # Build the negative m values
    negative = (
        np.sqrt(2.0)
        * (-1.0) ** m_positive
        * spherical_harmonics[m_grid > 0.0].imag
    )

    # Concatenate the negative, zero and positive m values
    all_spherical_harmonics = np.concatenate(
        (np.flip(negative, axis=0), zero, positive), axis=0
    )

    # Return spherical harmonics (coefficients) sorted by l and m
    return all_spherical_harmonics[inds[-1]]


def real_to_complex_old(clm_real):
    """
    Converts the real spherical harmonics coefficients to complex spherical ....
    """

    l_max = np.rint(np.sqrt(len(clm_real))).astype(int) - 1
    clm_complex = np.zeros(
        (l_max + 1, l_max + 1), dtype=np.cdouble
    )  # for non-negative M in complex spherical harmonics basis
    i = 0
    for L in range(l_max + 1):
        for M in range(L + 1):
            if M == 0:
                clm_complex[L, M] = clm_real[i]
                i += 1
            else:
                clm_complex[L, M] = (
                    ((-1) ** M)
                    * (-1j * clm_real[i] + clm_real[i + 1])
                    / np.sqrt(2.0)
                )
                i += 2

    return clm_complex


def real_to_complex(clm_real):
    """
    Converts the real spherical harmonics coefficients to complex spherical ....
    """

    # l_max is the maximum ell value, the +1 used later is to count ell = 0 too
    l_max = np.rint(np.sqrt(len(clm_real))).astype(int) - 1

    # Initialize the complex (M > 0 only) spherical harmonics coefficients
    clm_complex = np.zeros((l_max + 1, l_max + 1), dtype=np.cdouble)

    i = 0
    for L in range(l_max + 1):
        for M in range(L + 1):
            if M == 0:
                clm_complex[L, M] = clm_real[i]
                i += 1
            else:
                clm_complex[L, M] = (
                    ((-1) ** M)
                    * (-1j * clm_real[i] + clm_real[i + 1])
                    / np.sqrt(2.0)
                )
                i += 2

    return clm_complex


def complex_to_real_old(clm_complex):
    l_max = np.shape(clm_complex)[0] - 1
    clm_real = np.zeros((l_max + 1) ** 2)
    i = 0
    for ell in range(l_max + 1):
        for m in range(ell + 1):
            if m == 0:
                clm_real[i] = clm_complex[ell, m].real
                i += 1
            else:
                clm_real[i] = (
                    -((-1) ** m) * np.sqrt(2.0) * clm_complex[ell, m].imag
                )  # negative m
                clm_real[i + 1] = (
                    ((-1) ** m) * np.sqrt(2.0) * clm_complex[ell, m].real
                )  # positive m
                i += 2

    return clm_real


def complex_to_real(clm_complex):
    l_max = np.shape(clm_complex)[0] - 1
    clm_real = np.zeros((l_max + 1) ** 2)
    i = 0
    for ell in range(l_max + 1):
        for m in range(ell + 1):
            if m == 0:
                clm_real[i] = clm_complex[ell, m].real
                i += 1
            else:
                clm_real[i] = (
                    -((-1) ** m) * np.sqrt(2.0) * clm_complex[ell, m].imag
                )  # negative m
                clm_real[i + 1] = (
                    ((-1) ** m) * np.sqrt(2.0) * clm_complex[ell, m].real
                )  # positive m
                i += 2

    return clm_real


def sqrt_to_lin_conversion(gLM_grid, l_max_lin=-1, real_basis_in=True):

    if real_basis_in:
        gLM_complex = real_to_complex(gLM_grid)
    else:
        gLM_complex = gLM_grid

    l_max_sqrt = np.shape(gLM_complex)[0] - 1

    if l_max_lin < 0:
        l_max_lin = 2 * l_max_sqrt

    clm_complex = np.zeros((l_max_lin + 1, l_max_lin + 1), dtype=np.cdouble)
    gLnegM_complex = ((-1) ** np.arange(l_max_sqrt + 1))[
        np.newaxis, :
    ] * np.conj(gLM_complex)

    for ell in range(l_max_lin + 1):
        for m in range(ell + 1):
            for L1 in range(l_max_sqrt + 1):
                L2_grid_all = np.arange(l_max_sqrt + 1)
                mask_L2 = (np.abs(L1 - L2_grid_all) <= ell) * (
                    L2_grid_all >= ell - L1
                )  # conditions from lmin and l_max_lin selection rules
                L2_grid = L2_grid_all[mask_L2]
                for L2 in L2_grid:
                    cg0 = wigners.clebsch_gordan(L1, 0, L2, 0, ell, 0)
                    if cg0 != 0.0:
                        prefac = np.sqrt(
                            (2.0 * L1 + 1.0)
                            * (2.0 * L2 + 1.0)
                            / (4.0 * np.pi * (2.0 * ell + 1.0))
                        )
                        M1_grid_all = np.arange(-L1, L1 + 1)
                        M2_grid_all = m - M1_grid_all
                        mask_M = np.abs(M2_grid_all) <= L2
                        M1_grid = M1_grid_all[mask_M]
                        M2_grid = M2_grid_all[mask_M]
                        for iM, M1 in enumerate(M1_grid):
                            M2 = M2_grid[iM]
                            cg1 = wigners.clebsch_gordan(L1, M1, L2, M2, ell, m)
                            b1 = (
                                gLM_complex[L1, M1]
                                if M1 >= 0
                                else gLnegM_complex[L1, -M1]
                            )
                            b2 = (
                                gLM_complex[L2, M2]
                                if M2 >= 0
                                else gLnegM_complex[L2, -M2]
                            )
                            clm_complex[ell, m] += prefac * cg0 * cg1 * b1 * b2

    clm_real = complex_to_real(clm_complex)

    return clm_real


def wrapper_sqrt_to_lin_conversion(arg_grid):
    if len(arg_grid.shape) > 1:  # not real_basis_in for sqrt_to_lin_conversion
        l_max_lin = int(np.real(arg_grid[0, 1]))
        arg_grid[0, 1] = 0.0

        return sqrt_to_lin_conversion(
            arg_grid, l_max_lin=l_max_lin, real_basis_in=False
        )
    return sqrt_to_lin_conversion(
        arg_grid[:-1], l_max_lin=int(arg_grid[-1]), real_basis_in=True
    )


def get_spherical_harmonics(l_max, theta, phi):
    """
    Compute the spherical harmonics for a given maximum ell value and for a
    given set of theta and phi values.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.
    theta : numpy.ndarray or jax.numpy.ndarray
        Array of polar angles (co-latitudes).
    phi : numpy.ndarray or jax.numpy.ndarray
        Array of azimuthal angles (longitudes).

    Returns:
    --------
    all_spherical_harmonics : numpy.ndarray
        2D array of spherical harmonics computed for the given maximum ell
        value, and theta and phi values. The shape will be (lm, pp), where pp
        is the number of theta and phi values, and lm = (l_max + 1)**2 is the
        number of spherical harmonics coefficients.

    """

    # Create arrays the m_values and the indexes to sort
    inds = get_sort_indexes(l_max)

    # Unpack m_grid and sorted_indexes
    l_grid = inds[0]
    m_grid = inds[1]

    # Compute all the spherical harmonics
    spherical_harmonics = sph_harm(
        m_grid[:, None], l_grid[:, None], phi[None, :], theta[None, :]
    )

    # Select only the m values that are allowed for a given ell
    m_positive = m_grid[m_grid > 0.0]

    # Return sorted
    return complex_to_real_conversion(
        spherical_harmonics, l_max, m_grid, m_positive[:, None]
    )


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
