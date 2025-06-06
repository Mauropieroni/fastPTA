# Global
import os
import yaml
import scipy
import numpy as np
import healpy as hp

from wigners import clebsch_gordan

import jax
import jax.numpy as jnp


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


if compare_versions(scipy.__version__, "1.17.0"):
    from scipy.special import sph_harm_y

else:
    from scipy.special import sph_harm

    def sph_harm_y(ell, m, theta, phi):
        return sph_harm(m, ell, phi, theta)


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


def get_l_max_real(real_spherical_harmonics):
    """
    Given the real spherical harmonics coefficients, this function returns the
    maximum ell value.

    Parameters:
    -----------
    real_spherical_harmonics : numpy.ndarray
        Array of real spherical harmonics coefficients. If dimension is > 1, lm
        must be the first index

    Returns:
    --------
    l_max : int
        Maximum ell value.

    """

    return int(np.sqrt(len(real_spherical_harmonics)) - 1)


def get_l_max_complex(complex_spherical_harmonics):
    """
    Given the complex spherical harmonics coefficients, this function returns
    the maximum ell value.

    Parameters:
    -----------
    complex_spherical_harmonics : numpy.ndarray
        Array of complex spherical harmonics coefficients. If dimension is > 1,
        lm must be the first index

    Returns:
    --------
    l_max : int
        Maximum ell value.

    """

    return int(np.sqrt(1.0 + 8.0 * len(complex_spherical_harmonics)) / 2 - 1.5)


def get_n_coefficients_complex(l_max):
    """
    Given the maximum ell value, this function returns the number of spherical
    harmonics coefficients for the complex representation.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.

    Returns:
    --------
    n_coefficients : int
        Number of spherical harmonics coefficients.

    """

    return int((l_max + 1) * (l_max + 2) / 2)


def get_n_coefficients_real(l_max):
    """
    Given the maximum ell value, this function returns the number of spherical
    harmonics coefficients for the real representation.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.

    Returns:
    --------
    n_coefficients : int
        Number of spherical harmonics coefficients.

    """

    return int((l_max + 1) ** 2)


def get_sort_indexes(l_max):
    """
    Given the maximum ell value, this function returns the indexes to sort the
    indexes of the spherical harmonics coefficients when going from real to
    complex representation and viceversa.

    The complex representation is assumed to be sorted as in map2alm of healpy
    (see https://healpy.readthedocs.io/en/latest/) i.e. according to (m, l).
    The ouput allows to sort the real representation according to (l, m).

    Parameters:
    -----------
    l_max : int
        Maximum ell value.

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
    l_grid, m_grid = np.meshgrid(l_values, m_values, indexing="xy")

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


def spherical_harmonics_projection(quantity, l_max):
    """
    Compute the spherical harmonics projection of a given quantity. Quantity
    should be an array in pixel space, and compatible with healpy (see
    https://healpy.readthedocs.io/en/latest/). The spherical harmonics
    coefficients are sorted as described in the get_sort_indexes function in
    utils.

    Parameters:
    -----------
    quantity : numpy.ndarray
        Array of quantities to project on spherical harmonics.
    l_max : int
        Maximum ell value.

    Returns:
    --------
    real_alm : numpy.ndarray
        Array of real spherical harmonics coefficients, with len
        lm = (l_max + 1)**2 is the number of spherical harmonics coefficients.

    """

    # Get the complex alm coefficients.
    # These are sorted with the m values and are only for m>=0
    alm = hp.map2alm(quantity, lmax=l_max)

    # Create arrays the m_values and the indexes to sort
    inds = get_sort_indexes(l_max)

    # Unpack m_grid and sorted_indexes
    m_grid = inds[1]
    sorted_indexes = inds[-1]

    # Compute the sign for the m > 0 values
    sign = (-1.0) ** m_grid[m_grid > 0]

    # The m != 0 values are multiplied by sqrt(2) and then take real/imag part
    positive_alm = np.sqrt(2.0) * alm[m_grid > 0]

    # Build the real alm selecting imaginary and real part and sort in the two
    # blocks in ascending order
    negative_m = np.flip(sign * positive_alm.imag)
    positive_m = sign * positive_alm.real

    # Concatenate the negative, zero and positive m values
    real_alm = np.concatenate((negative_m, alm[m_grid == 0.0].real, positive_m))

    # Sort with the indexes
    return real_alm[sorted_indexes]


def complex_to_real_conversion(spherical_harmonics):
    """
    Converts the complex spherical harmonics (or the coefficients) to real
    spherical harmonics (or the coefficients).

    Parameters:
    -----------
    spherical_harmonics : numpy.ndarray
        2D (or 1D) array of complex spherical harmonics coefficients.
        If 2D, the shape is (lm, pp), where lm runs over l,m (with m > 0), and
        pp is the number of theta and phi values. If 1D, the shape is (lm,).

    Returns:
    --------
    all_spherical_harmonics : numpy.ndarray
        2D (or 1D) array of real spherical harmonics coefficients. If 2D, the
        shape is (lm, pp), where lm runs over l,m (with -l <= m <= l), and pp
        is the number of theta and phi values. If 1D, the shape is (lm,).

    """

    # Get the right value of l_max from the input complex coefficients
    l_max = get_l_max_complex(spherical_harmonics)

    # Create arrays the m_values and the indexes to sort
    _, m_grid, _, _, sort_indexes = get_sort_indexes(l_max)

    # Pick only m = 0
    zero_m = spherical_harmonics[m_grid == 0.0].real

    # Compute the sign for the m > 0 values
    sign = (-1.0) ** m_grid[m_grid > 0]

    # The m != 0 values are multiplied by sqrt(2) and then take real/imag part
    positive_spherical = np.sqrt(2.0) * spherical_harmonics[m_grid > 0.0]

    # Build the m > 0 values
    positive_m = np.einsum("i,i...->i...", sign, positive_spherical.real)

    # Build the m < 0 values
    negative_m = np.einsum("i,i...->i...", sign, positive_spherical.imag)

    # Concatenate the negative, zero and positive m values
    all_spherical_harmonics = np.concatenate(
        (np.flip(negative_m, axis=0), zero_m, positive_m), axis=0
    )

    # Return spherical harmonics (coefficients) sorted by l and m
    return all_spherical_harmonics[sort_indexes]


def real_to_complex_conversion(real_spherical_harmonics):
    """
    Converts the real spherical harmonics (or the coefficients) back to complex
    spherical harmonics (or the coefficients).

    Parameters:
    -----------
    real_spherical_harmonics : numpy.ndarray
        1D array of real spherical harmonics coefficients.
        The shape is (lm,), where lm runs over l,m (with -l <= m <= l).
    l_max : int
        Maximum ell value.

    Returns:
    --------
    complex_spherical_harmonics : numpy.ndarray
        1D array of complex spherical harmonics coefficients.
        The shape is (lm,), where lm runs over l,m (with m >= 0).
    """

    # Get the right value of l_max from the input real coefficients
    l_max = get_l_max_real(real_spherical_harmonics)

    # Get sort indexes
    _, _, _, mm, sort_indexes = get_sort_indexes(l_max)

    # Reorder the input real coefficients to the original order
    ordered_real_spherical_harmonics = np.zeros_like(real_spherical_harmonics)
    ordered_real_spherical_harmonics[sort_indexes] = real_spherical_harmonics

    # Split the ordered real coefficients into negative, zero, and positive
    # m values
    zero_m = ordered_real_spherical_harmonics[mm == 0]
    positive_m = ordered_real_spherical_harmonics[mm > 0]
    negative_m = ordered_real_spherical_harmonics[mm < 0]

    # Compute the corresponding m values for positive and negative m
    m_positive = mm[mm > 0]

    # Reconstruct the complex coefficients
    complex_positive_m = (positive_m + 1j * negative_m[::-1]) / (
        np.sqrt(2.0) * (-1.0) ** m_positive
    )

    # Combine zero and positive m values to form the full complex coefficients
    complex_spherical_harmonics = np.concatenate(
        (zero_m, complex_positive_m), axis=0
    )

    return complex_spherical_harmonics


def get_real_spherical_harmonics(l_max, theta, phi):
    """
    Compute the real spherical harmonics for a given maximum ell value and for
    a given set of theta and phi values.

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
    spherical_harmonics = sph_harm_y(
        l_grid[:, None], m_grid[:, None], theta[None, :], phi[None, :]
    )

    # Return sorted
    return complex_to_real_conversion(spherical_harmonics)


def get_CL_from_real_clm(clm_real):
    """
    Compute the angular power spectrum from the spherical harmonics
    coefficients.

    Parameters:
    -----------
    clm_real : numpy.ndarray
        Array of spherical harmonics coefficients, if dimension > 1 the first
        axis must run over the coefficients.

    Returns:
    --------
    CL : numpy.ndarray
        Array of angular power spectrum.

    """

    # Get the shape of the input coefficients
    clm_shape = clm_real.shape

    # Get the maximum ell value
    l_max = get_l_max_real(clm_real)

    # Compute the angular power spectrum
    CL = np.zeros(tuple([l_max + 1] + list(clm_shape[1:])))

    # A counter for the coefficients used up to that value of ell
    i = 0

    for ell in range(0, l_max + 1):
        # Average the square of the coefficients for all ms at fixed ell
        CL[ell] = np.mean(clm_real[i : i + 2 * ell + 1] ** 2, axis=0)

        # Update the counter
        i += 2 * ell + 1

    return CL


def get_Cl_limits(
    means,
    cov,
    shape_params,
    n_points=int(1e4),
    limit_cl=0.95,
    max_iter=100,
    prior=5.0 / (4.0 * np.pi),
):
    """
    Compute the upper limit on the angular power spectrum from the means and
    covariance matrix of the spherical harmonics coefficients.

    Parameters:
    -----------
    means : numpy.ndarray
        Array of means for the spherical harmonics coefficients.
    cov : numpy.ndarray
        Array of covariance matrix for the spherical harmonics coefficients.
    shape_params : int
        Number of parameters for the SGWB shape.
    n_points : int, optional
        Number of points to generate.
    limit_cl : float, optional
        Quantile to compute the upper limit.
    max_iter : int, optional
        Maximum number of iterations to generate points.
    prior : float, optional
        Prior value to restrict the points.

    Returns:
    --------
    Cl_limits : numpy.ndarray
        Array of upper limits on the angular power spectrum from the covariance
    Cl_limits_prior : numpy.ndarray
        Array of upper limits on the angular power spectrum including the prior

    """

    # Generate gaussian data from the covariance matrix
    data = np.random.multivariate_normal(
        means, cov, n_points, check_valid="ignore", tol=1e-4
    )

    # Select only the points that are within the prior
    data_prior = data[np.max(np.abs(data[:, shape_params:]), axis=-1) <= prior]

    # Initialize the counter and the length of the data
    i_add = 0
    len_restricted = len(data_prior)

    # Use a while loop to generate enough points
    while len_restricted < n_points and i_add < max_iter:

        # Generate more points
        add_data = np.random.multivariate_normal(
            means, cov, 10 * n_points, check_valid="ignore", tol=1e-4
        )

        # Select only the points that are within the prior and append
        data_prior = np.append(
            data_prior,
            add_data[
                np.max(np.abs(add_data[:, shape_params:]), axis=-1) <= prior
            ],
            axis=0,
        )

        # Update the counter and the length of the data
        len_restricted = len(data_prior)
        i_add += 1

    # Compute the angular power spectra without and with the prior
    correlations_lm = get_CL_from_real_clm(data.T[shape_params - 1 :])[1:]
    correlations_lm_prior = get_CL_from_real_clm(
        data_prior.T[shape_params - 1 :]
    )[1:]

    # Compute the upper limits without the prior
    Cl_limits = np.quantile(correlations_lm, limit_cl, axis=-1)

    # And with the prior if there are enough points
    if len_restricted == 0:
        Cl_limits_prior = np.nan * Cl_limits

    else:
        Cl_limits_prior = np.quantile(correlations_lm_prior, limit_cl, axis=-1)

    return Cl_limits, Cl_limits_prior


def sqrt_to_lin_conversion(gLM_grid, l_max_lin=-1, real_basis_input=False):
    """
    Convert the sqrt basis to the linear basis.

    Parameters:
    -----------
    gLM_grid : numpy.ndarray
        Array of sqrt basis coefficients.
    l_max_lin : int, optional
        Maximum ell value for the linear basis. Default is -1.
    real_basis_input : bool, optional
        If True, the input is in the real basis. Default is False.

    Returns:
    --------
    clm_real : numpy.ndarray
        Array of real coefficients in the linear basis.

    """

    # If gLM are in the real basis convert the complex basis
    if real_basis_input:
        gLM_complex = real_to_complex_conversion(gLM_grid)
    else:
        gLM_complex = gLM_grid

    # Get the maximum ell value for the sqrt basis
    l_max_sqrt = get_l_max_complex(gLM_complex)

    # Get the grid of all possible L values for the sqrt basis
    L_grid_all = np.arange(l_max_sqrt + 1)

    # If the maximum ell value for the linear basis is not provided, set it
    if l_max_lin < 0:
        l_max_lin = 2 * l_max_sqrt

    # Get the number of coefficients for the linear basis
    n_coefficients = get_n_coefficients_complex(l_max_lin)

    # Initialize the array for the linear basis
    clm_complex = np.zeros(n_coefficients, dtype=np.cdouble)

    # Get the indexes for the linear and sqrt basis
    l_lin, m_lin, _, _, _ = get_sort_indexes(l_max_lin)
    l_sqrt, m_sqrt, _, _, _ = get_sort_indexes(l_max_sqrt)

    # Compute the sign for the gLM with m < 0
    gLnegM_complex = (-1) ** np.abs(m_sqrt) * np.conj(gLM_complex)

    for ind_linear in range(len(m_lin)):
        # Get the values of ell and m
        m = m_lin[ind_linear]
        ell = l_lin[ind_linear]

        for L1 in L_grid_all:

            # Build a mask using the conditions from the selection rules
            mask_L2 = (np.abs(L1 - L_grid_all) <= ell) * (
                L_grid_all >= ell - L1
            )

            # Run over the L2 allowed by the mask
            for L2 in L_grid_all[mask_L2]:
                # Compute the Clebsch-Gordan coefficient for all ms = 0
                cg0 = clebsch_gordan(L1, 0, L2, 0, ell, 0)

                # If the coefficient is not zero compute the prefactor
                if cg0 != 0.0:
                    prefac = np.sqrt(
                        (2.0 * L1 + 1.0)
                        * (2.0 * L2 + 1.0)
                        / (4.0 * np.pi * (2.0 * ell + 1.0))
                    )

                    # These are all the values of M1 to use
                    M1_grid_all = np.arange(-L1, L1 + 1)

                    # Enforce m +M1 + M2 = 0
                    M2_grid_all = m - M1_grid_all

                    # Check that the values of M2 are consistent with L2
                    mask_M = np.abs(M2_grid_all) <= L2

                    # Apply the mask
                    M1_grid = M1_grid_all[mask_M]
                    M2_grid = M2_grid_all[mask_M]

                    for iM in range(len(M1_grid)):
                        # Get the values of M1 and M2
                        M1 = M1_grid[iM]
                        M2 = M2_grid[iM]

                        # Compute the Clebsch-Gordan coefficient for ms neq 0
                        cg1 = clebsch_gordan(L1, M1, L2, M2, ell, m)

                        # Mask to get the corresponding value of gLM_complex
                        b1_mask = (l_sqrt == L1) & (m_sqrt == np.abs(M1))
                        b2_mask = (l_sqrt == L2) & (m_sqrt == np.abs(M2))

                        # Get the values of gLM_complex for the given L and M
                        b1 = (
                            gLM_complex[b1_mask]
                            if M1 >= 0
                            else gLnegM_complex[b1_mask]
                        )

                        b2 = (
                            gLM_complex[b2_mask]
                            if M2 >= 0
                            else gLnegM_complex[b2_mask]
                        )

                        # Multiply everything and sum to the right index
                        clm_complex[ind_linear] += prefac * cg0 * cg1 * b1 * b2

    return complex_to_real_conversion(clm_complex)


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
