# Global imports
import healpy as hp
import numpy as np
import scipy

# Local imports
from fastPTA.utils import compare_versions

if compare_versions(scipy.__version__, "1.15.0"):
    from scipy.special import sph_harm_y

else:
    from scipy.special import sph_harm

    def sph_harm_y(ell, m, theta, phi):
        return sph_harm(m, ell, phi, theta)


def get_l_max_real(real_spherical_harmonics):
    """
    Given the real spherical harmonics coefficients, this function returns the
    maximum ell value.

    Parameters:
    -----------
    real_spherical_harmonics : Array
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
    complex_spherical_harmonics : Array
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
    l_grid : Array
        Array of l values.
    m_grid : Array
        Array of m values.
    ll : Array
        Array of l values corresponding to the sorted indexes.
    mm : Array
        Array of m values corresponding to the sorted indexes.
    sort_indexes : Array
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
    quantity : Array
        Array of quantities to project on spherical harmonics.
    l_max : int
        Maximum ell value.

    Returns:
    --------
    real_alm : Array
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


def project_correlation_spherical_harmonics(quantity, l_max):
    """
    Compute the spherical harmonics projection of the correlation matrix (in
    the pulsar-pulsar axes) of a given quantity.

    TBD: This function should use the fact that quantity is symmetric in the
    pulsar pulsar indexes to reduce computations and increase efficiency.

    Parameters:
    -----------
    quantity : Array
        3D array to be projected on spherical harmonics. Should have shape
        (N, N, P), where N is the number of pulsars and P is the number of
        pixels, which should be compatible with healpy
        (see https://healpy.readthedocs.io/en/latest/).
    l_max : int
        Maximum ell value.

    Returns:
    --------
    real_alm : Array
        3D array of real spherical harmonics coefficients.
        It has shape (lm, N, N), where N is the number of pulsars and
        lm = (l_max + 1)**2 is the number of spherical harmonics coefficients.

    """

    # Get the shape of the quantity to project on spherical harmonics
    shape = list(quantity.shape)

    # Reshape quantity so that it can be passed to hp.map2alm
    qquantity = np.reshape(quantity, (int(shape[0] ** 2), shape[-1]))

    # Get all the alms
    real_alm = np.apply_along_axis(
        spherical_harmonics_projection, 1, qquantity, l_max
    )

    # Reshape to get the same shape as before
    return np.reshape(real_alm, (shape[0], shape[0], real_alm.shape[-1])).T


def complex_to_real_conversion(spherical_harmonics):
    """
    Converts the complex spherical harmonics (or the coefficients) to real
    spherical harmonics (or the coefficients).

    Parameters:
    -----------
    spherical_harmonics : Array
        2D (or 1D) array of complex spherical harmonics coefficients.
        If 2D, the shape is (lm, pp), where lm runs over l,m (with m > 0), and
        pp is the number of theta and phi values. If 1D, the shape is (lm,).

    Returns:
    --------
    all_spherical_harmonics : Array
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
    real_spherical_harmonics : Array
        1D array of real spherical harmonics coefficients.
        The shape is (lm,), where lm runs over l,m (with -l <= m <= l).
    l_max : int
        Maximum ell value.

    Returns:
    --------
    complex_spherical_harmonics : Array
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
    theta : Array
        Array of polar angles (co-latitudes).
    phi : Array
        Array of azimuthal angles (longitudes).

    Returns:
    --------
    all_spherical_harmonics : Array
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


def get_map_from_real_clms(clms_real, Nside, l_max=None):
    """
    Get the HEALPix map from the real spherical harmonic coefficients.

    Parameters:
    -----------
    clms_real : Array
        Real spherical harmonic coefficients with shape as described above a
        vector order according to l and m
    Nside : int
        HEALPix Nside parameter.
    l_max : int, optional
        Maximum multipole moment to consider. If None, defaults to the maximum
        value based on the input coefficients.

    Returns:
    --------
    my_map : Array
        HEALPix map with shape (Npix,).
    """

    # If not provided, get the maximum ell value from the input coefficients
    l_max = get_l_max_real(clms_real) if l_max is None else l_max

    # Get the complex spherical harmonics coefficients
    clms_complex = real_to_complex_conversion(clms_real)

    # Convert the complex coefficients to a map using healpy
    my_map = hp.alm2map(clms_complex, Nside, lmax=l_max)

    # return the map
    return my_map


def get_CL_from_real_clm(clm_real):
    """
    Compute the angular power spectrum from the spherical harmonics
    coefficients.

    Parameters:
    -----------
    clm_real : Array
        Array of spherical harmonics coefficients, if dimension > 1 the first
        axis must run over the coefficients.

    Returns:
    --------
    CL : Array
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


def get_dCL_from_real_clm(clm_real, dclm_real):
    """
    Compute the uncertainty on the angular power spectrum from the spherical
    harmonics coefficients using linear error propagation

    Parameters:
    -----------
    clm_real : Array
        Array of spherical harmonics coefficients, if dimension > 1 the first
        axis must run over the coefficients.
    dclm_real : Array
        Array of uncertainties on the spherical harmonics coefficients, must
        have the same shape as clm_real.

    Returns:
    --------
    dCL : Array
        Array of uncertainties on the angular power spectrum.

    """

    # Get the shape of the input coefficients
    clm_shape = clm_real.shape

    # Get the maximum ell value
    l_max = get_l_max_real(clm_real)

    # Compute the angular power spectrum
    dCL = np.zeros(tuple([l_max + 1] + list(clm_shape[1:])))

    # A counter for the coefficients used up to that value of ell
    i = 0

    for ell in range(0, l_max + 1):
        # Average the square of the coefficients for all ms at fixed ell
        dCL[ell] = 2 * np.mean(
            np.abs(
                clm_real[i : i + 2 * ell + 1] * dclm_real[i : i + 2 * ell + 1]
            ),
            axis=0,
        )

        # Update the counter
        i += 2 * ell + 1

    return dCL


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
    means : Array
        Array of means for the spherical harmonics coefficients.
    cov : Array
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
    Cl_limits : Array
        Array of upper limits on the angular power spectrum from the covariance
    Cl_limits_prior : Array
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
