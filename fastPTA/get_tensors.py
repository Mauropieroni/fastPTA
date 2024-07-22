# Global
import tqdm
import numpy as np
import healpy as hp
import pandas as pd

import jax
import jax.numpy as jnp

from scipy.special import legendre
from scipy.special import sph_harm
from scipy.integrate import simpson

# Local
import fastPTA.utils as ut
from fastPTA.generate_new_pulsar_configuration import generate_pulsars_catalog


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])


# Just some constants
log_A_curn_default = -13.94
log_gamma_curn_default = 2.71
integration_points = 10000


@jax.jit
def unit_vector(theta, phi):
    """
    Compute the unit vector in 3D Cartesian coordinates given spherical
    coordinates. Theta is the polar angle (co-latitude) and phi is the
    azimuthal angle (longitude). The input angles theta and phi should be given
    in radians.

    Parameters:
    -----------
    theta : numpy.ndarray or jax.numpy.ndarray
        Array of angles in radians representing the polar angle (co-latitude).
    phi : numpy.ndarray or jax.numpy.ndarray
        Array of angles in radians representing the azimuthal angle (longitude).

    Returns:
    --------
    unit_vec : numpy.ndarray or jax.numpy.ndarray
        Array of unit vectors in 3D Cartesian coordinates corresponding to
        the given spherical coordinates.

    """

    # Compute the x component of the unit vector
    x_term = jnp.sin(theta) * jnp.cos(phi)

    # Compute the y component of the unit vector
    y_term = jnp.sin(theta) * jnp.sin(phi)

    # Compute the z component of the unit vector
    z_term = jnp.cos(theta)

    # Assemble the unit vector and return it with the right shape
    return jnp.array([x_term, y_term, z_term]).T


@jax.jit
def dot_product(theta_1, phi_1, theta_2, phi_2):
    """
    Compute the dot product of two unit vectors given their spherical
    coordinates. Theta is the polar angle (co-latitude) and phi is the
    azimuthal angle (longitude). The input angles theta and phi should be given
    in radians.

    Parameters:
    -----------
    theta_1 : numpy.ndarray or jax.numpy.ndarray
        Array of angles in radians representing the polar angle (co-latitude)
        of the first vector.
    phi_1 : numpy.ndarray or jax.numpy.ndarray
        Array of angles in radians representing the azimuthal angle (longitude)
        of the first vector.
    theta_2 : numpy.ndarray or jax.numpy.ndarray
        Array of angles in radians representing the polar angle (co-latitude)
        of the second vector.
    phi_2 : numpy.ndarray or jax.numpy.ndarray
        Array of angles in radians representing the azimuthal angle (longitude)
        of the second vector.

    Returns:
    --------
    dot_product : numpy.ndarray or jax.numpy.ndarray
        Array of dot products computed for the given unit vectors.

    """

    # Sum of the product of x and y components
    term1 = jnp.sin(theta_1) * jnp.sin(theta_2) * jnp.cos(phi_1 - phi_2)

    # The product of the z components
    term2 = jnp.cos(theta_1) * jnp.cos(theta_2)

    return term1 + term2


@jax.jit
def HD_correlations(zeta_IJ):
    """
    Compute the Hellings and Downs correlations for two line of sights with
    angular separations zeta_IJ (in radiants).

    Parameters:
    -----------
    zeta_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of angular separations zeta_IJ.

    Returns:
    --------
    HD correlations : numpy.ndarray or jax.numpy.ndarray
        Array of correlations computed for the angular separations zeta_IJ.

    """

    # Compute the difference between 1 and p_I dot p_j and divide by 2
    diff_IJ = 0.5 * (1.0 - zeta_IJ)

    # This function does not include the Kronecker-Delta term for pulsar
    # self-correlations
    return jnp.where(
        diff_IJ > 0.0, 0.5 + 1.5 * diff_IJ * (jnp.log(diff_IJ) - 1 / 6), 0.5
    )


# Some default values to compute the HD curve
x = jnp.linspace(-1.0, 1.0, integration_points)
HD_value = HD_correlations(x)


@jax.jit
def get_WN(WN_par, dt):
    """
    Compute the white noise amplitude for a catalog of pulsars given the
    the white noise amplitudes and sampling rates (see Eq. 5 of 2404.02864).
    The time step dt should be provided in seconds.

    Parameters:
    -----------
    WN_par : numpy.ndarray or jax.numpy.ndarray
        White noise parameters for the pulsar.
    dt : numpy.ndarray or jax.numpy.ndarray
        Time steps for the pulsar.

    Returns:
    --------
    WN_amplitude : numpy.ndarray or jax.numpy.ndarray
        Array of white noise amplitudes.

    """

    return 1e-100 + jnp.array(1e-12 * 2 * WN_par**2 * dt)


@jax.jit
def get_pl_colored_noise(frequencies, log10_ampl, gamma):
    """
    Compute power-law colored noise for given frequencies and parameters.

    Parameters:
    -----------
    frequencies : numpy.ndarray or jax.numpy.ndarray
        Array of frequencies (in Hz) at which to compute the colored noise.
    log10_ampl : numpy.ndarray or jax.numpy.ndarray
        Array of base-10 logarithm of amplitudes.
    gamma : numpy.ndarray or jax.numpy.ndarray
        Array of power-law indices.

    Returns:
    --------
    colored_noise : numpy.ndarray or jax.numpy.ndarray
        Array of power-law colored noise computed for the given frequencies.

    """

    amplitude_prefactor = (10**log10_ampl) ** 2 / 12.0 / jnp.pi**2 / ut.f_yr**3
    frequency_dependent_term = (ut.f_yr / frequencies)[None, :] ** gamma[
        :, None
    ]

    return amplitude_prefactor[:, None] * frequency_dependent_term


@jax.jit
def get_noise_omega(frequencies, noise):
    """
    Takes pulsar noises and convert them in Omega units.

    Parameters:
    -----------
    frequencies : numpy.ndarray or jax.numpy.ndarray
        Array of frequencies.
    noise : numpy.ndarray or jax.numpy.ndarray
        Array representing the noise.

    Returns:
    --------
    numpy.ndarray
        Noise converted to Omega units.

    """

    # Factor to convert to strain
    convert_to_strain_factor = 12 * jnp.pi**2 * frequencies**2

    # Factor to convert to omega units
    convert_to_omega_factor = (
        ut.strain_to_Omega(frequencies) * convert_to_strain_factor
    )

    # Convert the noise to omega units
    return convert_to_omega_factor[:, None, None] * (
        (noise.T)[..., None] * jnp.eye(len(noise))[None, ...]
    )


@jax.jit
def get_pulsar_noises(
    frequencies,
    WN_par,
    log10_A_red,
    gamma_red,
    log10_A_dm,
    gamma_dm,
    log10_A_sv,
    gamma_sv,
    dt,
):
    """
    Compute noise components for given parameters and frequencies.

    The components included are (see Eq. 4 of 2404.02864):

    - White noise (WN) with parameter WN_par.
    - Red noise (RN) with (log10)amplitudes log10_A_red and power-law indices
    gamma_red.
    - Dispersion measure noise (DM) with amplitudes log10_A_dm and power-law
    indices gamma_dm.
    - Scattering variation noise (SV) with amplitudes log10_A_sv and power-law
    indices gamma_sv.

    Parameters:
    -----------
    frequencies : numpy.ndarray or jax.numpy.ndarray
        Array of frequencies (in Hz).
    WN_par : float
        Array of white noise parameter.
    log10_A_red : numpy.ndarray or jax.numpy.ndarray
        Array of base-10 logarithm of amplitudes for red noise.
    gamma_red : numpy.ndarray or jax.numpy.ndarray
        Array of power-law indices for red noise.
    log10_A_dm : numpy.ndarray or jax.numpy.ndarray
        Array of base-10 logarithm of amplitudes for dispersion measure noise.
    gamma_dm : numpy.ndarray or jax.numpy.ndarray
        Array of power-law indices for dispersion measure noise.
    log10_A_sv : numpy.ndarray or jax.numpy.ndarray
        Array of base-10 logarithm of amplitudes for scattering variation noise.
    gamma_sv : numpy.ndarray or jax.numpy.ndarray
        Array of power-law indices for scattering variation noise.
    dt : float
        Array of time steps in seconds.

    Returns:
    --------
    noise : numpy.ndarray or jax.numpy.ndarray
        Array of noise components computed for the given parameters and
        frequencies.

    """

    # white noise coverted from microsecond to second
    WN = get_WN(WN_par, dt)

    # red noise powerlaw for A expressed in strain amplitude
    RN = get_pl_colored_noise(frequencies, log10_A_red, gamma_red)

    # dispersion measure noise powerlaw for A
    # expressed in strain amplitude (evaluated at 1.4 GHz channel)
    DM = get_pl_colored_noise(frequencies, log10_A_dm, gamma_dm)

    # scattering variation noise powerlaw for A
    # expressed in strain amplitude (evaluated at 1.4 GHz channel)
    SV = get_pl_colored_noise(frequencies, log10_A_sv, gamma_sv)

    return WN[:, None] + RN + DM + SV


@jax.jit
def transmission_function(frequencies, T_obs):
    """
    Compute the transmission function (see Eq. 3 of 2404.02864), which
    represents the attenuation of signals, for some frequencies given the
    observation time.

    Parameters:
    -----------
    frequencies : numpy.ndarray or jax.numpy.ndarray
        Array of frequencies (in Hz).
    T_obs : float
        Observation time (in seconds).

    Returns:
    --------
    transmission : numpy.ndarray or jax.numpy.ndarray
        Array of transmission values computed for the given frequencies and
        observation time.

    """

    return 1 / (1 + 1 / (frequencies * T_obs) ** 6)


@jax.jit
def get_time_tensor(frequencies, pta_span_yrs, Tspan_yr):
    """
    Computes the time tensor (i.e., the part of the response depending on the
    observation times) for given frequencies and observation times.

    Parameters:
    -----------
    frequencies : numpy.ndarray or jax.numpy.ndarray
        Array of frequencies.
    pta_span_yrs : float
        Average span of the PTA data in years.
    Tspan_yr : float
        Time span for individual pulsars in years.

    Returns:
    --------
    time_tensor : numpy.ndarray or jax.numpy.ndarray
        Time tensor computed for the given frequencies, PTA span, and
        individual pulsar spans.

    """

    # Do a mesh of the observation times
    time_1, time_2 = jnp.meshgrid(Tspan_yr, Tspan_yr)
    # pick the minimium time in each pair
    time_IJ = jnp.min(jnp.array([time_1, time_2]), axis=0)

    # Compute the transmission function for all times and frequencies
    transmission = transmission_function(
        frequencies[:, None], (Tspan_yr * ut.yr)[None, :]
    )

    # Build the tensor product of the two transimission function
    transmission_tensor = transmission[:, :, None] * transmission[:, None, :]

    # Return the tensor product weighted by the total observation time
    return jnp.sqrt((time_IJ / pta_span_yrs)[None, ...] * transmission_tensor)


@jax.jit
def gamma_pulsar_pair_analytical(
    theta_1, phi_1, theta_2, phi_2, theta_k, phi_k
):
    """
    Compute the analytical expression for the gamma function (see Eq. 13 of
    2407.xxxxx).

    Parameters:
    -----------
    theta_1 : numpy.ndarray or jax.numpy.ndarray
        Array of polar angles (co-latitudes) for the first pulsar.
    phi_1 : numpy.ndarray or jax.numpy.ndarray
        Array of azimuthal angles (longitudes) for the first pulsar.
    theta_2 : numpy.ndarray or jax.numpy.ndarray
        Array of polar angles (co-latitudes) for the second pulsar.
    phi_2 : numpy.ndarray or jax.numpy.ndarray
        Array of azimuthal angles (longitudes) for the second pulsar.
    theta_k : numpy.ndarray or jax.numpy.ndarray
        Array of polar angles (co-latitudes) for the pixel vectors.
    phi_k : numpy.ndarray or jax.numpy.ndarray
        Array of azimuthal angles (longitudes) for the pixel vectors.

    Returns:
    --------
    gamma : numpy.ndarray or jax.numpy.ndarray
        Array of gamma values computed for the given pulsar pairs and pixel
        vectors.

    """

    # Compute p_dot_k
    p_dot_k = dot_product(theta_k, phi_k, theta_1, phi_1)

    # Compute q_dot_k
    q_dot_k = dot_product(theta_k, phi_k, theta_2, phi_2)

    # Compute p_dot_q
    p_dot_q = dot_product(theta_1, phi_1, theta_2, phi_2)

    # This is the second term
    second_term = -(1.0 - p_dot_k) * (1.0 - q_dot_k)

    # This is the numerator of the first term
    numerator = 2.0 * (p_dot_q - p_dot_k * q_dot_k) ** 2.0

    # This is the denominator of the first term
    denominator = (1.0 + p_dot_k) * (1.0 + q_dot_k)

    # Where the denominator is non zero, just numerator / denominator,
    # where it's zero a bit more care is needed, if pI != pJ is zero
    first_term = jnp.where(denominator != 0.0, numerator / denominator, 0.0)

    conditions = (
        (denominator == 0.0)
        & (phi_1 - phi_2 == 0.0)
        & (theta_1 - theta_2 == 0.0)
    )

    # Correct first term where the denominator is zero and pI = pJ
    first_term = jnp.where(
        conditions, -2.0 * second_term, first_term  # type: ignore
    )

    # Sum all the terms up
    return first_term + second_term


@jax.jit
def gamma_analytical(theta, phi, theta_k, phi_k):
    """
    Compute the analytical expression for the gamma function (see Eq. 13 of
    2407.xxxxx).

    Parameters:
    -----------
    theta : numpy.ndarray or jax.numpy.ndarray
        Array of polar angles (co-latitudes) for the pulsars.
    phi : numpy.ndarray or jax.numpy.ndarray
        Array of azimuthal angles (longitudes) for the pulsars.
    theta_k : numpy.ndarray or jax.numpy.ndarray
        Array of polar angles (co-latitudes) for the pixel vectors.
    phi_k : numpy.ndarray or jax.numpy.ndarray
        Array of azimuthal angles (longitudes) for the pixel vectors.

    Returns:
    --------
    gamma : numpy.ndarray or jax.numpy.ndarray
        Array of gamma values computed for all pulsar pairs and for all pixels
        The shape will be ulsar index, pulsar index, pixel index

    """

    return gamma_pulsar_pair_analytical(
        theta[:, None, None],
        phi[:, None, None],
        theta[None, :, None],
        phi[None, :, None],
        theta_k[None, None, :],
        phi_k[None, None, :],
    )


@jax.jit
def gamma(p_I, hat_k):
    """
    Compute the gamma function (see Eq. 13 of  2407.xxxxx).

    Parameters:
    -----------
    p_I : numpy.ndarray or jax.numpy.ndarray
        2D Array of unit vectors representing the pulsar directions.
        Assumed to have shape (N, 3), N is the number of pulsars.

    hat_k : numpy.ndarray or jax.numpy.ndarray
        Array of unit vectors representing the pixel directions.
        Assumed to have shape (P, 3), P is the number of pixels.

    Returns:
    --------
    gamma : numpy.ndarray or jax.numpy.ndarray
        3D array of gamma values computed for all pulsar pairs and pixels
        The shape will be (N, N, P) where  N is the number of pulsars and P is
        the number of pixels.

    """

    # This is the dot product of the unit vectors pointing towards the pulsars
    pIpJ = jnp.einsum("iv,jv->ij", p_I, p_I)

    # This is the dot product of p_I and hat_k
    pIdotk = jnp.einsum("iv,jv->ij", p_I, hat_k)

    # Create the sum and difference vectors to use later
    sum = 1 + pIdotk
    diff = 1 - pIdotk

    # Compute the second term
    second_term = -diff[:, None, :] * diff[None, ...]

    # This is the pIdotk * pJdotk term in the numerator of the first term
    pk_qk = pIdotk[:, None, :] * pIdotk[None, ...]

    # Get the numerator of the first term
    numerator = 2 * (pIpJ[..., None] - pk_qk) ** 2

    # Get the denominator of the first term
    denominator = sum[:, None, :] * sum[None, ...]

    # Where the denominator is non zero, just numerator / denominator
    first_term = jnp.where(denominator != 0.0, numerator / denominator, 0.0)

    # Correct first term where the denominator is zero and pI = pJ
    first_term = jnp.where(
        ((denominator == 0.0) & (jnp.bool_(jnp.floor(pk_qk)))),
        -2.0 * second_term,
        first_term,
    )

    # Sum the two terms and return
    return first_term + second_term


def get_sort_indexes(l_max):
    """
    Given the maximum ell value, this function returns the indexes to sort the
    spherical harmonics coefficients as returned by map2alm of healpy (see
    https://healpy.readthedocs.io/en/latest/). The coefficients are sorted in
    the following way:

    - First all the negative m values for a given ell are sorted in decreasing
        order.
    - Then the m=0 values are sorted.
    - Finally all the positive m values for a given ell are sorted in increasing
        order.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.

    Returns:
    --------
    m_grid : numpy.ndarray
        Array of m values.

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

    # Select only the m values that are allowed for a given ell
    l_grid = l_flat[np.abs(m_flat) <= l_flat]
    m_grid = m_flat[np.abs(m_flat) <= l_flat]

    # Create a vector with all the m<0 and then all the m>=0
    mm = np.append(-np.flip(m_grid[m_grid > 0]), m_grid)

    # Create a vector with all the ls corresponding to mm
    ll = np.append(np.flip(l_grid[m_grid > 0]), l_grid)

    # Return the sorted indexes
    return m_grid, np.lexsort((mm, ll))


def spherial_harmonics_projection(quantity, l_max):
    """
    Compute the spherical harmonics projection of a given quantity. Quantity
    should be an array in pixel space, and compatible with healpy (see
    https://healpy.readthedocs.io/en/latest/). The spherical harmonics
    coefficients are sorted as described in the get_sort_indexes function.

    Parameters:
    -----------
    quantity : numpy.ndarray
        Array of quantities to project on spherical harmonics.

    l_max : int
        Maximum ell value.

    Returns:
    --------
    real_alm : numpy.ndarray
        Array of real spherical harmonics coefficients, with len (l_max + 1)**2

    """

    # Get the complex alm coefficients.
    # These are sorted with the m values and are only for m>=0
    alm = hp.map2alm(quantity, lmax=l_max)

    # Create arrays the m_values and the indexes to sort
    m_grid, sort_indexes = get_sort_indexes(l_max)

    # Select only the m values that are allowed for a given ell
    m_p = m_grid[m_grid > 0]

    # Build the real alm selecting imaginary and real part
    negative_m = np.flip(np.sqrt(2) * (-1) ** m_p * alm[m_grid > 0].imag)
    positive_m = np.sqrt(2) * (-1) ** m_p * alm[m_grid > 0].real

    # Concatenate the negative, zero and positive m values
    real_alm = np.concatenate((negative_m, alm[m_grid == 0].real, positive_m))

    # Sort with the indexes
    return real_alm[sort_indexes]


def spherial_harmonics_projection_pulsars(quantity, l_max):
    """
    Compute the spherical harmonics projection of a the correlation matrix
    given quantity.

    TBD: This function should use the fact that quantity is symmetric in the
    pulsar pulsar indexes to reduce computations and increase efficiency.

    Parameters:
    -----------
    quantity : numpy.ndarray
        3D array to be projected on spherical harmonics. Should have shape
        (N, N, P), where N is the number of pulsars and P is the number of
        pixels, which should be compatible with healpy
        (see https://healpy.readthedocs.io/en/latest/).

    l_max : int
        Maximum ell value.

    Returns:
    --------
    real_alm : numpy.ndarray
        3D array of real spherical harmonics coefficients.
        It has shape ( (l_max + 1)**2, N, N), where N is the number of pulsars.

    """

    # Get the shape of the quantity to project on spherical harmonics
    shape = list(quantity.shape)

    # Reshape quantity so that it can be passed to hp.map2alm
    qquantity = np.reshape(quantity, (int(shape[0] ** 2), shape[-1]))

    # Get all the alms
    real_alm = np.apply_along_axis(
        spherial_harmonics_projection, 1, qquantity, l_max
    )

    # Reshape to get the same shape as before
    return np.reshape(real_alm, (shape[0], shape[0], real_alm.shape[-1])).T


def get_correlations_lm_IJ(p_I, l_max, nside):
    """
    Compute the correlations in spherical harmonics basis for a given pulsar
    catalog. The correlations are computed up to a maximum ell value l_max and
    for a given nside.

    Parameters:
    -----------
    p_I : numpy.ndarray
        2D array containing signal data, assumed to have shape (N, M, M)
        Array of unit vectors representing the pulsar directions.

    l_max : int
        Maximum ell value.

    nside : int
        Resolution parameter for the HEALPix grid.

    Returns:
    --------
    correlations_lm : numpy.ndarray
        3D array of correlations computed in spherical harmonics basis.
        It has shape ( (l_max + 1)**2, N, N), where N is the number of pulsars.

    """

    # Given nside get a pixelization of the sky
    npix = hp.nside2npix(nside)
    theta_k, phi_k = hp.pix2ang(nside, jnp.arange(npix))
    theta_k = jnp.array(theta_k)
    phi_k = jnp.array(phi_k)

    # Get the k vector (i.e., the sky direction) for all the pixels
    hat_k = unit_vector(theta_k, phi_k)

    # Compute gamma in all the pixels
    gamma_pq = 3 / 8 * gamma(p_I, hat_k)

    # Project gamma onto spherical harmonics
    correlations_lm = spherial_harmonics_projection_pulsars(gamma_pq, l_max)

    # Multiply by 1 + delta_{IJ} and return
    return correlations_lm * (1 + np.eye(len(p_I)))[None, ...]


def get_response_IJ_lm(p_I, time_tensor_IJ, l_max, nside):
    """
    Compute the response tensor for given angular separations and time tensors.

    Parameters:
    -----------
    p_I : numpy.ndarray
        2D array containing signal data, assumed to have shape (N, M, M)
        Array of unit vectors representing the pulsar directions.
    """

    # Compute the correlations on lm basis
    correlations_lm_IJ = get_correlations_lm_IJ(p_I, l_max, nside)

    # combine the Hellings and Downs part and the time part
    return time_tensor_IJ[None, ...] * correlations_lm_IJ[:, None, ...]


@jax.jit
def get_chi_tensor_IJ(zeta_IJ):
    """
    Computes the chi_IJ tensor as expressed in eq. 15 of
    https://arxiv.org/pdf/2404.02864.pdf for given angular separations.

    Parameters:
    -----------
    zeta_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of angular separations for all pulsar pairs.

    Returns:
    --------
    chi_IJ : numpy.ndarray or jax.numpy.ndarray
        chi_IJ tensor for all the pulsars pairs.

    """

    # Compute HD and add the self correlation term
    return HD_correlations(zeta_IJ) + 0.5 * jnp.eye(len(zeta_IJ))


@jax.jit
def get_response_IJ(zeta_IJ, time_tensor_IJ):
    """
    Compute the response tensor for given angular separations and time tensors.

    Parameters:
    -----------
    zeta_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of angular separations for all pulsar pairs.
    time_tensor_IJ : numpy.ndarray or jax.numpy.ndarray
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies.

    Returns:
    --------
    response_IJ : numpy.ndarray or jax.numpy.ndarray
        Response tensor for all the pulsars pairs.

    """

    # Compute the chi_IJ tensor for the angular separations
    chi_tensor_IJ = get_chi_tensor_IJ(zeta_IJ)

    # combine the Hellings and Downs part and the time part
    return time_tensor_IJ * chi_tensor_IJ[None, ...]


def get_HD_Legendre_coefficients(order):
    """
    Compute Legendre coefficients for Hellings and Downs correlations for
    polynomials up to some order

    Parameters:
    -----------
    order : int
        Maximum order of Legendre coefficients to compute.

    Returns:
    --------
    coefficients : numpy.ndarray or jax.numpy.ndarray
        Array of Legendre coefficients computed up to the given order.

    """

    # Some l dependent normalization factor
    l_coeffs = (2 * jnp.arange(order + 1) + 1) / 2

    return jnp.array(
        [
            # Project onto Legendre polynomials
            simpson(legendre(i)(x) * HD_value, x=x) * l_coeffs[i]
            for i in range(order + 1)
        ]
    )


def get_polynomials_IJ(zeta_IJ, order):
    """
    Compute Legendre polynomials for given angular separations and order.

    Parameters:
    -----------
    zeta_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of angular separations between pairs of pulsars.
    order : int
        Maximum order of Legendre polynomials.

    Returns:
    --------
    polynomials_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of Legendre polynomials computed for the given angular separations
        and order.

    """

    polynomials_IJ = []

    for i in range(order + 1):
        polynomials_IJ.append(legendre(i)(zeta_IJ))

    return jnp.array(polynomials_IJ)


@jax.jit
def Legendre_projection(time_tensor_IJ, polynomials_IJ):
    """
    Projects the pulsar angular information onto Legendre polynomials

    Parameters:
    -----------
    time_tensor_IJ : numpy.ndarray or jax.numpy.ndarray
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies. The shape is (
        len(frequencies), len(pulsars), len(pulsars)
    polynomials_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of Legendre polynomials. The shape is (order + 1, len(pulsars),
        len(pulsars)

    Returns:
    --------
    projection : numpy.ndarray or jax.numpy.ndarray
        Legendre projection

    """

    return time_tensor_IJ[None, ...] * polynomials_IJ[:, None, ...]


def HD_projection_Legendre(zeta_IJ, time_tensor_IJ, order):
    """
    Projects Hellings and Downs correlations onto Legendre polynomials.

    Parameters:
    -----------
    zeta_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of angular separations between pairs of pulsars.
    time_tensor_IJ : numpy.ndarray or jax.numpy.ndarray
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies.
    order : int
        Maximum order of Legendre polynomials.

    Returns:
    --------
    Tuple containing:
    - HD_functions : numpy.ndarray or jax.numpy.ndarray
        Hellings and Downs correlations projected onto Legendre polynomials.
    - HD_coefficients : numpy.ndarray or jax.numpy.ndarray
        Legendre coefficients for Hellings and Downs correlations up to the
        given order.

    """

    # Gets the Legendre coefficients for HD
    HD_coefficients = get_HD_Legendre_coefficients(order)

    # Gets the values of the HD polynomials for all angular separations
    polynomials_IJ = get_polynomials_IJ(zeta_IJ, order)

    # Projects the pulsar catalog onto Legendre polynomials
    HD_functions = Legendre_projection(time_tensor_IJ, polynomials_IJ)

    return HD_functions, HD_coefficients


@jax.jit
def binned_projection(zeta_IJ, time_tensor_IJ, masks):
    """
    Compute binned projection of the Hellings and Downs correlations.

    Parameters:
    -----------
    zeta_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of angular separations between pairs of pulsars.
    time_tensor_IJ : numpy.ndarray or jax.numpy.ndarray
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies.
    masks : numpy.ndarray or jax.numpy.ndarray
        Array of masks representing binned intervals for Hellings and Downs
        correlations.

    Returns:
    --------
    binned_projection :  numpy.ndarray or jax.numpy.ndarray
        Binned projection of the Hellings and Downs correlations.

    """

    return (
        time_tensor_IJ - jnp.eye(len(zeta_IJ))[None, ...] * time_tensor_IJ
    ) * masks[:, None, ...]


def HD_projection_binned(zeta_IJ, time_tensor_IJ, order):
    """
    Projects Hellings and Downs correlations onto binned intervals.
    NB!! For consistency with the Legendre version it uses order +1 bins!

    Parameters:
    -----------
    zeta_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of angular separations between pairs of pulsars.
    time_tensor_IJ : numpy.ndarray or jax.numpy.ndarray
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies.
    order : int
        Number of bins used in the analysis.

    Returns:
    --------
    Tuple containing:
    - HD_functions : numpy.ndarray or jax.numpy.ndarray
        Binned Hellings and Downs correlations.
    - HD_coefficients : numpy.ndarray or jax.numpy.ndarray
        Coefficients of the binned Hellings and Downs correlations

    """

    # Ensure Hellings and Downs correlations values are within bounds
    zeta_IJ = jnp.where(zeta_IJ > 1, 1, zeta_IJ)
    zeta_IJ = jnp.where(zeta_IJ < -1, -1, zeta_IJ)  # type: ignore
    xi_vals = jnp.arccos(zeta_IJ)

    # Compute bin edges for binning the Hellings and Downs correlations values
    bin_edges = jnp.linspace(0 + 1e-2, jnp.pi + 1e-2, order + 2)

    # Compute mean values for each bin
    mean_x = 0.5 * (jnp.cos(bin_edges[:-1]) + np.cos(bin_edges[1:]))

    # Compute Hellings and Downs correlations coefficients for binned intervals
    HD_coefficients = HD_correlations(mean_x)

    # Create masks for each bin
    masks = np.zeros(shape=(order + 1, len(zeta_IJ), len(zeta_IJ)))

    for i in range(len(masks)):
        masks[i][(xi_vals > bin_edges[i]) & (xi_vals < bin_edges[i + 1])] = 1.0

    # And converts to jax array
    masks = jnp.array(masks)

    # Project Hellings and Downs correlations functions onto binned intervals
    HD_functions_IJ = binned_projection(zeta_IJ, time_tensor_IJ, masks)

    return HD_functions_IJ, HD_coefficients


def get_tensors(
    frequencies,
    path_to_pulsar_catalog=ut.path_to_default_pulsar_catalog,
    pta_span_yrs=10.33,
    add_curn=False,
    order=0,
    method="legendre",
    anisotropies=False,
    l_max=0,
    nside=16,
    regenerate_catalog=False,
    **generate_catalog_kwargs
):
    """
    Generate all tensors (noise, response and Hellings and Downs) needed for
    gravitational wave data analysis. The noise tensors is returned in omega
    units. The Hellings and Downs correlations are projected onto Legendre
    polynomials or binned intervals based on the chosen method.

    Parameters:
    -----------
    frequencies : numpy.ndarray or jax.numpy.ndarray
        Array of frequencies.
    path_to_pulsar_catalog : str, optional
        Path to the pulsars data file.
        Default is path_to_default_pulsar_catalog.
    pta_span_yrs : float, optional
        Average span of the PTA data in years.
        Default is 10.33 years.
    add_curn : bool, optional
        Whether to add common (spatially) uncorrelated red noise (CURN).
        Default is False.
    order : int, optional
        Maximum order of Legendre polynomials/ number of bins for the Hellings
        and Downs correlations projection.
        Default is 0.
    method : str, optional
        Method for Hellings and Downs correlations projection.
        Options are "legendre" or "binned".
        Default is "legendre".
    regenerate_catalog : bool, optional
        Whether to regenerate the pulsars data file.
        Default is False.
    **generate_catalog_kwargs : dict, optional
        Additional keyword arguments for the generate_pulsars_catalog function.

    Returns:
    --------
    Tuple containing:
    - strain_omega : numpy.ndarray or jax.numpy.ndarray
        Noise in omega units.
    - response_IJ : numpy.ndarray or jax.numpy.ndarray
        Response tensor.
    - HD_functions_IJ : numpy.ndarray or jax.numpy.ndarray
        Hellings and Downs correlations functions projected onto Legendre
        polynomials or binned intervals.
    - HD_coefficients : numpy.ndarray or jax.numpy.ndarray
        Legendre coefficients for Hellings and Downs correlations values up to
        the given order.

    """

    # Load or regenerate pulsar catalog
    try:
        if regenerate_catalog:
            raise FileNotFoundError
        pulsars_DF = pd.read_csv(path_to_pulsar_catalog, sep=" ")

    except FileNotFoundError:
        generate_catalog_kwargs["outname"] = path_to_pulsar_catalog
        pulsars_DF = generate_pulsars_catalog(**generate_catalog_kwargs)

    # unpack all parameters
    WN_par = jnp.array(pulsars_DF["wn"].values)
    log10_A_red = jnp.array(pulsars_DF["log10_A_red"].values)
    gamma_red = jnp.array(pulsars_DF["g_red"].values)
    log10_A_dm = jnp.array(pulsars_DF["log10_A_dm"].values)
    gamma_dm = jnp.array(pulsars_DF["g_dm"].values)
    log10_A_sv = jnp.array(pulsars_DF["log10_A_sv"].values)
    gamma_sv = jnp.array(pulsars_DF["g_sv"].values)
    Tspan_yr = jnp.array(pulsars_DF["Tspan"].values)
    dt = jnp.array(pulsars_DF["dt"].values)
    theta = jnp.array(pulsars_DF["theta"].values)
    phi = jnp.array(pulsars_DF["phi"].values)
    pi_vec = unit_vector(theta, phi)

    # get noises for all pulsars
    noise = get_pulsar_noises(
        frequencies,
        WN_par,
        log10_A_red,
        gamma_red,
        log10_A_dm,
        gamma_dm,
        log10_A_sv,
        gamma_sv,
        dt,
    )

    # add curn if present
    if add_curn:
        log10_A_curn = jnp.repeat(log_A_curn_default, len(WN_par))
        gamma_curn = jnp.repeat(log_gamma_curn_default, len(WN_par))

        curn = get_pl_colored_noise(frequencies, log10_A_curn, gamma_curn)
        noise += curn

    # convert the noise in strain and then omega units
    strain_omega = get_noise_omega(frequencies, noise)

    # get the time tensor
    time_tensor_IJ = get_time_tensor(frequencies, pta_span_yrs, Tspan_yr)

    # compute angular separations
    zeta_IJ = jnp.einsum("ik, jk->ij", pi_vec, pi_vec)

    if not anisotropies:
        # compute the response
        response_IJ = get_response_IJ(zeta_IJ, time_tensor_IJ)

    else:
        response_IJ = get_response_IJ_lm(pi_vec, time_tensor_IJ, l_max, nside)

    # and if needed the HD part
    if order > 0 and method.lower() == "legendre":
        HD_functions_IJ, HD_coefficients = HD_projection_Legendre(
            zeta_IJ, time_tensor_IJ, order
        )

    elif order > 0 and method.lower() == "binned":
        HD_functions_IJ, HD_coefficients = HD_projection_binned(
            zeta_IJ, time_tensor_IJ, order
        )

    else:
        HD_functions_IJ = jnp.zeros(
            shape=(0, len(frequencies), len(WN_par), len(WN_par))
        )
        HD_coefficients = jnp.zeros(shape=(0,))

    return strain_omega, response_IJ, HD_functions_IJ, HD_coefficients
