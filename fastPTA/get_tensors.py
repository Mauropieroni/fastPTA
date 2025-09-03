# Global
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import simpson
from scipy.special import legendre

# Local
import fastPTA.utils as ut
import fastPTA.pulsar_noises as pn
import fastPTA.data.datastream as gds
from fastPTA.angular_decomposition import spherical_harmonics as spha
from fastPTA.generate_new_pulsar_configuration import generate_pulsars_catalog

# Set the device
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# Just some constants
log_A_curn_default = -13.94
log_gamma_curn_default = 2.71
integration_points = 10000


@jax.jit
def HD_correlations(zeta_IJ):
    """
    Compute the Hellings and Downs correlations for two line of sights with
    angular separations zeta_IJ (in radiants).

    Parameters:
    -----------
    zeta_IJ : Array
        2D array of angular separations zeta_IJ. The angular separations should
        be given in radians, and the array should have shape (N, N), where N
        is the number of pulsars.

    Returns:
    --------
    HD correlations : Array
        2D array of correlations computed for the angular separations zeta_IJ.
        The array has shape (N, N), where N is the number of pulsars.

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
def transmission_function(frequencies, T_obs):
    """
    Compute the transmission function (see Eq. 3 of 2404.02864), which
    represents the attenuation of signals, for some frequencies given the
    observation time.

    Parameters:
    -----------
    frequencies : Array
        Array of frequencies (in Hz).
    T_obs : float
        Observation time (in seconds).

    Returns:
    --------
    transmission : Array
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
    frequencies : Array
        Array of frequencies.
    pta_span_yrs : float
        Average span of the PTA data in years.
    Tspan_yr : float
        Time span for individual pulsars in years.

    Returns:
    --------
    time_tensor : Array
        3D array representing the time tensor computed for the given
        frequencies, PTA span, and individual pulsar spans. It has shape
        (F, N, N), where F is the number of frequencies and N is the number
        of pulsars.

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
    2407.14460).

    Parameters:
    -----------
    theta_1 : Array
        Array of polar angles (co-latitudes) for the first pulsar.
    phi_1 : Array
        Array of azimuthal angles (longitudes) for the first pulsar.
    theta_2 : Array
        Array of polar angles (co-latitudes) for the second pulsar.
    phi_2 : Array
        Array of azimuthal angles (longitudes) for the second pulsar.
    theta_k : Array
        Array of polar angles (co-latitudes) for the pixel vectors.
    phi_k : Array
        Array of azimuthal angles (longitudes) for the pixel vectors.

    Returns:
    --------
    gamma : Array
        Array of gamma values computed for the given pulsar pairs and pixel
        vectors.

    """

    # Compute p_dot_k
    p_dot_k = ut.dot_product(theta_k, phi_k, theta_1, phi_1)

    # Compute q_dot_k
    q_dot_k = ut.dot_product(theta_k, phi_k, theta_2, phi_2)

    # Compute p_dot_q
    p_dot_q = ut.dot_product(theta_1, phi_1, theta_2, phi_2)

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
    2407.14460).

    Parameters:
    -----------
    theta : Array
        Array of polar angles (co-latitudes) for the pulsars.
    phi : Array
        Array of azimuthal angles (longitudes) for the pulsars.
    theta_k : Array
        Array of polar angles (co-latitudes) for the pixel vectors.
    phi_k : Array
        Array of azimuthal angles (longitudes) for the pixel vectors.

    Returns:
    --------
    gamma : Array
        3D array of gamma values computed for all pulsar pairs and for all
        pixels. The shape will be (N, N, pp), where N is the number of pulsars
        and pp is the number of pixels.

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
    Compute the gamma function (see Eq. 13 of  2407.14460).

    Parameters:
    -----------
    p_I : Array
        2D array of unit vectors representing the pulsar directions.
        Assumed to have shape (N, 3), N is the number of pulsars.

    hat_k : Array
        Array of unit vectors representing the pixel directions.
        Assumed to have shape (pp, 3), pp is the number of pixels.

    Returns:
    --------
    gamma : Array
        3D array of gamma values computed for all pulsar pairs and pixels
        The shape will be (N, N, pp) where  N is the number of pulsars and pp is
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


def get_correlations_lm_IJ_spherical_harmonics_basis(p_I, l_max, gamma_pq):
    """
    Compute the correlations in spherical harmonics basis for a given pulsar
    catalog. The correlations are computed up to a maximum ell value l_max and
    for a given nside.

    Parameters:
    -----------
    p_I : Array
        2D array of unit vectors representing the pulsar directions.
        Assumed to have shape (N, 3), N is the number of pulsars.
    l_max : int
        Maximum ell value.
    gamma_pq : Array
        3D array of gamma values computed for all pulsar pairs and pixels.
        The shape should be (N, N, pp), where N is the number of pulsars and
        pp is the number of pixels

    Returns:
    --------
    correlations_lm : numpy.ndarray
        3D array of correlations computed in spherical harmonics basis.
        It has shape (lm, N, N), where N is the number of pulsars and
        lm = (l_max + 1)**2 is the number of spherical harmonics coefficients.

    """

    # Project gamma onto spherical harmonics
    correlations_lm = spha.project_correlation_spherical_harmonics(
        gamma_pq, l_max
    )

    # Multiply by 1 + delta_{IJ} and return
    return correlations_lm * (1 + np.eye(len(p_I)))[None, ...]


def get_correlations_lm_IJ_sqrt_basis(p_I, l_max, theta_k, phi_k, gamma_pq):
    """
    Compute the correlations in sqrt basis for a given pulsar catalog. The
    correlations are computed up to a maximum ell value l_max and for a given
    nside.

    Parameters:
    -----------
    p_I : Array
        2D array of unit vectors representing the pulsar directions.
        Assumed to have shape (N, 3), N is the number of pulsars.
    l_max : int
        Maximum ell value.
    theta_k : Array
        Array of polar angles (co-latitudes) for the pixel vectors.
    phi_k : Array
        Array of azimuthal angles (longitudes) for the pixel vectors.
    gamma_pq : Array
        3D array of gamma values computed for all pulsar pairs and pixels.
        The shape should be (N, N, pp), where N is the number of pulsars and
        pp is the number of pixels

    Returns:
    --------
    correlations_lm : numpy.ndarray
        4D array of correlations computed in sqrt basis.
        It has shape (lm, lm, N, N), where N is the number of pulsars and
        lm = (l_max + 1)**2 is the number of spherical harmonics coefficients.

    """

    # Get the number of pixels
    npix = hp.nside2npix(theta_k)

    # spherical harmonis with shape (lm, pp)
    spherical_harmonics = spha.get_spherical_harmonics(l_max, theta_k, phi_k)

    # Quadratic spherical_harmonics basis with shape (lm, lm, pp)
    quadratic = spherical_harmonics[:, None] * spherical_harmonics[None, :]

    # Project gamma onto the sqrt basis
    correlations_lm = (
        np.einsum("ijp,nmp->ijnm", quadratic, gamma_pq) * (4 * jnp.pi) / npix
    )

    return correlations_lm * (1 + np.eye(len(p_I)))[None, ...]


def get_correlations_lm_IJ(
    p_I, l_max, nside, lm_basis="spherical_harmonics_basis"
):
    """
    Compute the response tensor for given angular separations and time tensors.

    Parameters:
    -----------
    p_I : Array
        2D array of unit vectors representing the pulsar directions.
        Assumed to have shape (N, 3), N is the number of pulsars.
    l_max : int
        Maximum ell value.
    nside : int
        Resolution parameter for the HEALPix grid.
    lm_basis : str
        Basis to compute the correlations.
        Can be either "spherical_harmonics_basis" or "sqrt_basis".

    Returns:
    --------
    correlations_lm_IJ : numpy.ndarray
        3D or 4D array of correlations computed in the given basis.
        If lm_basis is "spherical_harmonics_basis", it has shape
        (lm, N, N), where lm = (l_max + 1)**2 is the number of spherical
        harmonics coefficients and N is the number of pulsars.
        If lm_basis is "sqrt_basis", it has shape (lm, lm, N, N).
    """

    # Given nside get a pixelization of the sky
    npix = hp.nside2npix(nside)
    theta_k, phi_k = hp.pix2ang(nside, jnp.arange(npix))
    theta_k = jnp.array(theta_k)
    phi_k = jnp.array(phi_k)

    # Get the k vector (i.e., the sky direction) for all the pixels
    hat_k = gds.unit_vector(theta_k, phi_k)

    # Compute gamma in all the pixels, the shape is (N, N, pp)
    gamma_pq = 3.0 / 8.0 * gamma(p_I, hat_k)

    # Compute the correlations on lm basis
    if lm_basis.lower() == "spherical_harmonics_basis":
        correlations_lm_IJ = get_correlations_lm_IJ_spherical_harmonics_basis(
            p_I, l_max, gamma_pq
        )

    elif lm_basis.lower() == "sqrt_basis":
        correlations_lm_IJ = get_correlations_lm_IJ_sqrt_basis(
            p_I, l_max, theta_k, phi_k, gamma_pq
        )

    # return the correlations
    return correlations_lm_IJ


def get_response_IJ_lm(
    p_I, time_tensor_IJ, l_max, nside, lm_basis="spherical_harmonics_basis"
):
    """
    Compute the response tensor for given angular separations and time tensors.

    Parameters:
    -----------
    p_I : Array
        2D array of unit vectors representing the pulsar directions.
        Assumed to have shape (N, 3), N is the number of pulsars.
    time_tensor_IJ : Array
        3D array containing the attenuations due to the observation time for
        all the pulsars and for all frequencies. Should have shape (F, N, N),
        where F is the number of frequencies and N is the number of pulsars.
    l_max : int
        Maximum ell value.
    nside : int
        Resolution parameter for the HEALPix grid.
    lm_basis : str
        Basis to compute the correlations.
        Can be either "spherical_harmonics_basis" or "sqrt_basis".

    Returns:
    --------
    response_IJ : Array
        4D array containing response tensor for all the pulsars pairs.
        It has shape (lm, F, N, N), where lm is the number of coefficients for
        the anisotropy decomposition (spherical harmonics or sqrt basis), F is
        the number of frequencies, N is the number of pulsars.

    """

    # Compute the correlations on lm basis
    correlations_lm_IJ = get_correlations_lm_IJ(
        p_I, l_max, nside, lm_basis=lm_basis
    )

    # combine the Hellings and Downs part and the time part
    return time_tensor_IJ[None, ...] * correlations_lm_IJ[:, None, ...]


@jax.jit
def get_chi_tensor_IJ(zeta_IJ):
    """
    Computes the chi_IJ tensor as expressed in eq. 15 of
    https://arxiv.org/pdf/2404.02864.pdf for given angular separations.

    Parameters:
    -----------
    zeta_IJ : Array
        2D array of angular separations zeta_IJ. The angular separations should
        be given in radians, and the array should have shape (N, N), where N
        is the number of pulsars.

    Returns:
    --------
    chi_IJ : Array
        2D array representing the chi_IJ tensor for all the pulsars pairs.
        It has shape (N, N), where N is the number of pulsars.

    """

    # Compute HD and add the self correlation term
    return HD_correlations(zeta_IJ) + 0.5 * jnp.eye(len(zeta_IJ))


@jax.jit
def get_response_IJ(zeta_IJ, time_tensor_IJ):
    """
    Compute the response tensor for given angular separations and time tensors.

    Parameters:
    -----------
    zeta_IJ : Array
        2D array of angular separations zeta_IJ. The angular separations should
        be given in radians, and the array should have shape (N, N), where N
        is the number of pulsars.
    time_tensor_IJ : Array
        3D array representing the time tensor containing the attenuations due
        to the observation time for all the pulsars and for all frequencies.
        It should have shape (F, N, N), where F is the number of frequencies
        and N is the number of pulsars.

    Returns:
    --------
    response_IJ : Array
        3D array containing the response tensor for all the pulsars pairs. It
        has shape (F, N, N), where F is the number of frequencies and N is the
        number of pulsars.

    """

    # Compute the chi_IJ tensor for the angular separations
    chi_tensor_IJ = get_chi_tensor_IJ(zeta_IJ)

    # combine the Hellings and Downs part and the time part
    return time_tensor_IJ * chi_tensor_IJ[None, ...]


def get_HD_Legendre_coefficients(HD_order):
    """
    Compute Legendre coefficients for Hellings and Downs correlations for
    polynomials up to some HD_order

    Parameters:
    -----------
    HD_order : int
        Maximum order of Legendre coefficients to compute.

    Returns:
    --------
    coefficients : Array
        Array of Legendre coefficients computed up to the given HD_order.

    """

    # Some l dependent normalization factor
    l_coeffs = (2 * jnp.arange(HD_order + 1) + 1) / 2

    return jnp.array(
        [
            # Project onto Legendre polynomials
            simpson(legendre(i)(x) * HD_value, x=x) * l_coeffs[i]
            for i in range(HD_order + 1)
        ]
    )


def get_polynomials_IJ(zeta_IJ, HD_order):
    """
    Compute Legendre polynomials for given angular separations and HD_order.

    Parameters:
    -----------
    zeta_IJ : Array
        2D array of angular separations zeta_IJ. The angular separations should
        be given in radians, and the array should have shape (N, N), where N
        is the number of pulsars.
    HD_order : int
        Maximum HD_order of Legendre polynomials.

    Returns:
    --------
    polynomials_IJ : Array
        Array of Legendre polynomials computed for the given angular separations
        and HD_order.

    """

    # Create an array to store the Legendre polynomials
    polynomials_IJ = []

    # Compute the Legendre polynomials for all the angular separations
    for i in range(HD_order + 1):
        polynomials_IJ.append(legendre(i)(zeta_IJ))

    return jnp.array(polynomials_IJ)


@jax.jit
def Legendre_projection(time_tensor_IJ, polynomials_IJ):
    """
    Projects the pulsar angular information onto Legendre polynomials

    Parameters:
    -----------
    time_tensor_IJ : Array
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies. The shape is (F, N, N), where
        F is the number of frequencies and N is the number of pulsars.
    polynomials_IJ : Array
        Array of Legendre polynomials. The shape is (HD_order + 1, N, N), where
        HD_order is the maximum order of Legendre polynomials and N is the
        number of pulsars.

    Returns:
    --------
    projection : Array
        4D array with the Legendre projection of the Hellings and Downs
        correlations. The shape is (HD_order + 1, F, N, N), where HD_order is
        the maximum order of Legendre polynomials, F is the number of
        frequencies, and N is the number of pulsars.

    """

    return time_tensor_IJ[None, ...] * polynomials_IJ[:, None, ...]


def HD_projection_Legendre(zeta_IJ, time_tensor_IJ, HD_order):
    """
    Projects Hellings and Downs correlations onto Legendre polynomials.

    Parameters:
    -----------
    zeta_IJ : Array
        2D array of angular separations zeta_IJ. The angular separations should
        be given in radians, and the array should have shape (N, N), where N
        is the number of pulsars.
    time_tensor_IJ : Array
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies. The shape is (F, N, N), where
        F is the number of frequencies and N is the number of pulsars.
    HD_order : int
        Maximum HD_order of Legendre polynomials.

    Returns:
    --------
    Tuple containing:
    - HD_functions : Array
        4D array with the Legendre projection of the Hellings and Downs
        correlations. The shape is (HD_order + 1, F, N, N), where HD_order is
        the maximum order of Legendre polynomials, F is the number of
        frequencies, and N is the number of pulsars.
    - HD_coefficients : Array
        Legendre coefficients for Hellings and Downs correlations up to the
        given HD_order.

    """

    # Gets the Legendre coefficients for HD
    HD_coefficients = get_HD_Legendre_coefficients(HD_order)

    # Gets the values of the HD polynomials for all angular separations
    polynomials_IJ = get_polynomials_IJ(zeta_IJ, HD_order)

    # Projects the pulsar catalog onto Legendre polynomials
    HD_functions = Legendre_projection(time_tensor_IJ, polynomials_IJ)

    return HD_functions, HD_coefficients


@jax.jit
def binned_projection(zeta_IJ, time_tensor_IJ, masks):
    """
    Compute binned projection of the Hellings and Downs correlations.

    Parameters:
    -----------
    zeta_IJ : Array
        2D array of angular separations zeta_IJ. The angular separations should
        be given in radians, and the array should have shape (N, N), where N
        is the number of pulsars.
    time_tensor_IJ : Array
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies. The shape is (F, N, N), where
        F is the number of frequencies and N is the number of pulsars.
    masks : Array
        Array of masks representing binned intervals for Hellings and Downs
        correlations.

    Returns:
    --------
    binned_projection : Array
        4D array with the binned projection of the Hellings and Downs
        correlations. The shape is (HD_order + 1, F, N, N), where HD_order is
        number of bins, F is the number of frequencies, and N is the number of
        pulsars.

    """

    return (
        time_tensor_IJ - jnp.eye(len(zeta_IJ))[None, ...] * time_tensor_IJ
    ) * masks[:, None, ...]


def HD_projection_binned(zeta_IJ, time_tensor_IJ, HD_order):
    """
    Projects Hellings and Downs correlations onto binned intervals.
    NB!! For consistency with the Legendre version it uses HD_order +1 bins!

    Parameters:
    -----------
    zeta_IJ : Array
        2D array of angular separations zeta_IJ. The angular separations should
        be given in radians, and the array should have shape (N, N), where N
        is the number of pulsars.
    time_tensor_IJ : Array
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies. The shape is (F, N, N), where
        F is the number of frequencies and N is the number of pulsars.
    HD_order : int
        Number of bins used in the analysis.

    Returns:
    --------
    Tuple containing:
    - HD_functions : Array
        4D array with the binned projection of the Hellings and Downs
        correlations. The shape is (HD_order + 1, F, N, N), where HD_order is
        number of bins, F is the number of frequencies, and N is the number of
        pulsars.
    - HD_coefficients : Array
        Coefficients of the binned Hellings and Downs correlations

    """

    # Ensure Hellings and Downs correlations values are within bounds
    xi_vals = jnp.arccos(jnp.clip(zeta_IJ, -1.0, 1.0))

    # Compute bin edges for binning the Hellings and Downs correlations values
    bin_edges = jnp.linspace(0.0, jnp.pi, HD_order + 2)

    # Compute mean values for each bin
    mean_x = 0.5 * (jnp.cos(bin_edges[:-1]) + np.cos(bin_edges[1:]))

    # Compute Hellings and Downs correlations coefficients for binned intervals
    HD_coefficients = HD_correlations(mean_x)

    # Create masks for each bin
    masks = np.zeros(shape=(HD_order + 1, len(zeta_IJ), len(zeta_IJ)))

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
    HD_order=0,
    HD_basis="legendre",
    anisotropies=False,
    lm_basis="spherical_harmonics_basis",
    l_max=0,
    nside=16,
    regenerate_catalog=False,
    **generate_catalog_kwargs,
):
    """
    Generate all tensors (noise, response and Hellings and Downs) needed for
    GW data analysis. The noise tensors is returned in omega units. The
    Hellings and Downs correlations are projected onto Legendre polynomials
    or binned intervals based on the chosen HD_basis

    Parameters:
    -----------
    frequencies : Array
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
    HD_order : int, optional
        Maximum order of Legendre polynomials/ number of bins for the Hellings
        and Downs correlations projection.
        Default is 0.
    HD_basis : str, optional
        Basis for Hellings and Downs correlations projection.
        Options are "legendre" or "binned".
        Default is "legendre".
    anisotropies : bool, optional
        Whether to include anisotropies in the response tensor.
        Default is False.
    lm_basis : str, optional
        Basis for the anisotropy decomposition.
        Options are "spherical_harmonics_basis" or "sqrt_basis".
        Default is "spherical_harmonics_basis".
    l_max : int, optional
        Maximum ell value for the anisotropy decomposition.
        Default is 0.
    nside : int, optional
        Resolution parameter for the HEALPix grid.
        Default is 16.
    regenerate_catalog : bool, optional
        Whether to regenerate the pulsars data file.
        Default is False.
    **generate_catalog_kwargs : dict, optional
        Additional keyword arguments for the generate_pulsars_catalog function.

    Returns:
    --------
    Tuple containing:
    - strain_omega : Array
        Noise converted to Omega units. The shape will be (F, N, N), where F
        is the number of frequencies and N is the number of pulsars.
    - response_IJ : Array
        Array containing the response tensor for all the pulsars pairs.
        If anisotropies is False, it has shape (F, N, N), where F is the number
        of frequencies and N is the number of pulsars. If anisotropies is True,
        it has shape (lm, F, N, N), where lm is the number of coefficients for
        the anisotropy decomposition (spherical harmonics or sqrt basis).
    - HD_functions_IJ : Array
        4D array with the Legendre or binned projection of the Hellings and
        Downs correlations. The shape is (HD_order + 1, F, N, N), where
        HD_order is the maximum order of Legendre polynomials / bins, F is the
        number of frequencies,and N is the number of pulsars.
    - HD_coefficients : Array
        Legendre coefficients for Hellings and Downs correlations values up to
        the given HD_order.

    """

    # Load or regenerate pulsar catalog
    try:
        if regenerate_catalog:
            raise FileNotFoundError

        pulsars_catalog = ut.load_table(path_to_pulsar_catalog)

    except FileNotFoundError:
        generate_catalog_kwargs["outname"] = path_to_pulsar_catalog
        pulsars_catalog = generate_pulsars_catalog(**generate_catalog_kwargs)

    # unpack all parameters
    WN_par = jnp.array(pulsars_catalog["wn"])
    log10_A_red = jnp.array(pulsars_catalog["log10_A_red"])
    gamma_red = jnp.array(pulsars_catalog["g_red"])
    log10_A_dm = jnp.array(pulsars_catalog["log10_A_dm"])
    gamma_dm = jnp.array(pulsars_catalog["g_dm"])
    log10_A_sv = jnp.array(pulsars_catalog["log10_A_sv"])
    gamma_sv = jnp.array(pulsars_catalog["g_sv"])
    Tspan_yr = jnp.array(pulsars_catalog["Tspan"])
    dt = jnp.array(pulsars_catalog["dt"])
    theta = jnp.array(pulsars_catalog["theta"])
    phi = jnp.array(pulsars_catalog["phi"])
    pi_vec = gds.unit_vector(theta, phi)

    # get noises for all pulsars
    noise = pn.get_pulsar_noises(
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

        curn = pn.get_pl_colored_noise(frequencies, log10_A_curn, gamma_curn)
        noise += curn

    # convert the noise in strain and then omega units
    strain_omega = pn.get_noise_omega(frequencies, noise)

    # get the time tensor
    time_tensor_IJ = get_time_tensor(frequencies, pta_span_yrs, Tspan_yr)

    # compute angular separations
    zeta_IJ = jnp.einsum("ik, jk->ij", pi_vec, pi_vec)

    if not anisotropies:
        # compute the response
        response_IJ = get_response_IJ(zeta_IJ, time_tensor_IJ)

    else:
        response_IJ = get_response_IJ_lm(
            pi_vec, time_tensor_IJ, l_max, nside, lm_basis=lm_basis
        )

    # and if needed the HD part
    if HD_order > 0 and HD_basis.lower() == "legendre":
        HD_functions_IJ, HD_coefficients = HD_projection_Legendre(
            zeta_IJ, time_tensor_IJ, HD_order
        )

    elif HD_order > 0 and HD_basis.lower() == "binned":
        HD_functions_IJ, HD_coefficients = HD_projection_binned(
            zeta_IJ, time_tensor_IJ, HD_order
        )

    else:
        HD_functions_IJ = jnp.zeros(
            shape=(0, len(frequencies), len(WN_par), len(WN_par))
        )
        HD_coefficients = jnp.zeros(shape=(0,))

    return strain_omega, response_IJ, HD_functions_IJ, HD_coefficients
