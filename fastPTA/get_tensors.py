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

# If you want to use your GPU change here
jax.config.update("jax_default_device", jax.devices("cpu")[0])


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
    return jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ]
    ).T


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

    # The 1e-15 term is added to regularize the log
    diff_IJ = 0.5 * (1.0 - zeta_IJ) + 1e-15

    # This function does not include the Kronecker-Delta term for pulsar
    # self-correlations
    return 0.5 + 1.5 * diff_IJ * (jnp.log(diff_IJ) - 1 / 6)


# Some default values to compute the HD curve
x = jnp.linspace(-1, 1, integration_points)
HD_value = HD_correlations(x)


@jax.jit
def get_WN(WN_par, dt):
    """
    Compute the white noise amplitude for a catalog of pulsars given the
    the white noise amplitudes and sampling rates. The time step dt should be
    provided in seconds.

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
    Tspan_yr,
    dt,
):
    """
    Compute noise components for given parameters and frequencies.
    The components included are:
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
    Tspan_yr : float
        Array of time spans in years.
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
    Compute the transmission function, which represents the attenuation of
    signals, for some frequencies given the observation time.

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

    # Do a mesh of the observation times and pick the minimium in each pair
    time_1, time_2 = jnp.meshgrid(Tspan_yr, Tspan_yr)
    time_IJ = jnp.min(jnp.array([time_1, time_2]), axis=0)

    # Compute thte transmission function for all times and frequencies
    transmission = transmission_function(
        frequencies[:, None], (Tspan_yr * ut.yr)[None, :]
    )

    # Return the tensor product weighted by the total observation time
    return jnp.sqrt(
        (time_IJ / pta_span_yrs)[None, ...]
        * transmission[:, :, None]
        * transmission[:, None, :]
    )


@jax.jit
def Gamma(p, q, Omega):
    """
    TO ADD
    """
    pq = jnp.einsum("iv,jv->ij", p, q)
    pOmega = jnp.einsum("iv,jv->ij", p, Omega)
    qOmega = jnp.einsum("iv,jv->ij", q, Omega)
    numerator = (
        2 * (pq[..., None] - pOmega[:, None, :] * qOmega[None, ...]) ** 2
    )
    denominator = (1 + pOmega[:, None, :]) * (1 + qOmega[None, ...])
    term2 = -(1 - pOmega[:, None, :]) * (1 - qOmega[None, ...])
    return numerator / (1e-30 + denominator) + term2


def get_correlations_lm_IJ(p_I, lm_order, nside):
    """
    TO ADD
    """

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, jnp.arange(npix))
    theta = jnp.array(theta)
    phi = jnp.array(phi)

    gamma_pq = 3 / 8 * Gamma(p_I, p_I, unit_vector(theta, phi))

    correlations_lm = np.zeros(
        shape=(int(1 + lm_order) ** 2, len(p_I), len(p_I))
    )

    i = 0
    for ell in tqdm.tqdm(range(lm_order + 1)):
        for m in jnp.linspace(0, ell, ell + 1, dtype=int):
            if m != 0:
                sp_lm1 = sph_harm(m, ell, phi, theta) * jnp.sqrt(4 * jnp.pi)
                sp_lm2 = sph_harm(-m, ell, phi, theta) * jnp.sqrt(4 * jnp.pi)
                sp1 = (1j / jnp.sqrt(2) * (sp_lm1 - (-1) ** m * sp_lm2)).real
                sp2 = (1 / jnp.sqrt(2) * (sp_lm2 + (-1) ** m * sp_lm1)).real
                correlations_lm[i] = jnp.mean(
                    gamma_pq * sp1[None, None, :], axis=-1
                )
                correlations_lm[i + 1] = jnp.mean(
                    gamma_pq * sp2[None, None, :], axis=-1
                )

                i += 2

            else:
                sp0 = sph_harm(m, ell, phi, theta).real * np.sqrt(4 * jnp.pi)
                correlations_lm[i] = jnp.mean(
                    gamma_pq * sp0[None, None, :], axis=-1
                )

                i += 1

    return correlations_lm * (1 + np.eye(len(p_I)))[None, ...]


def get_response_IJ_lm(p_I, time_tensor_IJ, lm_order, nside):
    """
    TO ADD

    """

    # Compute the correlations on lm basis
    # To understand if a factor 0.5 * jnp.eye(len(zeta_IJ)) is missing!!!!
    correlations_lm_IJ = get_correlations_lm_IJ(p_I, lm_order, nside)

    # combine the Hellings and Downs part and the time part
    return time_tensor_IJ[None, ...] * correlations_lm_IJ[:, None, ...]


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

    # Compute HD and add the self correlation term
    chi_tensor_IJ = HD_correlations(zeta_IJ) + 0.5 * jnp.eye(len(zeta_IJ))

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


@jax.jit
def Legendre_projection(time_tensor_IJ, polynomials_IJ):
    """
    Projects the pulsar angular information onto Legendre polynomials

    Parameters:
    -----------
    time_tensor_IJ : numpy.ndarray or jax.numpy.ndarray
        Time tensor containing the attenuations due to the observation time for
        all the pulsars and for all frequencies.
    polynomials_IJ : numpy.ndarray or jax.numpy.ndarray
        Array of Legendre polynomials.

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
    polynomials_IJ = jnp.array([legendre(i)(zeta_IJ) for i in range(order + 1)])

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
    lm_order=0,
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
        Tspan_yr,
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

    if not anisotropies:
        # compute angular separations
        zeta_IJ = jnp.einsum("ik, jk->ij", pi_vec, pi_vec)

        # compute the response
        response_IJ = get_response_IJ(zeta_IJ, time_tensor_IJ)

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

    else:

        response_IJ = get_response_IJ_lm(
            pi_vec, time_tensor_IJ, lm_order, nside
        )

        HD_functions_IJ = jnp.zeros(
            shape=(0, len(frequencies), len(WN_par), len(WN_par))
        )
        HD_coefficients = jnp.zeros(shape=(0,))

    return strain_omega, response_IJ, HD_functions_IJ, HD_coefficients
