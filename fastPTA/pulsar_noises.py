# Global
import jax
import jax.numpy as jnp

# Local
import fastPTA.utils as ut


@jax.jit
def get_WN(WN_parameters, dt):
    """
    Compute the white noise amplitude for a catalog of pulsars given the
    the white noise amplitudes and sampling rates (see Eq. 5 of 2404.02864).
    The time step dt should be provided in seconds.

    Parameters:
    -----------
    WN_parameters : Array
        White noise parameters for the pulsar.
    dt : Array
        Time steps for the pulsar.

    Returns:
    --------
    WN_amplitude : Array
        Array of white noise amplitudes.

    """

    return 1e-100 + jnp.array(1e-12 * 2 * WN_parameters**2 * dt)


@jax.jit
def get_pl_colored_noise(frequencies, log10_ampl, gamma):
    """
    Compute power-law colored noise for given frequencies and parameters.

    Parameters:
    -----------
    frequencies : Array
        Array of frequencies (in Hz) at which to compute the colored noise.
    log10_ampl : Array
        Array of base-10 logarithm of amplitudes.
    gamma : Array
        Array of power-law indices.

    Returns:
    --------
    colored_noise : Array
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
    frequencies : Array
        Array of frequencies.
    noise : Array
        Array representing the noise.

    Returns:
    --------
    noise_omega : Array
        Noise converted to Omega units. The shape will be (F, N, N), where F
        is the number of frequencies and N is the number of pulsars.

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
    frequencies : Array
        Array of frequencies (in Hz).
    WN_par : float
        Array of white noise parameter.
    log10_A_red : Array
        Array of base-10 logarithm of amplitudes for red noise.
    gamma_red : Array
        Array of power-law indices for red noise.
    log10_A_dm : Array
        Array of base-10 logarithm of amplitudes for dispersion measure noise.
    gamma_dm : Array
        Array of power-law indices for dispersion measure noise.
    log10_A_sv : Array
        Array of base-10 logarithm of amplitudes for scattering variation noise.
    gamma_sv : Array
        Array of power-law indices for scattering variation noise.
    dt : float
        Array of time steps in seconds.

    Returns:
    --------
    noise : Array
        3D array of noise components computed for the given parameters and
        frequencies. The shape will be (F, N, N), where F is the number of
        frequencies and N is the number of pulsars.

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
