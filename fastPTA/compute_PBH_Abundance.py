import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator


# Local
import fastPTA.utils as ut


if ut.compare_versions(jax.__version__, "0.4.24"):
    from jax.numpy import trapezoid
else:
    from jax.numpy import trapz as trapezoid

from functools import partial


# Set some global parameters for jax
jax.config.update("jax_enable_x64", True)

# If you want to use your GPU change here
jax.config.update("jax_default_device", jax.devices("cpu")[0])


# Constants
h = 0.674
Meq = 2.8e17
rm_to_k = 2.227
Omega_DM = 0.12 / h**2
g_sm_tot = 106.75

# Flag to regenerate the file containing the temperature as a function of the
# hubble mass in solar masses and of the comoving wavenumber in Mpc^{-1}
regenerate_T_file = False

# Importing data
g_data = np.loadtxt(ut.path_to_defaults + "gstar_T.txt")

# This contains some quantities for the PBH abundance
data_f_PBH_MH = np.loadtxt(ut.path_to_defaults + "data_f_PBH_MH.txt")


# --- Define interpolators from the data
# Effective number of relativistic dofs as a function of temperature in MeV
relativistic_dofs = RegularGridInterpolator(
    (g_data[:, 0],),
    g_data[:, 1],
    bounds_error=False,
    fill_value=None,
)

# Effective number of entropy dofs as a function of temperature in MeV
entropy_dofs = RegularGridInterpolator(
    (g_data[:, 0],),
    g_data[:, 3],
    bounds_error=False,
    fill_value=None,
)

# Kappa of critical collapse (appearing in eq A11 of 2503.10805) as a function
# of mass in a Hubble volume in solar masses
kappa_QCD = RegularGridInterpolator(
    (data_f_PBH_MH[:, 0],),
    data_f_PBH_MH[:, 1],
    bounds_error=False,
    fill_value=None,
)

# Gamma of critical collapse (appearing in eq A11 of 2503.10805) as a function
# of mass in a Hubble volume in solar masses
gamma_QCD = RegularGridInterpolator(
    (data_f_PBH_MH[:, 0],),
    data_f_PBH_MH[:, 2],
    bounds_error=False,
    fill_value=None,
)

# Critical value of the compaction function (lower limit for the integral in
# eq A11 of 2503.10805) as a function of mass in a Hubble volume in solar masses
delta_QCD = RegularGridInterpolator(
    (data_f_PBH_MH[:, 0],),
    data_f_PBH_MH[:, 3],
    bounds_error=False,
    fill_value=None,
)

# Phi (scalar potential appearing in eq A3 of 2503.10805) as a function of
# mass in a Hubble volume in solar masses
phi_QCD = RegularGridInterpolator(
    (data_f_PBH_MH[:, 0],),
    data_f_PBH_MH[:, 4],
    bounds_error=False,
    fill_value=None,
)

# After we defined the interpolators we can delete the data
del g_data, data_f_PBH_MH


@jax.jit
def k_of_T_MeV(temperature_MeV):
    """
    Compute the comoving wavenumber k corresponding to a given temperature in
    MeV (see eq. 2 of 2503.10805).

    Parameters:
    -----------
    temperature_MeV : Array
        Temperature in MeV.

    Returns:
    --------
    k : Array
        Comoving wavenumber k in Mpc^{-1}.

    """

    # The factor 1e3 is to convert MeV to GeV
    prefactor = (1.5e7 / 1e3) * temperature_MeV

    # Second factor in eq. 2 of 2503.10805
    relativistic_dofs_term = jnp.sqrt(
        relativistic_dofs(temperature_MeV) / g_sm_tot
    )

    # Third factor in eq. 2 of 2503.10805
    entropy_dofs_term = (entropy_dofs(temperature_MeV) / g_sm_tot) ** (
        -1.0 / 3.0
    )

    return prefactor * entropy_dofs_term * relativistic_dofs_term


@jax.jit
def hubble_mass_of_T_MeV(temperature_MeV):
    """
    Compute the mass in a Hubble volume in solar masses corresponding to a given
    temperature in MeV (see eq. A2 of 2503.10805).

    Parameters:
    -----------
    temperature_MeV : Array
        Temperature in MeV.

    Returns:
    --------
    M_H : Array
        mass in a Hubble volume in solar masses.

    """

    prefactor = 4.76e4 / temperature_MeV**2.0
    relativistic_dofs_term = jnp.sqrt(
        relativistic_dofs(temperature_MeV) / g_sm_tot
    )

    return prefactor / relativistic_dofs_term


# If the flag is set to True, regenerate the file
if regenerate_T_file:
    T_range = jnp.geomspace(1e-3, 1e10, 10000)
    M_of_T = hubble_mass_of_T_MeV(T_range)
    k_of_T = k_of_T_MeV(T_range)
    np.savetxt(
        ut.path_to_defaults + "T_data.txt",
        np.vstack((T_range, M_of_T, k_of_T)).T,
    )

# Load the data if the flag is set to False
else:
    T_range, M_of_T, k_of_T = np.loadtxt(ut.path_to_defaults + "T_data.txt").T

# Define the interpolators
# The mass in is solar masses, k is in Mpc^{-1} and T is in MeV
T_of_M_H = RegularGridInterpolator((jnp.flip(M_of_T),), jnp.flip(T_range))
T_of_k = RegularGridInterpolator((k_of_T,), T_range)

# Delete the data
del T_range, M_of_T, k_of_T


@jax.jit
def M_H_of_k(k_vec_mpc, rm_to_k_factor):
    """
    Compute the mass in a Hubble volume corresponding to a given comoving
    wavenumber. This matches 36 of 2503.10805.

    Parameters:
    -----------
    k_vec_mpc : Array
        Comoving wavenumber k in Mpc^{-1}.
    rm_to_k_factor : float
        Parameter mapping length scales in wave numbers.

    Returns:
    --------
    M_H : Array
        mass in a Hubble volume in solar masses.

    """

    # Compute the temperature corresponding to the comoving wavenumber
    temperature = T_of_k(k_vec_mpc)

    # Compute the k_dependent term in eq. 36 of the draft
    prefactor = 10.7e12 * (k_vec_mpc / rm_to_k_factor) ** -2.0

    # Compute the relativistic and entropy dofs terms in eq. 36 of the draft
    relativistic_dofs_term = jnp.sqrt(relativistic_dofs(temperature) / g_sm_tot)
    entropy_dofs_term = (entropy_dofs(temperature) / g_sm_tot) ** (-2.0 / 3.0)

    return prefactor * relativistic_dofs_term * entropy_dofs_term


@jax.jit
def window(k_vec, r_max):
    """
    Compute the window function (see eq A10 of 2503.10805).

    Parameters:
    -----------
    k_vec : Array
        Comoving wavenumber k in some units.
    r_max : float
        Typical size of the perturbation in the same units as 1 / k_vec.

    Returns:
    --------
    window : Array
        Window function.

    """

    # The dimensionless variable appearing in the window function
    x = k_vec * r_max

    return 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x**3.0


@jax.jit
def transfer_function(k_vec, r_max):
    """
    Compute the window function (see eq A10 of 2503.10805).

    Parameters:
    -----------
    k_vec : Array
        Comoving wavenumber k in some units.
    r_max : float
        Typical size of the perturbation in the same units as 1 / k_vec.

    Returns:
    --------
    transfer_function : Array
        Transfer function.

    """

    return window(k_vec, r_max / jnp.sqrt(3.0))


@jax.jit
def integrand_spectrum(k_vec, r_max, scalar_spectrum):
    """
    Compute the integrand in eq A8 of 2503.10805 given spectrum, window and
    transfer function.

    Parameters:
    -----------
    k_vec : Array
        Comoving wavenumber k in some units.
    r_max : Array
        Length scale for the perturbations in the same units as 1 / k_vec.
    scalar_spectrum : Array
        Scalar power spectrum.

    Returns:
    --------
    integrand_spectrum : Array
        Integrand of the spectrum.
    """

    # Evaluate the factors in eq A8 of 2503.10805
    prefactor = (r_max * k_vec) ** 4.0
    window_term = window(k_vec, r_max) ** 2.0
    transfer_term = transfer_function(k_vec, r_max) ** 2.0

    return prefactor * window_term * transfer_term * scalar_spectrum


@jax.jit
def compute_sigma_c_NL_QCD(k_vec, r_max, scalar_spectrum, mass_hubble_volume):
    """
    Compute the std (as sqrt of the variance) of the compaction function.
    This is the square root of the integral in eq. A8 of 2503.10805.

    Parameters:
    -----------
    k_vec : Array
        Comoving wavenumber k in some units.
    r_max : float
        Length scale for the perturbations in the same units as 1 / k_vec.
    scalar_spectrum : Array
        Scalar power spectrum.
    mass_hubble_volume : Array
        mass in a Hubble volume in solar masses.

    Returns:
    --------
    sigma_c : float
        Standard deviation of the compaction function.

    """

    # Compute the integrand in eq A8 of 2503.10805
    to_integrate = integrand_spectrum(k_vec, r_max, scalar_spectrum)

    # Compute the integral in eq A8 of 2503.10805
    integral_result = trapezoid(to_integrate, x=jnp.log(k_vec), axis=-1)

    # Return the square root of the integral times Phi
    return jnp.sqrt((4.0 / 9.0) * integral_result) * phi_QCD(mass_hubble_volume)


@jax.jit
def P_G(cal_C_G, sigma_c):
    """
    Compute the probability of the compaction function given its variance.
    This function corresponds to eq A7 after marginalizing over the scalar
    perturbation zeta_G (-inf,inf).

    Parameters:
    -----------
    cal_C_G : Array
        Gaussian part of the compaction function in some units.
    sigma_c : float
        Standard deviation of the compaction function. This quantity must
        be in the same units as cal_C_G.

    Returns:
    --------
    P_G : Array
        Probability of the compaction function.

    """

    normalization = 1.0 / sigma_c / jnp.sqrt(2.0 * jnp.pi)
    exp_term = jnp.exp(-((cal_C_G / sigma_c) ** 2.0) / 2.0)

    return normalization * exp_term


@jax.jit
def integrand_beta(cal_C_G, sigma_c, mass_hubble_volume):
    """
    Compute the integrand in eq A12 of 2503.10805.

    Parameters:
    -----------
    cal_C_G : Array
        Gaussian part of the compaction function in some units.
    mass_hubble_volume : Array
        mass in a Hubble volume in solar masses.
    sigma_c : float
        Standard deviation of the compaction function. This quantity must be in
        the same units as cal_C_G.

    Returns:
    --------
    integrand : Array
        Integrand in eq A12 of 2503.10805.

    """

    # All the "constants" in eq A12 of 2503.10805 vary around the QCD PT
    # A coefficient related to the eos parameter (see text after eq A4)
    P_QCD = phi_QCD(mass_hubble_volume)

    # This is the threshold value cal_C_th to form a PBH cal_C_th see eq A12
    cal_C_th = delta_QCD(mass_hubble_volume)

    # Proportionality coefficient for critical collapse (see eq A12)
    K_QCD = kappa_QCD(mass_hubble_volume)

    # Critical exponent for the critical collapse (see eq A12)
    G_QCD = gamma_QCD(mass_hubble_volume)

    # This is the quantity in the first condition in eq A13 of 2503.10805
    # we take the full cal_C as Gaussian part + first quadratic term - threshold
    condition_1 = cal_C_G - 1.0 / (4.0 * P_QCD) * cal_C_G**2 - cal_C_th

    # This is the quantity in the second condition in eq A13 of 2503.10805
    condition_2 = 2.0 * P_QCD - cal_C_G

    # This is the integrand in eq A12 of 2503.10805
    result = K_QCD * condition_1**G_QCD * P_G(cal_C_G, sigma_c)

    # Return the integrand where the conditions are satisfied else return 0
    return jnp.where((condition_1 >= 0.0) & (condition_2 >= 0.0), result, 0.0)


@jax.jit
def compute_beta_NL_C_QCD(
    k_vec, r_max, scalar_spectrum, mass_hubble_volume, cal_C_G_vec
):
    """
    Compute the beta (for the f_PBH integral) in eq A12 of 2503.10805.

    Parameters:
    -----------
    k_vec : Array
        Comoving wavenumber k in some units.
        This is a 2d array, the first index runs over the masses and the second
        index runs over the set of ks we integrate over to get the variance.
    r_max : Array
        Typical size of the perturbation in the same units as 1 / k_vec.
        This is a 1d array with size equal to the second axis of k_vec.
    scalar_spectrum : Array
        Scalar power spectrum.
        This is a 2d array with the same shape as k_vec.
    mass_hubble_volume : Array
        mass in a Hubble volume in solar masses.
    cal_C_G_vec : Array
        Gaussian part of the compaction function in some units.
        This is a 2d array, the first index runs over the set of values to be
        used in the cal_G integral, the second index runs over masses.

    Returns:
    --------
    beta : Array
        Beta in eq A12 of 2503.10805.

    """

    # Get sigma_c from eq A8 of 2503.10805
    sigma_c = compute_sigma_c_NL_QCD(
        k_vec, r_max, scalar_spectrum, mass_hubble_volume
    )

    # Compute the integrand in eq A12 of 2503.10805
    beta_integrand = integrand_beta(cal_C_G_vec, sigma_c, mass_hubble_volume)

    # Return the integral in eq A12 of 2503.10805
    return trapezoid(beta_integrand, x=cal_C_G_vec, axis=0)


@partial(jax.jit, static_argnums=(3,))
def f_PBH_NL_QCD(r_max_vec_mpc, k_vec_mpc, scalar_spectrum, len_C_G_vec=100):
    """
    Compute the PBH abundance assuming a lognormal scalar power spectrum.

    Parameters:
    -----------
    r_max_vec_mpc : Array
        Typical size of the perturbation in Mpc.
        This is a 1d array with size equal to the first axis of k_vec_mpc.
    k_vec_mpc : Array
        Comoving wavenumber k in Mpc^{-1}.
        This is a 2d array, the first index runs over the masses and the second
        index runs over the set of ks we integrate over to get the variance.
    scalar_spectrum : Array
        Scalar power spectrum evaluated in k_vec_mpc.
    len_C_G_vec : int, optional
        Number of points in the CG vector (default is 100).

    Returns:
    --------
    f_PBH : float
        PBH abundance.

    """

    # mass in a Hubble volume corresponding to the comoving wavenumber
    M_H_vec = M_H_of_k(rm_to_k / r_max_vec_mpc, rm_to_k)

    # vector of values for the Gaussian part of the curvature perturbations
    cal_C_G_vec = jnp.linspace(0.0, 2 * phi_QCD(M_H_vec), len_C_G_vec)

    # compute the temperature corresponding to the mass in a Hubble volume
    T_v = T_of_M_H(M_H_vec)

    # some auxiliary quantities
    relativistic_dofs_term = (relativistic_dofs(T_v) / g_sm_tot) ** 0.75
    entropy_dofs_term = entropy_dofs(T_v) / g_sm_tot
    M_H_factor = 1.0 / 7.9e-10 / Omega_DM / M_H_vec**1.5

    # compute the prefactor in eq. A1 of 2503.10805
    prefactor = M_H_factor * relativistic_dofs_term / entropy_dofs_term

    # compute the beta as given by A12 of 2503.10805
    integrand_M = compute_beta_NL_C_QCD(
        k_vec_mpc, r_max_vec_mpc[:, None], scalar_spectrum, M_H_vec, cal_C_G_vec
    )

    # get f_PBH integrating beta over the mass in a Hubble volume
    return trapezoid(prefactor * integrand_M, x=M_H_vec)


# Lognormal spectrum
@jax.jit
def lognormal_spectrum(k_vec, amplitude, delta, ks):
    """
    A lognormal spectrum.

    Parameters:
    -----------
    k_vec : Array
        Comoving wavenumber k in some units.
    amplitude : float
        Amplitude of the spectrum.
    delta : float
        Width parameter.
    ks : float
        Pivot scale in the same units as k_vec.

    Returns:
    --------
    lognormal_spectrum : Array
        Lognormal spectrum.

    """

    normalization = 1.0 / delta / jnp.sqrt(2.0 * jnp.pi)
    exp_term = jnp.exp(-((jnp.log(k_vec / ks) / delta) ** 2.0) / 2.0)

    return amplitude * exp_term * normalization


@partial(jax.jit, static_argnums=(3, 4, 5))
def f_PBH_NL_QCD_lognormal(
    amplitude, delta, ks, len_k_vec=100, len_r_max_vec=100, len_C_G_vec=100
):
    """
    Compute the PBH abundance assuming a lognormal scalar power spectrum.

    Parameters:
    -----------
    amplitude : float
        Amplitude of the spectrum.
    delta : float
        Width of the lognormal spectrum.
    ks : float
        Pivot scale in Mpc^{-1}.
    len_k_vec : int, optional
        Number of points in the k vector (default is 100).
    len_r_max_vec : int, optional
        Number of points in the curvature vector (default is 100).
    len_C_G_vec : int, optional
        Number of points in the CG vector (default is 100).

    Returns:
    --------
    f_PBH : float
        PBH abundance.

    """

    # Define a range of values for the comoving wavenumber
    r_max_vec = jnp.logspace(-0.6, 1.2, len_r_max_vec)

    # get minimal and maximal values for the k
    k_min = 10.0 ** (-3.0 * delta)
    k_max = jnp.where(1 / k_min <= 1e3 / r_max_vec, 1 / k_min, 1e3 / r_max_vec)

    # define the k vector in log space
    k_vec = ks * jnp.geomspace(k_min, k_max, len_k_vec).T

    # compute the scalar spectrum (notice that we defined k_vec in units of ks)
    # when we call lognormal_spectrum we need to pass k_vec in units of ks or
    # otherwise we could pass k_vec and ks=1.0
    scalar_spectrum = lognormal_spectrum(k_vec, amplitude, delta, ks)

    # evaluate the PBH abundance
    return f_PBH_NL_QCD(
        r_max_vec / ks, k_vec, scalar_spectrum, len_C_G_vec=len_C_G_vec
    )


# @jax.jit
def find_A_NL_QCD(log10fPBH, Delta, ks, A_min=-2.5, A_max=-1.0):
    """
    Given some value of f_PBH, find the amplitude of the lognormal spectrum
    that gives that value.

    Parameters:
    -----------
    log10fPBH : float
        Logarithm in base 10 of the PBH abundance.
    Delta : float
        Width of the lognormal spectrum.
    ks : float
        Pivot scale in Mpc^{-1}.

    Returns:
    --------
    A : float
        Amplitude of the lognormal spectrum.

    """

    for _ in range(20):
        val = f_PBH_NL_QCD_lognormal(10.0 ** ((A_min + A_max) / 2), Delta, ks)
        A_max = jnp.where(val >= 10.0**log10fPBH, (A_min + A_max) / 2, A_max)
        A_min = jnp.where(val >= 10.0**log10fPBH, A_min, (A_min + A_max) / 2)

    return (A_min + A_max) / 2
