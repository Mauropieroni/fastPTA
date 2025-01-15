import numpy as np

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.interpolate import RegularGridInterpolator


# Local
import fastPTA.utils as ut

# Set some global parameters for jax
jax.config.update("jax_enable_x64", True)

# If you want to use your GPU change here
jax.config.update("jax_default_device", jax.devices("cpu")[0])  # type: ignore

# Constants
h = 0.674
Meq = 2.8e17
rorm = 2.227
kappa = rorm

Omega_DM = 0.12 / h**2
T_range_up = jnp.geomspace(1e-3, 1e10, 10000)
T_range_down = jnp.geomspace(1e10, 1e-3, 10000)

# Importing data
g_data = np.loadtxt(ut.path_to_defaults + "gstar_T.txt")

g_relativistic_dofs = g_data[:, [0, 1]] * jnp.array([1000, 1])
g_entropy_dofs = g_data[:, [0, 3]] * jnp.array([1000, 1])

# Define interpolators from the data
relativistic_dofs = RegularGridInterpolator(
    [g_relativistic_dofs[:, 0]],
    g_relativistic_dofs[:, 1],
    bounds_error=False,
    fill_value=None,
)

entropy_dofs = RegularGridInterpolator(
    [g_entropy_dofs[:, 0]],
    g_entropy_dofs[:, 1],
    bounds_error=False,
    fill_value=None,
)

# Import data
tb_Kappa_v_T = np.loadtxt(ut.path_to_defaults + "Kappa_alpha3_fits.txt")
tb_Gamma_v_T = np.loadtxt(ut.path_to_defaults + "gamma_alpha3_fits.txt")
tb_Delta_cv_T = np.loadtxt(ut.path_to_defaults + "deltac_alpha3_fits.txt")
tb_Phi_v_T = np.loadtxt(ut.path_to_defaults + "Phi_alpha3_fits.txt")


# Interpolators from the data
Kappa_QCD = RegularGridInterpolator(
    [tb_Kappa_v_T[:, 0]],
    tb_Kappa_v_T[:, 1],
    bounds_error=False,
    fill_value=None,
)

Gamma_QCD = RegularGridInterpolator(
    [tb_Gamma_v_T[:, 0]],
    tb_Gamma_v_T[:, 1],
    bounds_error=False,
    fill_value=None,
)

Delta_QCD = RegularGridInterpolator(
    [tb_Delta_cv_T[:, 0]],
    tb_Delta_cv_T[:, 1],
    bounds_error=False,
    fill_value=None,
)

Phi_QCD = RegularGridInterpolator(
    [tb_Phi_v_T[:, 0]],
    tb_Phi_v_T[:, 1],
    bounds_error=False,
    fill_value=None,
)

# @jax.jit
# def Kappa_QCD(temperature):
#     return jnp.interp(
#         temperature,
#         tb_Kappa_v_T[:, 0],
#         tb_Kappa_v_T[:, 1],
#         left=tb_Kappa_v_T[0, 1],
#         right=tb_Kappa_v_T[-1, 1],
#     )


# @jax.jit
# def Gamma_QCD(temperature):
#     return jnp.interp(
#         temperature,
#         tb_Gamma_v_T[:, 0],
#         tb_Gamma_v_T[:, 1],
#         left=tb_Gamma_v_T[0, 1],
#         right=tb_Gamma_v_T[-1, 1],
#     )


# @jax.jit
# def Delta_QCD(temperature):
#     return jnp.interp(
#         temperature,
#         tb_Delta_cv_T[:, 0],
#         tb_Delta_cv_T[:, 1],
#         left=tb_Delta_cv_T[0, 1],
#         right=tb_Delta_cv_T[-1, 1],
#     )


# @jax.jit
# def Phi_QCD(temperature):
#     return jnp.interp(
#         temperature,
#         tb_Phi_v_T[:, 0],
#         tb_Phi_v_T[:, 1],
#         left=tb_Phi_v_T[0, 1],
#         right=tb_Phi_v_T[-1, 1],
#     )


# @jax.jit
# def relativistic_dofs(temperature):
#     return jnp.interp(
#         temperature,
#         g_relativistic_dofs[:, 0],
#         g_relativistic_dofs[:, 1],
#         left=g_relativistic_dofs[0, 1],
#         right=g_relativistic_dofs[-1, 1],
#     )


# @jax.jit
# def entropy_dofs(temperature):
#     return jnp.interp(
#         temperature,
#         g_entropy_dofs[:, 0],
#         g_entropy_dofs[:, 1],
#         left=g_entropy_dofs[0, 1],
#         right=g_entropy_dofs[-1, 1],
#     )

del g_data, g_relativistic_dofs, g_entropy_dofs

# Values at specific points
relativistic_dofs_0 = relativistic_dofs(jnp.array([1e-9]))
entropy_dofs_0 = entropy_dofs(jnp.array([1e-9]))


# Interpolators for the temperature
@jax.jit
def k_of_T(temperature):
    return (
        1.5e7
        * (entropy_dofs(temperature) / 106.75) ** (-1.0 / 3.0)
        * jnp.sqrt(relativistic_dofs(temperature) / 106.75)
        * (temperature / 1e3)
    )


@jax.jit
def M_H_of_T(temperature):
    return (
        1.5e5
        / temperature**2.0
        / jnp.sqrt(relativistic_dofs(temperature) / 10.75)
    )


T_of_MH = RegularGridInterpolator([M_H_of_T(T_range_down)], T_range_down)
T_of_k = RegularGridInterpolator([k_of_T(T_range_up)], T_range_up)


# @jax.jit
# def T_of_MH(MH):
#     return jnp.interp(
#         MH,
#         M_H_of_T(T_range_down),
#         T_range_down,
#         # left=T_range_down[0],
#         # right=T_range_down[-1],
#     )


# @jax.jit
# def T_of_k(k):
#     return jnp.interp(
#         k,
#         k_of_T(T_range_up),
#         T_range_up,
#         #   left=T_range_up[0], right=T_range_up[-1]
#     )


@jax.jit
def M_H_of_k(k_vec, kappa):
    temperature = T_of_k(k_vec)
    return (
        1.07
        * 10.0
        * jnp.sqrt(relativistic_dofs(temperature) / 106.75)
        * (entropy_dofs(temperature) / 106.75) ** (-2.0 / 3.0)
        * (1e6 / (k_vec / kappa)) ** 2.0
    )


# Some Functions
@jax.jit
def window(k_vec, curvature):
    x = k_vec * curvature
    return 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x**3.0


@jax.jit
def transfer_function(k_vec, Rh):
    curvature = Rh / jnp.sqrt(3.0)
    return window(k_vec, curvature)


@jax.jit
def PC(log_k, delta):
    return (
        1.0
        / (jnp.sqrt(2.0 * jnp.pi) * delta)
        * jnp.exp(-(1.0 / 2.0) * (log_k / delta) ** 2.0)
    )


# Lognormal spectrum
@jax.jit
def lognormal_spectrum(k_vec, amplitude, delta):
    return (
        amplitude
        / (jnp.sqrt(2.0 * jnp.pi) * delta)
        * jnp.exp(-1.0 / 2.0 * jnp.log(k_vec) ** 2.0 / delta**2)
    )


@jax.jit
def integrand_spectrum(k_vec, amplitude, delta, curvature_matter):
    return (
        1.0
        / k_vec
        * (curvature_matter * k_vec) ** 4.0
        * window(k_vec, curvature_matter) ** 2.0
        * transfer_function(k_vec, curvature_matter) ** 2.0
        * lognormal_spectrum(k_vec, amplitude, delta)
    )


@jax.jit
def compute_var_NL_QCD(k_vec, amplitude, delta, curvature_matter, horizon_mass):

    to_integrate = integrand_spectrum(k_vec, amplitude, delta, curvature_matter)
    integral_result = jnp.trapezoid(to_integrate, x=k_vec, axis=0)

    return jnp.sqrt((4.0 / 9.0) * integral_result) * Phi_QCD(horizon_mass)


@jax.jit
def integrand_beta(CG, horizon_mass, sigma_C):

    P_QCD = Phi_QCD(horizon_mass)
    D_QCD = Delta_QCD(horizon_mass)
    K_QCD = Kappa_QCD(horizon_mass)
    G_QCD = Gamma_QCD(horizon_mass)

    condition_1 = 2.0 * P_QCD - CG
    condition_2 = CG - 1.0 / (4.0 * P_QCD) * CG**2 - D_QCD
    result = K_QCD * condition_2**G_QCD * PC(CG, sigma_C)

    return jnp.where((condition_1 >= 0.0) & (condition_2 >= 0.0), result, 0.0)


@jax.jit
def compute_beta_NL_C_QCD(
    k_vec, amplitude, delta, curvature_matter, horizon_mass, CG_vec
):

    sigma_C = compute_var_NL_QCD(
        k_vec, amplitude, delta, curvature_matter, horizon_mass
    )

    beta_integrand = integrand_beta(CG_vec, horizon_mass, sigma_C)

    return jnp.trapezoid(beta_integrand, x=CG_vec, axis=0)


@jax.jit
def f_PBH_NL_QCD(
    amplitude, delta, ks, len_k=100, len_curvature=100, len_CG=100
):
    curvature_vec = jnp.logspace(-0.6, 1.2, len_curvature)
    M_H_vec = M_H_of_k(kappa / curvature_vec * ks, kappa)

    T_v = T_of_MH(M_H_vec)
    prefactor = (
        1.0
        / Omega_DM
        / M_H_vec
        * (1.0 / M_H_vec) ** 0.5
        * (relativistic_dofs(T_v) / 106.75) ** 0.75
        / (entropy_dofs(T_v) / 106.75)
        / 7.9e-10
    )

    k_min = 10.0 ** (-3.0 * delta)
    k_max = jnp.where(
        1 / k_min <= 1e3 / curvature_vec, 1 / k_min, 1e3 / curvature_vec
    )
    k_vec = jnp.geomspace(k_min, k_max, len_k)

    CG_vec = jnp.linspace(0.0, 2 * Phi_QCD(M_H_vec), len_CG)

    integrand_M = compute_beta_NL_C_QCD(
        k_vec, amplitude, delta, curvature_vec, M_H_vec, CG_vec
    )

    return jnp.trapezoid(prefactor * integrand_M, x=M_H_vec)


@jax.jit
def find_A_NL_QCD(log10fPBH, Delta, ks):
    Am = -2.5
    AM = -1.0
    for _ in range(20):
        val = f_PBH_NL_QCD(10.0 ** ((Am + AM) / 2), Delta, ks)
        AM = jnp.where(val >= 10.0**log10fPBH, (Am + AM) / 2, AM)
        Am = jnp.where(val >= 10.0**log10fPBH, Am, (Am + AM) / 2)

    return (Am + AM) / 2
