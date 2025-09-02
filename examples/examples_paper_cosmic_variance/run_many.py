# Global imports
import sys
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from scipy.stats import norm

# Local imports
sys.path.append("../")


# Local imports
import examples_utils as eu  # noqa: E402
from fastPTA import get_tensors as gt  # noqa: E402
from fastPTA import utils as ut  # noqa: E402
from fastPTA.angular_decomposition import (  # noqa: E402
    spherical_harmonics as sph,  # noqa: E402
)  # noqa: E402
from fastPTA.data import data_correlations as dc  # noqa: E402
from fastPTA.data import generate_data as gd  # noqa: E402
from fastPTA.inference_tools import iterative_estimation as ie  # noqa: E402

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

# Print the path to the file
print(eu.path_to_file)

# Choose whether to add dipole and quadrupole contributions
add_dipole = False
add_quadropole = True

# Number of runs
N_runs = 10

# Number of pulsars you want to run over
NN_pulsars = [70, 120, 200, 500]

# Observation time in years
Tspan_yrs = 16.0

# Observation time in seconds
Tspan = Tspan_yrs * ut.yr

# Number of external frequency bins
nfreqs = 15

# Delta frequency for the internal frequencies
df_step = 0.1

# Minimal value for the internal frequencies
f_in = 0.5

# External frequency vector
fi = jnp.arange(1, nfreqs + 1) / Tspan

# Internal frequency vector
ff = jnp.arange(f_in, nfreqs + 1, step=df_step) / Tspan

# Spectral density for the external frequencies
S_fi = (fi / ut.f_yr) ** -7.0 / 3.0

# Spectral density for the internal frequencies
S_ff = (ff / ut.f_yr) ** -7.0 / 3.0

# HEALPix resolution parameter
Nside = 12

# Maximum multipole moment
l_max = 3

# Initialize the CLs
n_params = sph.get_n_coefficients_real(l_max)
clms_real = np.zeros(n_params)
clms_real[0] = 1 / np.sqrt(4 * np.pi)

if add_dipole:
    clms_real[2] = 1 / np.sqrt(4 * np.pi) / np.sqrt(3)
if add_quadropole:
    clms_real[6] = 1 / np.sqrt(5 * np.pi)

# Generate the HEALPix map
Pk = sph.get_map_from_real_clms(clms_real, Nside, l_max=l_max)

# Full spectrum (angular times frequency)
H_p_ff = Pk[:, None] * S_ff[None, :]

# Frequency-frequency normalization
C_ff = dc.get_D_IJ_fifj_normalization(Tspan, fi, ff, H_p_ff)

# Inverse of the frequency-frequency normalization
inv_ff = ut.compute_inverse(C_ff)

# Since the frequency structure is diagonalized, just rescale with the right
# normalization
f0 = fi**0
S_f0 = 3 / 4 * (2 * jnp.pi) ** 2 * S_fi**0

# Initial guess (quite far away from the true values)
full_guess = np.zeros(n_params)
full_guess[0] = np.sqrt(4 * np.pi)
full_guess *= 10


for npulsars in NN_pulsars:
    path = "generated_data/ff_"
    if add_dipole:
        path += "dipole_"
    if add_quadropole:
        path += "quadrupole_"

    outname = (
        path
        + str(N_runs)
        + "_"
        + str(npulsars)
        + "_"
        + str(Nside)
        + "_"
        + str(nfreqs)
        + ".npz"
    )
    print("Will store results in:", outname)

    # Initialize arrays to store results
    means = np.zeros((N_runs, n_params))
    stds = np.zeros((N_runs, n_params))
    cdfs = np.zeros((N_runs, n_params))

    # Generate pulsar sky positions and distances
    p_vec, cos_IJ, distances, theta_k, phi_k = (
        gd.generate_pulsar_sky_and_kpixels(npulsars, Nside)
    )

    # Convert distance from pc to meters
    distances *= ut.parsec

    # Get the gamma_IJ_lm for the pulsar catalog
    gamma_IJ_lm = gt.get_correlations_lm_IJ(p_vec, l_max, Nside)

    for nn in tqdm.tqdm(range(N_runs)):
        h_tilde = gd.generate_hpc_polarization_pixel_frequency(H_p_ff)

        D_IJ = dc.get_D_IJ_fifj(
            Tspan, fi, ff, h_tilde, distances, p_vec, theta_k, phi_k
        )

        rescale_D_IJ = jnp.diagonal(
            jnp.einsum("fg,glab->flab", inv_ff, D_IJ), axis1=0, axis2=1
        ).T

        # Get the estimated values for the parameters
        theta, uncertainties, _ = ie.iterative_estimation(
            ie.get_update_estimate_diagonal,
            full_guess,
            rescale_D_IJ,
            gamma_IJ_lm,
            f0,
            S_f0,
        )

        means[nn] = theta
        stds[nn] = uncertainties
        cdfs[nn] = norm.cdf((clms_real - theta) / stds[nn])

    np.savez(
        outname,
        means=means,
        stds=stds,
        cdfs=cdfs,
        p_vec=p_vec,
        cos_IJ=cos_IJ,
    )
