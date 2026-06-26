"""Shared utilities for the paper-figure notebooks.

Covers:
  - PTA + response + whitening matrix construction (real PTA response and
    synthetic patch responses)
  - toy single-bin source generation (uniform / exponential)
  - astrophysical single-bin sky-map generation using fastropop
  - KS-test wrappers (known-scale and estimated-scale variants)
  - the full rejection-fraction sweeps used by Fig 2
  - amplitude-distribution sample generation for Fig 3
  - plotting rcParams that gracefully degrade if LaTeX is unavailable

The SMBHB population code uses `fastropop` rather than the in-repo
`semi_analytic_populations_jax` module.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import kstest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import fastPTA.data.generate_data as gd
from fastPTA import utils as ut
from fastPTA.data import datastream as gds
from fastPTA.get_tensors import HD_correlations

import fastropop as frp

from .kolmogorov_smirnov import KolmogorovSmirnovBootstrapCache


# =========================================================================
# Plotting rcParams (usetex with mathtext fallback)
# =========================================================================

def set_paper_rcparams(width=None):
    """Apply the paper's matplotlib rcParams.

    Tries to enable LaTeX rendering (`text.usetex=True`). If a working
    LaTeX install is not detected, falls back to matplotlib's built-in
    mathtext renderer so the notebooks still produce a sensible figure.
    """
    import matplotlib.pyplot as plt

    COLUMN_WIDTH_PT = 246.0
    TEXT_WIDTH_PT = 510.0
    pt_to_inch = 1.0 / 72.27

    use_tex = (shutil.which("latex") is not None and
               shutil.which("dvipng") is not None)
    common = {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.titlesize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.5,
    }
    if use_tex:
        common["text.usetex"] = True
        common["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        common["text.usetex"] = False
        common["mathtext.fontset"] = "cm"
        print("[pta_helpers] LaTeX not found - using mathtext fallback.")
    plt.rcParams.update(common)
    return use_tex, COLUMN_WIDTH_PT * pt_to_inch, TEXT_WIDTH_PT * pt_to_inch


# =========================================================================
# PTA + real response
# =========================================================================

def get_R_without_pulsar_term(f_vec, distances, p_vec, theta_k, phi_k):
    """Compute the Hellings-Downs correlation matrix from the response-integral
    R_p, R_c matrices at one frequency."""
    # Compute the unit vector in the direction of the pixels
    k_vec = gds.unit_vector(theta_k, phi_k)

    # Get the plus and cross polarization tensors for the pixel directions
    e_p_k, e_c_k = gds.get_plus_cross(theta_k, phi_k)

    # Get the plus and cross pattern functions, the shapes are (..., n_pixels)
    F_p, F_c = gds.get_F_pc(p_vec, k_vec, e_p_k, e_c_k)

    # Compute the dot product in the exponent
    one_plus = 1.0 + jnp.einsum("pi,...i->...p", k_vec, p_vec)

    # Compute the factor (1 - exponential) in the response function
    exponential = 1. - 0. * jnp.exp(
        -2.0j
        * jnp.pi
        * jnp.einsum(
            "f,...,...p->...pf", f_vec, distances / ut.light_speed, one_plus
        )    )

    # Compute the response function for the plus and cross polarizations
    # The shapes of R_p_f and R_c_f will be (..., n_pixels, n_frequencies)
    R_p_f = jnp.einsum(
        "...,...f,f->...f", F_p, exponential, 1.0 / (2j * jnp.pi * f_vec)
    )
    R_c_f = jnp.einsum(
        "...,...f,f->...f", F_c, exponential, 1.0 / (2j * jnp.pi * f_vec)
    )

    return R_p_f, R_c_f

def setup_pta(npulsars, nside, fcenter, integrate_HD_numerically=False):
    """Generate pulsar positions, the polarized response, and the
    Gamma-based whitening matrix at one frequency.

    Returns a dict with: p_vec, theta_k, phi_k, r_p, r_c, gamma, w_mat,
    npix.
    """
    npix = hp.nside2npix(nside)
    p_vec, cos_IJ, distances, theta_k, phi_k = gd.generate_pulsar_sky_and_kpixels(
        npulsars, Nside=nside
    )
    distances = np.array(distances) * ut.parsec

    r_p, r_c = gds.get_R_pc(
        jnp.asarray(fcenter), distances, p_vec, theta_k, phi_k
    )
    r_p = np.array(r_p[..., 0])
    r_c = np.array(r_c[..., 0])
    gamma_R = (r_p @ r_p.conj().T + r_c @ r_c.conj().T) / npix
    gamma_R = 0.5 * (gamma_R + gamma_R.conj().T)
    w_mat_R = build_whitening_matrix(gamma_R)

    if integrate_HD_numerically:
        r_p_hd, r_c_hd = get_R_without_pulsar_term(
            jnp.asarray(fcenter), distances, p_vec, theta_k, phi_k)
        r_p_hd = np.array(r_p_hd[..., 0])
        r_c_hd = np.array(r_c_hd[..., 0])
        gamma_HD = (r_p_hd @ r_p_hd.conj().T + r_c_hd @ r_c_hd.conj().T) / npix
        gamma_HD = 0.5 * (gamma_HD + gamma_HD.conj().T) + 0.5 * (2/3)/(2*np.pi*fcenter[0])**2 * np.eye(npulsars)
    else:
        gamma_HD = (np.asarray(HD_correlations(cos_IJ)) +
                    0.5 * np.eye(npulsars)) * (2.0 / 3.0) / (2 * np.pi * fcenter[0]) ** 2
    w_mat_HD = build_whitening_matrix(gamma_HD)
    return {
        "npix": npix,
        "p_vec": np.array(p_vec),
        "theta_k": np.array(theta_k),
        "phi_k": np.array(phi_k),
        "r_p": r_p,
        "r_c": r_c,
        "gamma_R": gamma_R,
        "w_mat_R": w_mat_R,
        "gamma_HD": gamma_HD,
        "w_mat_HD": w_mat_HD,
    }


def build_whitening_matrix(gamma, eps=1e-15):
    """Whitening matrix W = Lambda^{-1/2} M^dagger from the
    eigendecomposition of the Hermitian covariance Gamma.
    """
    e_vals, e_vecs = np.linalg.eigh(gamma)
    e_vals = np.maximum(e_vals, eps)
    return np.einsum("i,il->il", 1.0 / np.sqrt(e_vals), e_vecs.conj().T)


def build_patch_response_matrices(p_vec, angular_patch, nside, rng):
    """Synthetic top-hat patch response of half-angle `angular_patch`
    (radians) centred on each pulsar's line of sight, with independent
    random phases per pixel for the two polarizations.

    Returns (R_p_patch, R_c_patch, W_patch) all in numpy.
    """
    npix = hp.nside2npix(nside)
    pk = np.zeros((len(p_vec), npix), dtype=float)
    for i in range(len(p_vec)):
        disc = hp.query_disc(nside=nside, vec=np.asarray(p_vec[i]),
                             radius=float(angular_patch))
        pk[i, disc] = 1.0

    phase_p = rng.uniform(0.0, 2 * np.pi, size=(len(p_vec), npix))
    phase_c = rng.uniform(0.0, 2 * np.pi, size=(len(p_vec), npix))
    r_p_patch = pk * np.exp(1j * phase_p)
    r_c_patch = pk * np.exp(1j * phase_c)

    gamma_patch = (r_p_patch @ r_p_patch.conj().T +
                   r_c_patch @ r_c_patch.conj().T) / npix
    gamma_patch = 0.5 * (gamma_patch + gamma_patch.conj().T)
    w_patch = build_whitening_matrix(gamma_patch)
    return r_p_patch, r_c_patch, w_patch


# =========================================================================
# Toy single-bin generation (uniform / exponential)
# =========================================================================

def generate_toy_pixel_map(rng, npix, nsources, dist_name,
                           halfwidth=10.0, scale_exp=10.0):
    """Place `nsources` discrete sources on randomly chosen pixels with
    complex amplitudes drawn from a uniform or exponential distribution
    (both normalized to unit variance per complex sample).
    """
    chosen_pixels = rng.choice(npix, size=(nsources,), replace=True)
    if dist_name == "uniform":
        vals = (
            rng.uniform(-halfwidth, halfwidth, size=(2, nsources))
            + 1j * rng.uniform(-halfwidth, halfwidth, size=(2, nsources))
        ) * np.sqrt(6.0 / (2 * halfwidth) ** 2)
    elif dist_name == "exponential":
        vals = (
            (rng.exponential(scale=scale_exp, size=(2, nsources)) / scale_exp - 1.0)
            + 1j * (rng.exponential(scale=scale_exp, size=(2, nsources)) / scale_exp - 1.0)
        ) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unknown toy distribution: {dist_name}")

    d_pixel = np.empty((2, npix), dtype=np.complex128)
    for pol in (0, 1):
        real_part = np.bincount(chosen_pixels, weights=vals[pol].real, minlength=npix)
        imag_part = np.bincount(chosen_pixels, weights=vals[pol].imag, minlength=npix)
        d_pixel[pol] = real_part + 1j * imag_part
    return d_pixel


def whitened_powers_from_pixels_R(d_pixel, pta_setup):
    """|w_I|^2 for a pixel-space toy source draw."""
    w_vec = pta_setup["w_mat_R"] @ (
        pta_setup["r_p"] @ d_pixel[0] + pta_setup["r_c"] @ d_pixel[1]
    )
    return np.abs(w_vec) ** 2

def whitened_powers_from_pixels_HD(d_pixel, pta_setup):
    """|w_I|^2 for a pixel-space toy source draw, using the HD-based whitening."""
    w_vec = pta_setup["w_mat_HD"] @ (
        pta_setup["r_p"] @ d_pixel[0] + pta_setup["r_c"] @ d_pixel[1]
    )
    return np.abs(w_vec) ** 2


# =========================================================================
# Astrophysical SMBHB single-bin generation (uses fastropop)
# =========================================================================

def make_lown0_population(nbins=14, target_bin_idx=None):
    """The lown0 SMBHB population used in the paper.

    Wraps `fastropop.SemiAnalyticPopulation` with the paper's fiducial
    parameters and selects the highest of `nbins` log-spaced NG15-style
    frequency bins by default.
    """
    if target_bin_idx is None:
        target_bin_idx = nbins - 1
    bin_edges = np.array([
        [(2 * i + 1) * frp.fminNG15 * frp.s, (2 * i + 3) * frp.fminNG15 * frp.s]
        for i in range(nbins)
    ])
    bin_centers = np.array([
        0.5 * (bin_edges[i][0] + bin_edges[i][1]) for i in range(nbins)
    ])
    fbounds = bin_edges[target_bin_idx]
    fcenter = np.array([bin_centers[target_bin_idx]])

    population_params = {
        "n0": 1e-8 / ((1e6 * frp.pc * frp.pcinMKS) ** 3 *
                      (1e9 * frp.yr * frp.yrinMKS)),
        "alphaM": 0.0,
        "Mstar": 1.8e8 * frp.MsunMKS,
        "betaz": 2.0,
        "z0": 1.8,
    }
    integration_limits = {"fbounds": fbounds}
    sampling_grids = {"fgrid": np.geomspace(fbounds[0], fbounds[1], 3000)}
    pta_params = {"fmin": fcenter / frp.s,
                  "fmax": fcenter / frp.s,
                  "Nfreqs": 1}

    pop = frp.SemiAnalyticPopulation(
        population_params=population_params,
        integration_limits=integration_limits,
        sampling_grids=sampling_grids,
        PTA_params=pta_params,
    )
    return pop, fcenter


def generate_single_bin_skymap(pop, nbinaries, nside, rng):
    """Draw a discrete SMBHB realization with the given count using
    fastropop's `generate_skymaps`, and return the per-pixel complex
    strain map for the single frequency bin.

    The returned array has shape ``(2, npix)`` (plus/cross polarizations
    along axis 0), i.e. the ``(2, npix, 1)`` skymap with the trivial
    frequency axis dropped, so it can be fed directly to
    ``whitened_powers_from_pixels_R`` / ``whitened_powers_from_pixels_HD``
    exactly like a toy pixel map.
    """
    pta_frequencies = np.asarray(pop.PTA_frequencies).ravel()
    seed = int(rng.integers(0, 2**31 - 1))
    skymaps_tot, _, _ = pop.generate_skymaps(
        Nbinaries=int(nbinaries),
        PTA_frequencies=pta_frequencies,
        Nside=nside,
        key=seed,
    )
    skymaps_tot = np.asarray(skymaps_tot)  # (2, npix, n_freq=1)
    return skymaps_tot[:, :, 0]            # (2, npix)


# =========================================================================
# KS test wrappers
# =========================================================================

def naive_ks_pvalue_known_scale(w_sq, expected_scale, df=2):
    """KS test of `w_sq` against chi2(df, scale=expected_scale)."""
    return float(kstest(w_sq, "chi2",
                        args=(df, 0.0, expected_scale)).pvalue)


def naive_ks_pvalue_estimated_scale(w_sq, df=2):
    """KS test of `w_sq` against chi2(df) with the scale estimated from
    the same sample. *Anti-conservative* under H0 (Lilliefors problem);
    use bootstrap calibration for valid p-values.
    """
    estimated_scale = float(np.mean(w_sq)) / df
    return float(kstest(w_sq, "chi2",
                        args=(df, 0.0, estimated_scale)).pvalue)


# =========================================================================
# Rejection sweeps used by Fig 2
# =========================================================================

def run_toy_subset(npulsars_list, nsources_list, nrealizations, nside,
                   fcenter, n_bootstrap=20000, seed=0):
    """Sweep over (Np, Ns) for the uniform and exponential toy
    populations, whitening with BOTH the response-integral Gamma_R and the
    analytic Hellings-Downs Gamma_HD. For each whitening we record:

      (a) ks_naive_R_{dist}  : naive KS, R whitening, fixed (known) scale
      (b) ks_naive_HD_{dist} : naive KS, HD whitening, fixed (known) scale
      (c) ks_boot_R_{dist}   : bootstrap KS, R whitening, scale estimated
      (d) ks_boot_HD_{dist}  : bootstrap KS, HD whitening, scale estimated

    The fixed scale for the naive tests is the theoretically-expected
    Ns/2 (unit-variance complex amplitudes, both whitenings normalized to
    the same trace, so the expected per-mode mean is Ns for either one).
    Returns a dict suitable for np.savez.
    """
    result = {
        "npulsars_list": np.array(npulsars_list, dtype=int),
        "nsources_list": np.array(nsources_list, dtype=int),
        "nrealizations": int(nrealizations),
        "nside": int(nside),
        "n_bootstrap": int(n_bootstrap),
    }
    arr_keys = []
    for dist_name in ["uniform", "exponential"]:
        for tag in ("ks_naive_R", "ks_naive_HD", "ks_boot_R", "ks_boot_HD"):
            key = f"{tag}_{dist_name}"
            arr_keys.append(key)
            result[key] = np.zeros(
                (len(npulsars_list), len(nsources_list), nrealizations)
            )

    rng = np.random.default_rng(seed)
    for i_np, npulsars in enumerate(npulsars_list):
        print(f"  Toy: Np = {npulsars}")
        ks_cache = KolmogorovSmirnovBootstrapCache(
            n=npulsars, df=2, n_bootstrap=n_bootstrap
        )
        pta = setup_pta(npulsars=npulsars, nside=nside, fcenter=fcenter)
        for i_ns, nsources in enumerate(nsources_list):
            print(f"    Ns = {nsources}")
            expected_scale = nsources / 2.0
            for j in range(nrealizations):
                for dist_name in ["uniform", "exponential"]:
                    d_pixel = generate_toy_pixel_map(
                        rng, pta["npix"], nsources, dist_name
                    )
                    w_sq_R = whitened_powers_from_pixels_R(d_pixel, pta)
                    w_sq_HD = whitened_powers_from_pixels_HD(d_pixel, pta)
                    # (a) naive R, fixed scale
                    result[f"ks_naive_R_{dist_name}"][i_np, i_ns, j] = (
                        naive_ks_pvalue_known_scale(w_sq_R, expected_scale)
                    )
                    # (b) naive HD, fixed scale
                    result[f"ks_naive_HD_{dist_name}"][i_np, i_ns, j] = (
                        naive_ks_pvalue_known_scale(w_sq_HD, expected_scale)
                    )
                    # (c) bootstrap R, scale estimated
                    _, p_boot_R = ks_cache.compute_test(w_sq_R, interpolate=False)
                    result[f"ks_boot_R_{dist_name}"][i_np, i_ns, j] = p_boot_R
                    # (d) bootstrap HD, scale estimated
                    _, p_boot_HD = ks_cache.compute_test(w_sq_HD, interpolate=False)
                    result[f"ks_boot_HD_{dist_name}"][i_np, i_ns, j] = p_boot_HD
    return result


def run_astro_subset(npulsars_list, nsources_list, nrealizations, nside,
                     n_bootstrap=20000, seed=0):
    """Same sweep but for the lown0 SMBHB population, generated with
    fastropop's `generate_skymaps` and fed through the *pixel* whitening
    functions (equivalent to the toy path, just with a physical sky map
    instead of a synthetic one). Whitens with BOTH the response-integral
    Gamma_R and the analytic Hellings-Downs Gamma_HD, recording:

      (a) ks_naive_R_astro  : naive KS, R whitening, fixed (known) scale
      (b) ks_naive_HD_astro : naive KS, HD whitening, fixed (known) scale
      (c) ks_boot_R_astro   : bootstrap KS, R whitening, scale estimated
      (d) ks_boot_HD_astro  : bootstrap KS, HD whitening, scale estimated

    Because the astrophysical strain amplitudes carry no a-priori unit
    normalization, the "fixed scale" for the naive tests is the ensemble
    mean of |w|^2 over all realizations at fixed (Np, Ns), computed
    separately for each whitening. These are stored in
    `fixed_scale_R_astro` / `fixed_scale_HD_astro` for reference.
    """
    pop, fcenter = make_lown0_population()
    result = {
        "npulsars_list": np.array(npulsars_list, dtype=int),
        "nsources_list": np.array(nsources_list, dtype=int),
        "nrealizations": int(nrealizations),
        "nside": int(nside),
        "n_bootstrap": int(n_bootstrap),
        "population_name": "lown0",
    }
    for k in ("ks_naive_R_astro", "ks_naive_HD_astro",
              "ks_boot_R_astro", "ks_boot_HD_astro"):
        result[k] = np.zeros(
            (len(npulsars_list), len(nsources_list), nrealizations)
        )
    for k in ("fixed_scale_R_astro", "fixed_scale_HD_astro"):
        result[k] = np.zeros((len(npulsars_list), len(nsources_list)))

    rng = np.random.default_rng(seed)
    for i_np, npulsars in enumerate(npulsars_list):
        print(f"  Astro: Np = {npulsars}")
        ks_cache = KolmogorovSmirnovBootstrapCache(
            n=npulsars, df=2, n_bootstrap=n_bootstrap
        )
        pta = setup_pta(npulsars=npulsars, nside=nside, fcenter=fcenter)
        for i_ns, nsources in enumerate(nsources_list):
            print(f"    Ns = {nsources}")
            w_sq_all_R = np.zeros((nrealizations, npulsars))
            w_sq_all_HD = np.zeros((nrealizations, npulsars))
            for j in range(nrealizations):
                d_pixel = generate_single_bin_skymap(
                    pop=pop, nbinaries=int(nsources), nside=nside, rng=rng
                )
                w_sq_R = whitened_powers_from_pixels_R(d_pixel, pta)
                w_sq_HD = whitened_powers_from_pixels_HD(d_pixel, pta)
                w_sq_all_R[j] = w_sq_R
                w_sq_all_HD[j] = w_sq_HD
                # (c) bootstrap R, scale estimated
                _, p_boot_R = ks_cache.compute_test(w_sq_R, interpolate=False)
                result["ks_boot_R_astro"][i_np, i_ns, j] = p_boot_R
                # (d) bootstrap HD, scale estimated
                _, p_boot_HD = ks_cache.compute_test(w_sq_HD, interpolate=False)
                result["ks_boot_HD_astro"][i_np, i_ns, j] = p_boot_HD
            # Fixed (known) scale = ensemble mean variance / 2, per whitening.
            fixed_scale_R = float(np.mean(w_sq_all_R) / 2.0)
            fixed_scale_HD = float(np.mean(w_sq_all_HD) / 2.0)
            result["fixed_scale_R_astro"][i_np, i_ns] = fixed_scale_R
            result["fixed_scale_HD_astro"][i_np, i_ns] = fixed_scale_HD
            for j in range(nrealizations):
                # (a) naive R, fixed scale
                result["ks_naive_R_astro"][i_np, i_ns, j] = (
                    naive_ks_pvalue_known_scale(w_sq_all_R[j], fixed_scale_R)
                )
                # (b) naive HD, fixed scale
                result["ks_naive_HD_astro"][i_np, i_ns, j] = (
                    naive_ks_pvalue_known_scale(w_sq_all_HD[j], fixed_scale_HD)
                )
    return result


# =========================================================================
# Fig 3 helpers: source-pixel selection and amplitude-distribution samples
# =========================================================================

def pick_source_pixels_real(r_p, r_c, ns, nside, min_sep_deg=30):
    """Pick `ns` source pixels that maximize total PTA response power,
    well separated on the sky.
    """
    total_power = np.sum(np.abs(r_p) ** 2 + np.abs(r_c) ** 2, axis=0)
    chosen = []
    available = np.ones(total_power.shape[0], dtype=bool)
    cos_min_sep = np.cos(np.radians(min_sep_deg))
    for _ in range(ns):
        powers = total_power.copy()
        powers[~available] = -1
        pix = int(np.argmax(powers))
        chosen.append(pix)
        vec_chosen = np.array(hp.pix2vec(nside, pix))
        for p in range(len(available)):
            vec_p = np.array(hp.pix2vec(nside, p))
            if np.dot(vec_chosen, vec_p) > cos_min_sep:
                available[p] = False
    return chosen


def pick_source_pixels_patch(p_vec, ns, nside, min_sep_deg=20):
    """Pick `ns` pixels close to pulsar positions and well-separated.
    Guarantees that each source falls inside at least one pulsar's
    patch cone.
    """
    chosen = []
    used_pulsars = set()
    cos_min_sep = np.cos(np.radians(min_sep_deg))
    for i, pvec in enumerate(p_vec):
        if i in used_pulsars:
            continue
        pix = hp.vec2pix(nside, *pvec)
        too_close = False
        vec_candidate = np.array(hp.pix2vec(nside, pix))
        for prev_pix in chosen:
            vec_prev = np.array(hp.pix2vec(nside, prev_pix))
            if np.dot(vec_candidate, vec_prev) > cos_min_sep:
                too_close = True
                break
        if not too_close:
            chosen.append(int(pix))
            used_pulsars.add(i)
        if len(chosen) >= ns:
            break
    if len(chosen) < ns:
        raise RuntimeError(
            f"Could only find {len(chosen)} well-separated patch source "
            f"pixels, need {ns}. Try reducing min_sep_deg."
        )
    return chosen


def gen_h0_uniform(rng, n):
    return rng.uniform(0, 2, n)


def gen_h0_exp(rng, n):
    return rng.exponential(1.0, n)


def gen_h0_lognorm(rng, n, sigma=1.2):
    mu = -0.5 * sigma ** 2
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def default_amp_dists():
    """The three amplitude distributions used for Fig 3."""
    return [
        ("uniform", gen_h0_uniform),
        ("exp", gen_h0_exp),
        ("lognorm", gen_h0_lognorm),
    ]


def generate_samples_multisrc(resp_vectors, amp_dists, n_real, rng_seed):
    """Generate |w_I|^2 samples for a set of complex per-source whitened
    response vectors and a list of amplitude distributions.

    Returns (raw_sorted_by_name, normalized_sorted_by_name) where the
    raw entries are |w_I|^2 and the normalized entries are
    |w_I|^2 / mean(|w_I|^2).
    """
    ns = len(resp_vectors)
    n_p = len(resp_vectors[0])
    raw = {}
    norm = {}
    for idx, (name, gen_fn) in enumerate(amp_dists):
        rng_local = np.random.default_rng(rng_seed + idx)
        raw_all = np.empty(n_real * n_p)
        norm_all = np.empty(n_real * n_p)
        for i in range(n_real):
            w = np.zeros(n_p, dtype=complex)
            for s in range(ns):
                h0 = gen_fn(rng_local, 1)[0]
                phi0 = rng_local.uniform(0, 2 * np.pi)
                w += resp_vectors[s] * h0 * np.exp(1j * phi0)
            w_sq = np.abs(w) ** 2
            mean_w_sq = float(np.mean(w_sq))
            raw_all[i * n_p:(i + 1) * n_p] = w_sq
            if mean_w_sq > 0:
                norm_all[i * n_p:(i + 1) * n_p] = w_sq / mean_w_sq
            else:
                norm_all[i * n_p:(i + 1) * n_p] = 0.0
        raw[name] = np.sort(raw_all)
        norm[name] = np.sort(norm_all)
    return raw, norm


def generate_samples_multisrc_linear(resp_vectors, amp_dists, n_real,
                                     rng_seed):
    """Like `generate_samples_multisrc` but returns the *linear* whitened
    data: stacked (Re(w_I), Im(w_I)), 2*Np samples per realization.

    raw[name]: stacked Re/Im, unnormalized.
    norm[name]: same, divided by the per-realization empirical standard
                deviation so the standardized sample has unit variance.

    Under H0 the marginal of the stacked Re/Im across pulsars is
    Gaussian; under H1 the *shape* stays approximately Gaussian (broad
    response + CLT in mode index) but the *scale* tracks the source
    amplitude distribution. Dividing by the empirical std removes the
    scale, so all amplitude distributions collapse onto N(0, 1)
    independently of which source distribution generated them - that is
    the demonstration of amplitude-blindness.
    """
    ns = len(resp_vectors)
    n_p = len(resp_vectors[0])
    block = 2 * n_p
    raw = {}
    norm = {}
    for idx, (name, gen_fn) in enumerate(amp_dists):
        rng_local = np.random.default_rng(rng_seed + idx)
        raw_all = np.empty(n_real * block)
        norm_all = np.empty(n_real * block)
        for i in range(n_real):
            w = np.zeros(n_p, dtype=complex)
            for s in range(ns):
                h0 = gen_fn(rng_local, 1)[0]
                phi0 = rng_local.uniform(0, 2 * np.pi)
                w += resp_vectors[s] * h0 * np.exp(1j * phi0)
            stacked = np.concatenate([w.real, w.imag])
            raw_all[i * block:(i + 1) * block] = stacked
            std = float(np.std(stacked))
            if std > 0:
                norm_all[i * block:(i + 1) * block] = stacked / std
            else:
                norm_all[i * block:(i + 1) * block] = 0.0
        raw[name] = np.sort(raw_all)
        norm[name] = np.sort(norm_all)
    return raw, norm
