# Global imports
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

# Local imports
import fastPTA.utils as ut
from fastPTA import generate_new_pulsar_configuration as gnpc
from fastPTA.data import data_correlations as dc

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)


def generate_gaussian(mean, sigma, size=None):
    """
    This function generates complex data by sampling real and imaginary parts
    independently from a Gaussian distribution with the specified mean and
    standard deviation.

    Parameters:
    -----------
    mean : Array
        Mean of the Gaussian distribution.
    sigma : Array
        Standard deviation of the Gaussian distribution.

    Returns:
    --------
    data : Array
        Complex data sampled from a Gaussian distribution.
    """

    # Generate the real part
    real = np.random.normal(loc=mean, scale=sigma, size=size)

    # Then the imaginary part
    imaginary = np.random.normal(loc=mean, scale=sigma, size=size)

    # Return the sum
    return (real + 1j * imaginary) / np.sqrt(2)


def generate_pulsar_sky_and_kpixels(
    Np,
    Nside,
    log10_loc_pc=3.1356587021094077,
    log10_scale_pc=0.2579495260515389,
):
    """
    Generate pulsar sky positions and k-pixels for a given number of pulsars.

    Parameters:
    -----------
    Np : int
        Number of pulsars to generate.
    Nside : int
        HEALPix Nside parameter for pixelization.
    log10_loc_pc : float, optional
        Mean of the normal distribution for log10 distances in parsecs.
        Defaults to 3.1356587021094077.
    log10_scale_pc : float, optional
        Std of the normal distribution for log10 distances in parsecs.
        Defaults to 0.2579495260515389.

    Returns:
    --------
    Npix : int
        Total number of pixels in the HEALPix map.
    p_vec : Array
        Array of pulsar sky positions in Cartesian coordinates.
    cos_IJ : Array
        Cosine of the angle between pulsar pairs.
    distance : Array
        Array of distances for the pulsars in parsecs.
    theta_k : Array
        Polar angles of the k-pixels.
    phi_k : Array
        Azimuthal angles of the k-pixels.
    """

    # Get the number of pixels in the HEALPix map
    Npix = hp.nside2npix(Nside)

    # Generate the k-pixels theta_k and phi_k, polar and azimuthal angles
    theta_k, phi_k = hp.pix2ang(Nside, np.arange(Npix))

    # Generate pulsar sky positions in Cartesian coordinates
    theta = jnp.arccos(np.random.uniform(-1.0, 1.0, Np))
    phi = jnp.array(np.random.uniform(0.0, 2.0 * jnp.pi, Np))

    # Convert spherical coordinates to Cartesian coordinates
    p_vec = jnp.array(hp.ang2vec(theta, phi))

    # Calculate the cosine of the angle between pulsar pairs
    cos_IJ = jnp.dot(p_vec, p_vec.T)

    # Ensure cos_IJ is within the valid range for a dot product
    cos_IJ = jnp.clip(cos_IJ, -1.0, 1.0)

    # Generate distances for the pulsars in parsecs
    distance = gnpc.generate_distance(
        Np, loc=log10_loc_pc, scale=log10_scale_pc
    )

    return p_vec, cos_IJ, distance, theta_k, phi_k


def generate_hpc_polarization_pixel_frequency(H_p_ff):
    """
    Generate the Gaussian data for the GW signal in pixel and frequency space.

    Parameters:
    -----------
    H_p_ff : Array
        The input data for the GW signal in pixel and frequency space.
        The shape of H_p_ff is (Npix, nfrequencies), where Npix is the number
        of pixels and nfrequencies is the number of frequency bins.

    Returns:
    --------
    h_tilde : Array
        The generated Gaussian data for the GW signal.
        The shape of h_tilde is (2, Npix, nfrequencies), where the first
        dimension corresponds to the two polarization (plus and cross).
    """

    # Number of pixels
    Npix = H_p_ff.shape[0]

    # Number of frequency bins
    nfrequencies = H_p_ff.shape[1]

    # Area of a pixel in steradians
    dOmega = 4 * jnp.pi / Npix

    # Std of the Gaussian distribution for the signal normalized to the pixels
    sigma = jnp.sqrt(H_p_ff * dOmega)

    # Generate the Gaussian data (h_tilde is the GW signal, 2 polarizations)
    h_tilde = generate_gaussian(0.0, sigma, size=(2, Npix, nfrequencies))

    return h_tilde


def generate_D_IJ(
    Nside,
    Np,
    fi,
    H_p_fi,
    log10_loc_pc=3.1356587021094077,
    log10_scale_pc=0.2579495260515389,
):
    """
    Generate the pulsar-pulsar correlation matrix D_IJ for a given set of
    pulsars and external (measured) frequencies (fi).

    Parameters:
    -----------
    Nside : int
        HEALPix Nside parameter for pixelization.
    Np : int
        Number of pulsars.
    fi : Array
        External (measured) frequencies.
    H_p_fi : Array
        The input data for the GW signal in pixel and frequency space.
        The shape of H_p_fi is (Npix, Nfi), where Npix is the number of pixels
        and Nfi is the number of external frequency (Nfi) bins.
    log10_loc_pc : float, optional
        Log10 of the location parameter for the distance distribution.
    log10_scale_pc : float, optional
        Log10 of the scale parameter for the distance distribution.

    Returns:
    --------
    p_vec : Array
        The pulsar sky positions in Cartesian coordinates.
    zeta_IJ : Array
        The angle between pulsar pairs.
    D_IJ : Array
        The pulsar-pulsar correlation matrix.
    """

    # Generate pulsar sky positions and distances
    p_vec, cos_IJ, distances, theta_k, phi_k = generate_pulsar_sky_and_kpixels(
        Np, Nside, log10_loc_pc=log10_loc_pc, log10_scale_pc=log10_scale_pc
    )

    # Calculate the angle between pulsar pairs
    zeta_IJ = jnp.arccos(cos_IJ)

    # Convert distance from pc to meters
    distances *= ut.parsec

    # Generate the Gaussian data (h_tilde is the GW signal)
    h_tilde = generate_hpc_polarization_pixel_frequency(H_p_fi)

    # Get the pulsar-pulsar correlation matrix D_IJ
    D_IJ = dc.get_D_IJ(
        fi,
        h_tilde,
        distances,
        p_vec,
        theta_k,
        phi_k,
    )

    return p_vec, zeta_IJ, D_IJ


def generate_D_IJ_fi(
    Nside,
    Np,
    Tspan,
    fi,
    ff,
    H_p_ff,
    log10_loc_pc=3.1356587021094077,
    log10_scale_pc=0.2579495260515389,
):
    """
    Generate the pulsar-pulsar correlation matrix D_IJ for a given set of
    pulsars and external (measured) frequencies (fi) after integrating over the
    sky pixels and a set of internal frequencies (ff).

    Parameters:
    -----------
    Nside : int
        HEALPix Nside parameter for pixelization.
    Np : int
        Number of pulsars.
    Tspan : float
        Time span for the observation (in seconds).
    fi : Array
        External (measured) frequencies.
    ff : Array
        Internal (simulated) frequencies.
    H_p_ff : Array
        The input data for the GW signal in pixel and frequency space.
        The shape of H_p_ff is (Npix, Nff), where Npix is the number of pixels
        and Nff is the number of internal frequency (Nff) bins.
    log10_loc_pc : float, optional
        Log10 of the location parameter for the distance distribution.
    log10_scale_pc : float, optional
        Log10 of the scale parameter for the distance distribution.

    Returns:
    --------
    p_vec : Array
        The pulsar sky positions in Cartesian coordinates.
    zeta_IJ : Array
        The angle between pulsar pairs.
    D_IJ : Array
        The pulsar-pulsar correlation matrix.
    """

    # Generate pulsar sky positions and distances
    p_vec, cos_IJ, distances, theta_k, phi_k = generate_pulsar_sky_and_kpixels(
        Np, Nside, log10_loc_pc=log10_loc_pc, log10_scale_pc=log10_scale_pc
    )

    # Calculate the angle between pulsar pairs
    zeta_IJ = jnp.arccos(cos_IJ)

    # Convert distance from pc to meters
    distances *= ut.parsec

    # Generate the Gaussian data (h_tilde is the GW signal)
    h_tilde = generate_hpc_polarization_pixel_frequency(H_p_ff)

    # Get the pulsar-pulsar correlation matrix D_IJ
    D_IJ = dc.get_D_IJ_fi(
        Tspan,
        fi,
        ff,
        h_tilde,
        distances,
        p_vec,
        theta_k,
        phi_k,
    )

    return p_vec, zeta_IJ, D_IJ


def generate_D_IJ_fifj(
    Nside,
    Np,
    Tspan,
    fi,
    ff,
    H_p_ff,
    log10_loc_pc=3.1356587021094077,
    log10_scale_pc=0.2579495260515389,
):
    """
    Generate the pulsar-pulsar correlation matrix D_IJ for a given set of
    pulsars and external (measured) frequencies (fi) after integrating over the
    sky pixels and a set of internal frequencies (ff).

    Parameters:
    -----------
    Nside : int
        HEALPix Nside parameter for pixelization.
    Np : int
        Number of pulsars.
    Tspan : float
        Time span for the observation (in seconds).
    fi : Array
        External (measured) frequencies.
    ff : Array
        Internal (simulated) frequencies.
    H_p_ff : Array
        The input data for the GW signal in pixel and frequency space.
        The shape of H_p_ff is (Npix, Nff), where Npix is the number of pixels
        and Nff is the number of internal frequency (Nff) bins.
    log10_loc_pc : float, optional
        Log10 of the location parameter for the distance distribution.
    log10_scale_pc : float, optional
        Log10 of the scale parameter for the distance distribution.

    Returns:
    --------
    p_vec : Array
        The pulsar sky positions in Cartesian coordinates.
    zeta_IJ : Array
        The angle between pulsar pairs.
    D_IJ : Array
        The pulsar-pulsar correlation matrix.
    """

    # Generate pulsar sky positions and distances
    p_vec, cos_IJ, distances, theta_k, phi_k = generate_pulsar_sky_and_kpixels(
        Np, Nside, log10_loc_pc=log10_loc_pc, log10_scale_pc=log10_scale_pc
    )

    # Calculate the angle between pulsar pairs
    zeta_IJ = jnp.arccos(cos_IJ)

    # Convert distance from pc to meters
    distances *= ut.parsec

    # Generate the Gaussian data (h_tilde is the GW signal)
    h_tilde = generate_hpc_polarization_pixel_frequency(H_p_ff)

    # Get the pulsar-pulsar correlation matrix D_IJ
    D_IJ = dc.get_D_IJ_fifj(
        Tspan,
        fi,
        ff,
        h_tilde,
        distances,
        p_vec,
        theta_k,
        phi_k,
    )

    return p_vec, zeta_IJ, D_IJ


def generate_MCMC_data(
    realization,
    frequency,
    signal_std,
    strain_omega,
    response_IJ,
    HD_functions_IJ,
    HD_coeffs,
    save_MCMC_data=True,
    path_to_MCMC_data="generated_data/MCMC_data.npz",
):
    """
    Generates (and might save) data for Markov Chain Monte Carlo (MCMC)
    analysis based on the provided parameters. If `realization` is True,
    it generates a realization of the data. Otherwise, it uses the expectation
    value. Data contain both signal and noise. If `save_MCMC_data` is True,
    the generated data is saved to a specified path.

    Parameters:
    -----------
    realization : bool
        Whether to generate a realization of the data.
    frequency : numpy.ndarray
        Array containing the frequency bins.
    signal_std : numpy.ndarray
        Array containing the standard deviation of the signal.
    strain_omega : numpy.ndarray
        Array containing the strain noise.
    response_IJ : numpy.ndarray
        Array containing the response function.
    HD_functions_IJ : numpy.ndarray
        Array containing the HD functions.
    HD_coeffs : numpy.ndarray
        Array containing the HD coefficients.
    save_MCMC_data : bool, optional
        Whether to save the generated data
        Default is True
    path_to_MCMC_data : str, optional
        Path to save the generated data
        Default is "generated_data/MCMC_data.npz"

    Returns:
    --------
    Tuple containing:
    - frequency: numpy.ndarray
        Array containing frequency bins.
    - data: numpy.ndarray
        Array containing the generated data.
    - response_IJ: numpy.ndarray
        Array containing response function.
    - strain_omega: numpy.ndarray
        Array containing strain noise.
    """

    if realization:
        # Use this part if you want data to be realized
        print("- Will generate a data realization\n")

        # Sqrt of pulsar-pulsar noise term
        noise_std = np.diagonal(np.sqrt(strain_omega), axis1=-2, axis2=-1)

        # Generate gaussian data with the right std for the noise
        noise_data = generate_gaussian(0, noise_std)

        # Combine to get nn take the real part only since the likelihood is
        # real and is multiplied by a 2 to keep track of that
        noise_part = np.real(
            noise_data[:, :, None] * np.conj(noise_data[:, None, :])
        )

        # Since we work with variables of f only (already integrated over theta
        # and phi), each signal component should be generated independently
        signal_std = signal_std[:, None, None] * np.ones_like(
            np.real(noise_part)
        )

        # Generate gaussian data with the right std for the signal
        signal_data = generate_gaussian(0, signal_std)

        # The response projects the signal in the different pulsar combinations
        # as for the noise, take the real part only
        signal_part = response_IJ * np.real(
            signal_data * np.conjugate(signal_data)
        )

    else:
        # Use this part if you don't want data to be realized
        print("- Data will use the expectation value\n")
        noise_part = strain_omega
        signal_part = response_IJ * (signal_std**2)[:, None, None]

    # Combine signal and noise
    data = signal_part + noise_part

    # Save the data
    if save_MCMC_data:
        np.savez(
            path_to_MCMC_data,
            frequency=frequency,
            data=data,
            response_IJ=response_IJ,
            strain_omega=strain_omega,
        )

    return frequency, data, response_IJ, strain_omega
