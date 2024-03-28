# Global
import time
import numpy as np
import emcee

# Setting the path to this file
import os, sys

file_path = os.path.dirname(__file__)
if file_path:
    file_path += "/"

sys.path.append(os.path.join(file_path, "../fastPTA/"))

# Local
from utils import *
from get_tensors import get_tensors
from signals import SMBBH_parameters, get_model

i_max_default = 100
R_convergence_default = 1e-1
R_criterion_default = "mean_squared"
burnin_steps_default = 300
MCMC_iteration_steps_default = 500


def generate_gaussian(mean, sigma):
    """
    This function generates complex data by sampling real and imaginary parts
    independently from a Gaussian distribution with the specified mean and
    standard deviation.

    Parameters:
    -----------
    mean : numpy.ndarray or jax.numpy.ndarray
        Mean of the Gaussian distribution.
    sigma : numpy.ndarray or jax.numpy.ndarray
        Standard deviation of the Gaussian distribution.

    Returns:
    --------
    data : numpy.ndarray or jax.numpy.ndarray
        Complex data sampled from a Gaussian distribution.

    """

    # Generate the real part
    real = np.random.normal(loc=mean, scale=sigma)

    # Then the imaginary part
    imaginary = np.random.normal(loc=mean, scale=sigma)

    # Return the sum
    return (real + 1j * imaginary) / np.sqrt(2)


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

        # Combine to get nn
        noise_part = noise_data[:, :, None] * np.conj(noise_data[:, None, :])

        # Generate gaussian data with the right std for the signal
        signal_data = generate_gaussian(0, signal_std)

        # The response projects the signal in the different pulsar combinations
        signal_part = (
            response_IJ * (signal_data * np.conj(signal_data))[:, None, None]
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


def get_MCMC_data(
    regenerate_MCMC_data,
    T_obs_yrs=10.33,
    n_frequencies=30,
    signal_label="power_law",
    signal_parameters=SMBBH_parameters,
    realization=True,
    save_MCMC_data=True,
    path_to_MCMC_data="generated_data/MCMC_data.npz",
    get_tensors_kwargs={},
    generate_catalog_kwargs={},
):
    """
    Loads or regenerate data for Markov Chain Monte Carlo (MCMC) analysis. If
    `regenerate_MCMC_data` is True or the MCMC data file is not found, the data
    is regenerated. Otherwise, it is loaded from the specified path. Data
    contain both signal and noise. If the data are generated, if `realization`
    is True, it generates a realization of the data. If `save_MCMC_data` is
    True, the generated data is saved to a specified path. Additional keyword
    arguments for get_tensors and generate_pulsars_catalog can be provided via
    get_tensors_kwargs and generate_catalog_kwargs.

    Parameters:
    -----------
    regenerate_MCMC_data : bool
        Whether to regenerate the MCMC data.
    T_obs_yrs : float, optional
        Observation time in years
        Default is 10.33.
    n_frequencies : int, optional
        Number of frequency bins
        Default is 30.
    signal_label : str, optional
        Label indicating the type of signal model to use
        Default is "power_law".
    signal_parameters : dict, optional
        Dictionary containing parameters for the signal model
        Default is SMBBH_parameters.
    realization : bool, optional
        Whether to generate a realization of the data
        Default is True.
    save_MCMC_data : bool, optional
        Whether to save the generated data
        Default is True.
    path_to_MCMC_data : str, optional
        Path to save or load the generated data
        Default is "generated_data/MCMC_data.npz".
    get_tensors_kwargs : dict, optional
        Additional keyword arguments for get_tensors
        Default is {}.
    generate_catalog_kwargs : dict, optional
        Additional keyword arguments for generate_catalog_kwargs
        Default is {}.

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

    try:
        if regenerate_MCMC_data:
            raise FileNotFoundError("Flag forces MCMC data regeneration")

        data = np.load(path_to_MCMC_data)
        frequency = data["frequency"]
        MCMC_data = data["data"]
        response_IJ = data["response_IJ"]
        strain_omega = data["strain_omega"]

    except FileNotFoundError:
        print("\nRegenerating MCMC data")

        # Setting the frequency vector from the observation time
        T_tot = T_obs_yrs * yr
        fmin = 1 / T_tot
        frequency = fmin * (1 + np.arange(n_frequencies))

        # Get the functions for the signal and its derivatives
        model = get_model(signal_label)
        signal_model = model["signal_model"]

        # Computing (sqrt of the) signal
        signal_std = np.sqrt(signal_model(frequency, signal_parameters))

        # Gets all the ingredients to compute the fisher
        strain_omega, response_IJ, HD_functions_IJ, HD_coeffs = get_tensors(
            frequency, **get_tensors_kwargs, **generate_catalog_kwargs
        )

        # Generate MCMC data
        frequency, MCMC_data, response_IJ, strain_omega = generate_MCMC_data(
            realization,
            frequency,
            signal_std,
            strain_omega,
            response_IJ,
            HD_functions_IJ,
            HD_coeffs,
            save_MCMC_data=save_MCMC_data,
            path_to_MCMC_data=path_to_MCMC_data,
        )

    return frequency, MCMC_data, response_IJ, strain_omega


def flat_prior(parameter, min_val, max_val):
    """
    Calculate the logarithm of a flat prior probability density function.

    Parameters:
    -----------
    parameter : float or jax.numpy.ndarray
        Parameter value(s) for which the prior probability is calculated.
    min_val : float
        Minimum allowed value for the parameter.
    max_val : float
        Maximum allowed value for the parameter.

    Returns:
    --------
    float or jax.numpy.ndarray
        Logarithm of the prior probability density function.

    """

    if not (parameter >= min_val and parameter <= max_val):
        return -jnp.inf
    else:
        return 0.0


def gaussian_prior(parameter, mean_val, std_val):
    """
    Calculate the logarithm of a Gaussian prior probability density function.

    Parameters:
    -----------
    parameter : float or jax.numpy.ndarray
        Parameter value(s) for which the prior probability is calculated.
    mean_val : float
        Mean value of the Gaussian distribution.
    std_val : float
        Standard deviation of the Gaussian distribution.

    Returns:
    --------
    float or jax.numpy.ndarray
        Logarithm of the prior probability density function.

    """

    return -0.5 * jnp.sum(
        jnp.log(2 * jnp.pi * std_val**2)
        + ((parameter - mean_val) / std_val) ** 2
    )


def log_prior(parameters, prior_parameters, which_prior="flat"):
    """
    Calculate the logarithm of the prior probability density function. Only
    flat and gaussian priors are supported at the moment

    Parameters:
    -----------
    parameters : jax.numpy.ndarray
        Array of parameter values for which the prior probability is calculated.
    prior_parameters : numpy.ndarray
        Array containing prior parameters, where each row represents the
        parameters for a single prior.
        For flat prior: [min_val, max_val]
        For Gaussian prior: [mean_val, std_val]
    which_prior : str, optional
        Type of prior distribution to use
        Default is "flat"

    Returns:
    --------
    float
        Logarithm of the prior probability density function.

    """

    log_p = 0.0

    for i in range(len(parameters)):
        if which_prior == "flat":
            log_p += flat_prior(parameters[i], *prior_parameters[:, i])
        elif which_prior == "gaussian":
            log_p += gaussian_prior(parameters[i], *prior_parameters[:, i])

    return log_p


@jit
def log_likelihood(
    frequency,
    signal_value,
    response_IJ,
    strain_omega,
    data,
):
    """
    Compute the logarithm of the likelihood assujming a Whittle likelihood.

    Parameters:
    -----------
    frequency : numpy.ndarray
        Array containing frequency bins.
    signal_value : numpy.ndarray
        Array containing the signal evaluated in all frequency bins.
    response_IJ : numpy.ndarray
        Array containing response function.
    strain_omega : numpy.ndarray
        Array containing strain noise.
    data : numpy.ndarray
        Array containing the observed data.

    Returns:
    --------
    float
        Logarithm of the likelihood.

    """

    # Covariance of the data
    covariance = response_IJ * signal_value[:, None, None] + strain_omega

    # Inverse of the covariance
    c_inverse = compute_inverse(covariance)

    # Log determinant
    sign, logdet = jnp.linalg.slogdet(covariance)

    # data term
    data_term = jnp.abs(jnp.einsum("ijk,ikj->i", c_inverse, data))

    # return the likelihood
    return -jnp.sum(logdet + data_term)


def log_posterior(
    signal_parameters,
    frequency,
    signal_model,
    response_IJ,
    strain_omega,
    data,
    prior_parameters,
    which_prior="flat",
):
    """
    Compute the logarithm of the posterior probability summing log likelihood
    and prior.

    Parameters:
    -----------
    signal_parameters : numpy.ndarray
        Array containing parameters of the signal model.
    frequency : numpy.ndarray
        Array containing frequency bins.
    signal_model : callable
        Function representing the signal model.
    response_IJ : numpy.ndarray
        Array containing response function.
    strain_omega : numpy.ndarray
        Array containing strain noise.
    data : numpy.ndarray
        Array containing the observed data.
    prior_parameters : numpy.ndarray
        Array containing prior parameters.
    which_prior : str, optional
        Type of prior distribution to use (default is "flat").

    Returns:
    --------
    float
        Logarithm of the posterior probability.

    """

    lp = log_prior(signal_parameters, prior_parameters)

    if not jnp.isfinite(lp):
        return -jnp.inf

    signal_value = signal_model(frequency, signal_parameters)

    log_lik = log_likelihood(
        frequency, signal_value, response_IJ, strain_omega, data
    )

    return lp + log_lik


def run_MCMC(
    priors,
    T_obs_yrs=10.33,
    n_frequencies=30,
    signal_label="power_law",
    signal_parameters=SMBBH_parameters,
    initial=[],
    which_prior="flat",
    regenerate_MCMC_data=False,
    realization=True,
    save_MCMC_data=True,
    path_to_MCMC_data="generated_data/MCMC_data.npz",
    i_max=i_max_default,
    R_convergence=R_convergence_default,
    R_criterion=R_criterion_default,
    burnin_steps=burnin_steps_default,
    MCMC_iteration_steps=MCMC_iteration_steps_default,
    path_to_MCMC_chains="generated_chains/MCMC_chains.npz",
    get_tensors_kwargs={},
    generate_catalog_kwargs={},
):
    """
    Run Markov Chain Monte Carlo (MCMC) to estimate the posterior distribution
    of the parameters given the observed data. The initial points are generated
    randomly within the priors. After a burn-in phase, the Gelman-Rubin
    statistic is used as a convergence diagnostic. Several MCMC iterations are
    run until the chains reach convergence. MCMC samples and log posterior
    probabilities are stored in the specified path.

    Parameters:
    -----------
    priors : numpy.ndarray
        Array containing prior parameters. Each row represents the parameters
        for a single prior distribution.
    T_obs_yrs : float, optional
        Total observation time in years
        Default is 10.33
    n_frequencies : int, optional
        Number of frequency bins
        Default is 30
    signal_label : str, optional
        Label indicating the type of signal model to use
        Default is "power_law"
    signal_parameters : numpy.ndarray, optional
        Array containing signal model parameters
        Default is SMBBH_parameters
    initial : list or numpy.ndarray, optional
        Initial parameter values for the MCMC walkers
        Default is empty
    which_prior : str, optional
        Type of prior distribution to use
        Default is "flat"
    regenerate_MCMC_data : bool, optional
        Flag indicating whether to regenerate MCMC data
        Default is False
    realization : bool, optional
        Flag indicating whether to generate a data realization
        Default is True
    save_MCMC_data : bool, optional
        Flag indicating whether to save MCMC data
        Default is True
    path_to_MCMC_data : str, optional
        Path to save MCMC data
        Default is "generated_data/MCMC_data.npz"
    i_max : int, optional
        Maximum number of iterations for convergence
        Default is i_max_default
    R_convergence : float, optional
        Convergence threshold for the Gelman-Rubin statistic
        Default is R_convergence_default
    R_criterion : str, optional
        Criterion to calculate the Gelman-Rubin statistic
        Default is R_criterion_default
    burnin_steps : int, optional
        Number of burn-in steps for the MCMC sampler
        Default is burnin_steps_default
    MCMC_iteration_steps : int, optional
        Number of MCMC iteration steps
        Default is MCMC_iteration_steps_default
    path_to_MCMC_chains : str, optional
        Path to save MCMC chains
        Default is "generated_chains/MCMC_chains.npz"
    get_tensors_kwargs : dict, optional
        Additional keyword arguments for getting tensors
        Default is an empty dictionary
    generate_catalog_kwargs : dict, optional
        Additional keyword arguments for generating the catalog
        Default is an empty dictionary

    Returns:
    --------
    Tuple containing:
    - samples: numpy.ndarray
        Array containing MCMC samples.
    - pdfs: numpy.ndarray
        Array containing log posterior probabilities for MCMC samples.

    """

    # Get the data
    frequency, MCMC_data, response_IJ, strain_omega = get_MCMC_data(
        regenerate_MCMC_data,
        T_obs_yrs=T_obs_yrs,
        n_frequencies=n_frequencies,
        signal_label=signal_label,
        signal_parameters=signal_parameters,
        realization=realization,
        save_MCMC_data=save_MCMC_data,
        path_to_MCMC_data=path_to_MCMC_data,
        get_tensors_kwargs=get_tensors_kwargs,
        generate_catalog_kwargs=generate_catalog_kwargs,
    )

    nwalkers = max(2 * len(priors.T), 5)

    # Generate the initial points if not provided
    if not initial and which_prior.lower() == "flat":
        initial = np.random.uniform(
            priors[0, :], priors[1, :], size=(nwalkers, len(priors.T))
        )
    elif not initial and which_prior.lower() == "gaussian":
        initial = np.random.uniform(
            priors[0, :], priors[1, :], size=(nwalkers, len(priors.T))
        )
    elif initial:
        pass
    else:
        raise ValueError("Cannot use that prior")

    nwalkers, ndims = initial.shape

    # Args for the posterior
    args = [
        jnp.array(frequency),
        get_model(signal_label)["signal_model"],
        jnp.array(response_IJ),
        jnp.array(strain_omega),
        jnp.array(MCMC_data),
        priors,
    ]

    # Kwargs for the posterior
    kwargs = {"which_prior": which_prior}

    # Set the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndims, log_posterior, args=args, kwargs=kwargs
    )

    start = time.perf_counter()

    print("Initial run")
    state = sampler.run_mcmc(initial, burnin_steps, progress=True)

    sampler.reset()
    print("Burn-in dropped, here starts the proper run")

    R = 1e100
    i = 0

    # Run until convergence or until reached maximum number of iterations
    while np.abs(R - 1) > R_convergence and i < i_max:
        # Run this iteration
        state = sampler.run_mcmc(state, MCMC_iteration_steps, progress=True)

        # Get Gelman-Rubin at this step
        R_array = get_R(sampler.get_chain())

        if R_criterion.lower() == "mean_squared":
            R = np.sqrt(np.mean(R_array**2))
        elif R_criterion.lower() == "max":
            R = np.max(R_array)
        else:
            raise ValueError("Cannot use R_criterion =", R_criterion)

        # state = None
        print("At this step R = %.4f" % (R))
        i += 1

    # Samples and pdfs
    samples = sampler.get_chain(flat=True)
    pdfs = sampler.lnprobability

    print("This took {0:.1f} seconds \n".format(time.perf_counter() - start))

    print("Storing as", path_to_MCMC_chains)
    np.savez(path_to_MCMC_chains, samples=samples, pdfs=pdfs)

    return samples, pdfs
