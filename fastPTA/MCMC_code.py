# Global
import time
import numpy as np
import emcee

import jax
import jax.numpy as jnp

# Local
import fastPTA.utils as ut
from fastPTA.signal_templates.signal_utils import SMBBH_parameters
from fastPTA.signals import get_signal_model
from fastPTA.get_tensors import get_tensors
from fastPTA.data.generate_data import generate_MCMC_data
from fastPTA.inference_tools.likelihoods import log_posterior


# Set the device
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# Setting some constants
i_max_default = 100
R_convergence_default = 1e-1
R_criterion_default = "mean_squared"
burnin_steps_default = 300
MCMC_iteration_steps_default = 500
power_law_model = get_signal_model("power_law")


def get_MCMC_data(
    regenerate_MCMC_data,
    T_obs_yrs=10.33,
    n_frequencies=30,
    signal_model=power_law_model,
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
    signal_model : signal_model object, optional
        Object containing the signal model and its derivatives
        Default is a power_law model
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
        frequency = (1.0 + jnp.arange(n_frequencies)) / (T_obs_yrs * ut.yr)

        # Computing (sqrt of the) signal
        signal_std = np.sqrt(
            signal_model.template(frequency, signal_parameters)
        )

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


def get_MCMC_samples(
    log_posterior,
    initial,
    log_posterior_args,
    i_max=i_max_default,
    R_convergence=R_convergence_default,
    R_criterion=R_criterion_default,
    burnin_steps=burnin_steps_default,
    MCMC_iteration_steps=MCMC_iteration_steps_default,
    print_progress=True,
):
    """
    Run Markov Chain Monte Carlo (MCMC) to estimate the posterior distribution
    of the parameters given the observed data. After a burn-in phase, the
    Gelman-Rubin statistic is used as a convergence diagnostic. Several MCMC
    iterations are run until the chains reach convergence. MCMC samples and
    log posterior probabilities are returned.

    Parameters:
    -----------
    log_posterior : function
        Function to compute the logarithm of the posterior probability.
    initial : numpy.ndarray
        Initial parameter values for the MCMC walkers.
    log_posterior_args : list
        List containing the arguments for the log posterior function.
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
    print_progress : bool, optional
        Whether to print progress of the MCMC run
        Default is True

    Returns:
    --------
    Tuple containing:
    - samples: numpy.ndarray
        Array containing MCMC samples.
    - pdfs: numpy.ndarray
        Array containing log posterior probabilities for MCMC samples.

    """

    nwalkers, ndims = initial.shape

    # Set the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndims, log_posterior, args=log_posterior_args
    )

    start = time.perf_counter()

    if print_progress:
        print("Initial run")

    state = sampler.run_mcmc(initial, burnin_steps, progress=print_progress)

    sampler.reset()
    if print_progress:
        print("Burn-in dropped, here starts the proper run")

    R = 1e100
    i = 0

    # Run until convergence or until reached maximum number of iterations
    while np.abs(R - 1) > R_convergence and i < i_max:
        # Run this iteration
        state = sampler.run_mcmc(
            state, MCMC_iteration_steps, progress=print_progress
        )

        # Get Gelman-Rubin at this step
        R_array = ut.get_R(sampler.get_chain())

        if R_criterion.lower() == "mean_squared":
            R = np.sqrt(np.mean(R_array**2))
        elif R_criterion.lower() == "max":
            R = np.max(R_array)
        else:
            raise ValueError("Cannot use R_criterion =", R_criterion)

        if print_progress:
            print("At this step R = %.4f" % (R))
        i += 1

    if print_progress:
        print(
            "This took {0:.1f} seconds \n".format(time.perf_counter() - start)
        )

    # return samples and pdfs
    return sampler.get_chain(flat=True), sampler.get_log_prob()


def run_MCMC(
    priors,
    T_obs_yrs=10.33,
    n_frequencies=30,
    signal_model=power_law_model,
    signal_parameters=SMBBH_parameters,
    initial=jnp.array([False]),
    regenerate_MCMC_data=False,
    realization=True,
    save_MCMC_data=True,
    path_to_MCMC_data="generated_data/MCMC_data.npz",
    i_max=i_max_default,
    R_convergence=R_convergence_default,
    R_criterion=R_criterion_default,
    burnin_steps=burnin_steps_default,
    MCMC_iteration_steps=MCMC_iteration_steps_default,
    print_progress=True,
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
    priors : prior object
        Object containing the prior probability density functions.
    T_obs_yrs : float, optional
        Total observation time in years
        Default is 10.33
    n_frequencies : int, optional
        Number of frequency bins
        Default is 30
    signal_model : signal_model object, optional
        Object containing the signal model and its derivatives
        Default is a power_law model
    signal_parameters : numpy.ndarray, optional
        Array containing signal model parameters
        Default is SMBBH_parameters
    initial : list or numpy.ndarray, optional
        Initial parameter values for the MCMC walkers
        Default is empty
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
    print_progress : bool, optional
        Whether to print progress of the MCMC run
        Default is True
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
        signal_model=signal_model,
        signal_parameters=signal_parameters,
        realization=realization,
        save_MCMC_data=save_MCMC_data,
        path_to_MCMC_data=path_to_MCMC_data,
        get_tensors_kwargs=get_tensors_kwargs,
        generate_catalog_kwargs=generate_catalog_kwargs,
    )

    # Number of dimensions
    ndims = len(priors.parameter_names)

    # Generate the initial points if not provided
    if not np.all(initial):
        nwalkers = max(2 * ndims, 5)
        initial = np.empty((nwalkers, ndims))

        for i in range(ndims):
            pp = priors.priors[priors.parameter_names[i]]
            initial[:, i] = pp["rvs"](**pp["pdf_kwargs"], size=nwalkers)

    else:
        nwalkers = len(initial)

    # Args for the posterior
    log_posterior_args = [
        jnp.array(MCMC_data),
        jnp.array(frequency),
        signal_model,
        jnp.array(response_IJ),
        jnp.array(strain_omega),
        priors,
    ]

    # Samples and pdfs
    samples, pdfs = get_MCMC_samples(
        log_posterior,
        initial,
        log_posterior_args,
        i_max=i_max,
        R_convergence=R_convergence,
        R_criterion=R_criterion,
        burnin_steps=burnin_steps,
        MCMC_iteration_steps=MCMC_iteration_steps,
        print_progress=print_progress,
    )

    print("Storing as", path_to_MCMC_chains)
    np.savez(path_to_MCMC_chains, samples=samples, pdfs=pdfs)

    # Return the samples and pdfs
    return samples, pdfs
