# Global
import time
import numpy as np
import emcee
import blackjax
import blackjax.ns.utils as ns_utils

import jax
import jax.numpy as jnp

# Local
import fastPTA.utils as ut
from fastPTA.signal_templates.signal_utils import SMBBH_parameters
from fastPTA.signals import get_signal_model
from fastPTA.get_tensors import get_tensors
from fastPTA.data.generate_data import generate_inference_data
from fastPTA.inference_tools.likelihoods import (
    log_posterior,
    log_likelihood,
    prepare_log_likelihood,
)

# Set the device
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# Setting some constants
i_max_default = 100
R_convergence_default = 1e-1
R_criterion_default = "mean_squared"
MCMC_iteration_steps_default = 500
# BlackjaxSampler jit-compiles its scan, so using the same number of steps for
# the burn-in and the iteration runs avoids a second compile.
burnin_steps_default = MCMC_iteration_steps_default
n_live_default = 500
dlogz_default = 1e-2
max_iterations_default = 100000
power_law_model = get_signal_model("power_law")


class BlackjaxSampler:
    """
    Helper class to define an interface for blackjax samplers that is compatible
    with get_MCMC_samples (run_mcmc, reset, get_chain, get_log_prob). Steps are
    run with jax.lax.scan so a whole run_mcmc call compiles and executes as a
    single program. Samplers in adaptive_samplers (e.g. "nuts", "hmc") get their
    step size and mass matrix tuned per walker via blackjax's window adaptation
    before the burn-in to increase efficiency.

    Parameters:
    -----------
    sampler : str
        Name of the blackjax sampling algorithm to use (e.g. "nuts", "hmc",
        "normal_random_walk").
    log_posterior : function
        Function to compute the logarithm of the posterior probability.
    log_posterior_args : list
        List containing the arguments for the log posterior function.
    nwalkers : int
        Number of MCMC walkers.
    ndims : int
        Number of parameters.
    num_adaptation_steps : int, optional
        Number of window adaptation steps, only used for samplers in
        adaptive_samplers. Default is 1000.
    sampler_kwargs : dict
        Keyword arguments for the blackjax sampling algorithm.

    """

    # blackjax algorithms tuned automatically via window adaptation
    adaptive_samplers = ("nuts", "hmc")

    def __init__(
        self,
        sampler,
        log_posterior,
        log_posterior_args,
        nwalkers,
        ndims,
        num_adaptation_steps=1000,
        **sampler_kwargs,
    ):

        def logdensity_fn(parameters):
            return log_posterior(parameters, *log_posterior_args)

        self.sampler = sampler
        self.logdensity_fn = logdensity_fn
        self.sampler_kwargs = sampler_kwargs
        self.num_adaptation_steps = num_adaptation_steps
        self.nwalkers = nwalkers
        self.ndims = ndims
        self.rng_key = ut.new_rng_key()
        self.chain = None
        self.pdfs = None
        self.initialized = False

        self.algorithm = getattr(blackjax, sampler)
        self.warmup = (
            blackjax.window_adaptation(
                self.algorithm, logdensity_fn, **sampler_kwargs
            )
            if sampler in self.adaptive_samplers
            else None
        )

    def reset(self):
        # Drop the stored chain and log-probs, mirroring
        # emcee.EnsembleSampler.reset
        self.chain = None
        self.pdfs = None

    def _initialize(self, initial):
        """
        Build the (batched, jitted) step function used by run_mcmc, tuning
        the step size and mass matrix first if the sampler needs it, and
        initialize the per-walker sampler state from the starting points.

        Parameters:
        -----------
        initial : array-like
            Starting parameter values for the walkers, shape (nwalkers,
            ndims).

        Returns:
        --------
        blackjax sampler state
            Initial per-walker state to be passed to run_mcmc.

        """

        # Build the (batched) step function, tuning the step size and mass
        # matrix first (if the sampler needs it)
        state = initial

        if self.warmup is not None:
            self.rng_key, warmup_key = jax.random.split(self.rng_key)
            warmup_keys = jax.random.split(warmup_key, self.nwalkers)
            run_warmup = jax.jit(
                jax.vmap(
                    lambda k, p: self.warmup.run(
                        k, p, self.num_adaptation_steps
                    )
                )
            )
            (state, tuned_params), _ = run_warmup(warmup_keys, state)

            # Each walker got its own step_size/inverse_mass_matrix from warmup.
            # The kernel has to be rebuilt per walker; tuned_step does this once
            # per walker at trace time
            tuned_step = jax.vmap(
                lambda key, s, params: self.algorithm(
                    self.logdensity_fn, **self.sampler_kwargs, **params
                ).step(key, s)
            )

            def step(keys, s):
                return tuned_step(keys, s, tuned_params)

        else:
            algorithm = self.algorithm(
                self.logdensity_fn, **self.sampler_kwargs
            )
            step = jax.vmap(algorithm.step)
            state = jax.vmap(algorithm.init)(state)

        def scan_body(state, keys):
            state, _ = step(keys, state)
            return state, (state.position, state.logdensity)

        # donate_argnums=0: state is never read again after this call (the
        # caller immediately reassigns its variable to the returned state),
        # so XLA can reuse its buffers instead of allocating fresh ones for
        # the output -- ~25% faster per run_mcmc call in a CPU benchmark
        self.scan = jax.jit(
            lambda state, keys: jax.lax.scan(scan_body, state, keys),
            donate_argnums=0,
        )
        self.initialized = True
        return state

    def run_mcmc(self, state, n_steps, progress=False):
        """
        Run n_steps of sampling from state, appending the result to the
        stored chain. All steps are run in a single jax.lax.scan, so the
        whole run compiles and executes as one program.

        Parameters:
        -----------
        state : numpy.ndarray or blackjax sampler state
            Starting points (on the first call) or the state returned by a
            previous run_mcmc call.
        n_steps : int
            Number of sampling steps to run.
        progress : bool, optional
            Whether to print the time taken to run the steps
            Default is False

        Returns:
        --------
        blackjax sampler state
            State to be passed to the next run_mcmc call.

        """

        if not self.initialized:
            state = self._initialize(state)

        self.rng_key, scan_key = jax.random.split(self.rng_key)
        keys = jax.random.split(scan_key, n_steps * self.nwalkers).reshape(
            n_steps, self.nwalkers, 2
        )

        start = time.perf_counter()
        state, (positions, pdfs) = self.scan(state, keys)

        if progress:
            print(
                "Ran {0} steps in {1:.1f} seconds".format(
                    n_steps, time.perf_counter() - start
                )
            )

        self.chain = self._append(self.chain, positions)
        self.pdfs = self._append(self.pdfs, pdfs)

        return state

    @staticmethod
    def _append(stored, new):
        # Append along the steps axis, or just start the chain the first time
        if stored is None:
            return new

        return jnp.concatenate([stored, new], axis=0)

    def get_chain(self, flat=False):
        """
        Return the stored chain.

        Parameters:
        -----------
        flat : bool, optional
            Whether to flatten the steps and walkers into a single axis
            Default is False

        Returns:
        --------
        jax.Array
            Array with shape (n_steps, nwalkers, ndims), or (n_steps *
            nwalkers, ndims) if flat.

        """
        return self.chain.reshape(-1, self.ndims) if flat else self.chain

    def get_log_prob(self, flat=False):
        """
        Return the stored log posterior probabilities, i.e. one value per
        sample in get_chain().

        Parameters:
        -----------
        flat : bool, optional
            Whether to flatten the steps and walkers into a single axis
            Default is False

        Returns:
        --------
        jax.Array
            Array with shape (n_steps, nwalkers), or (n_steps * nwalkers,)
            if flat.

        """

        return self.pdfs.reshape(-1) if flat else self.pdfs


def get_mcmc_sampler(
    sampler, log_posterior, log_posterior_args, nwalkers, ndims, sampler_kwargs
):
    """
    Build the sampler backend requested by get_MCMC_samples: an
    emcee.EnsembleSampler if sampler is "emcee", otherwise a BlackjaxSampler
    running the named blackjax algorithm.

    Parameters:
    -----------
    sampler : str
        Sampler to use, either "emcee" or the name of a blackjax sampling
        algorithm (e.g. "nuts", "hmc", "normal_random_walk").
    log_posterior : function
        Function to compute the logarithm of the posterior probability.
    log_posterior_args : list
        List containing the arguments for the log posterior function.
    nwalkers : int
        Number of MCMC walkers.
    ndims : int
        Number of parameters.
    sampler_kwargs : dict
        Keyword arguments for the sampler.

    Returns:
    --------
    emcee.EnsembleSampler or BlackjaxSampler
        Sampler exposing run_mcmc, reset, get_chain and get_log_prob.

    """

    if sampler == "emcee":
        return emcee.EnsembleSampler(
            nwalkers,
            ndims,
            log_posterior,
            args=log_posterior_args,
            **sampler_kwargs,
        )

    return BlackjaxSampler(
        sampler,
        log_posterior,
        log_posterior_args,
        nwalkers,
        ndims,
        **sampler_kwargs,
    )


def get_MCMC_samples(
    log_posterior,
    initial,
    log_posterior_args,
    sampler="emcee",
    sampler_kwargs={},
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
    sampler : str, optional
        Sampler to use, either "emcee" or the name of a blackjax sampling
        algorithm (e.g. "nuts", "hmc", "normal_random_walk")
        Default is "emcee"
    sampler_kwargs : dict, optional
        Keyword arguments for the sampler: passed to emcee.EnsembleSampler
        if sampler is "emcee", otherwise to the blackjax sampling algorithm.
        For samplers in BlackjaxSampler.adaptive_samplers (e.g. "nuts",
        "hmc") may also set num_adaptation_steps, the number of window
        adaptation steps used to tune the step size and mass matrix
        Default is {}
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
    mcmc_sampler = get_mcmc_sampler(
        sampler,
        log_posterior,
        log_posterior_args,
        nwalkers,
        ndims,
        sampler_kwargs,
    )

    start = time.perf_counter()

    if print_progress:
        print("Initial run")

    state = mcmc_sampler.run_mcmc(
        initial, burnin_steps, progress=print_progress
    )

    mcmc_sampler.reset()
    if print_progress:
        print("Burn-in dropped, here starts the proper run")

    R = 1e100
    i = 0
    R_state = None

    # Run until convergence or until reached maximum number of iterations
    while np.abs(R - 1) > R_convergence and i < i_max:
        # Run this iteration
        state = mcmc_sampler.run_mcmc(
            state, MCMC_iteration_steps, progress=print_progress
        )

        # Get Gelman-Rubin at this step: fold in only the new batch from
        # this iteration (always the same shape) rather than recomputing
        # from the whole chain (which grows every iteration and would
        # defeat ut.get_R's jit if used directly here, see ut.update_R)
        new_batch = mcmc_sampler.get_chain()[-MCMC_iteration_steps:]
        R_state, R_array = ut.update_R(R_state, new_batch)

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

    # return samples and pdfs as plain numpy arrays: emcee already gives numpy,
    # blackjax gives jax arrays, and downstream consumers (np.savez, corner)
    # expect numpy, so convert once here rather than on every run_mcmc call.
    # Both are flattened the same way, so pdfs[i] is samples[i]'s log-prob
    return (
        np.array(mcmc_sampler.get_chain(flat=True)),
        np.array(mcmc_sampler.get_log_prob(flat=True)),
    )


def get_nested_samples(
    logprior_fn,
    loglikelihood_fn,
    initial,
    num_inner_steps=None,
    num_delete=1,
    dlogz=dlogz_default,
    max_iterations=max_iterations_default,
    n_posterior_samples=5000,
    print_progress=True,
):
    """
    Run Nested Sampling (blackjax's nested slice sampler) to estimate the
    posterior distribution and the Bayesian evidence given the log-prior and
    log-likelihood functions. Live points (`initial`) are replaced from
    below each iteration; the run stops once their remaining evidence
    contribution drops below a fraction dlogz of the evidence found so far.
    Dead and live points are then resampled by importance weight into
    equally-weighted posterior samples.

    Parameters:
    -----------
    logprior_fn : function
        Function to compute the log-prior probability of a single particle.
    loglikelihood_fn : function
        Function to compute the log-likelihood of a single particle.
    initial : numpy.ndarray
        Initial live points, drawn from the priors.
    num_inner_steps : int, optional
        Slice-sampling steps per new live point. If None, uses max(5, 2 *
        ndims), as recommended by blackjax
        Default is None
    num_delete : int, optional
        Live points deleted and replaced per iteration. Values above 1
        speed up the run but make the evidence estimate coarser, since the
        mean-shrinkage approximation used here is exact only for 1
        Default is 1
    dlogz : float, optional
        Stop once the live points' remaining evidence contribution drops
        below this fraction of the evidence found so far
        Default is dlogz_default
    max_iterations : int, optional
        Maximum number of iterations, reached only if dlogz is not met
        Default is max_iterations_default
    n_posterior_samples : int, optional
        Number of equally-weighted posterior samples to draw
        Default is 5000
    print_progress : bool, optional
        Whether to print progress of the run
        Default is True

    Returns:
    --------
    Tuple containing:
    - samples: numpy.ndarray
        Array containing equally-weighted posterior samples.
    - logZ_mean: float
        Estimated log-evidence.
    - logZ_std: float
        Uncertainty (standard deviation) on the log-evidence estimate.

    """

    n_live, ndims = initial.shape
    num_inner_steps = num_inner_steps or max(5, 2 * ndims)

    # Set the sampler
    algorithm = blackjax.nss(
        logprior_fn, loglikelihood_fn, num_inner_steps, num_delete=num_delete
    )

    rng_key = ut.new_rng_key()
    rng_key, init_key = jax.random.split(rng_key)
    state = algorithm.init(initial, rng_key=init_key)
    step = jax.jit(algorithm.step)

    start = time.perf_counter()

    # The prior volume shrinks by a constant factor exp(-num_delete / n_live)
    # every iteration (Skilling's mean-shrinkage approximation)
    logdX = ns_utils.log1mexp(jnp.array(-num_delete / n_live))

    # Run until the live points can no longer meaningfully add to the
    # evidence, or until reached the maximum number of iterations
    dead = []
    logZ = -jnp.inf
    i = 0

    while i < max_iterations:
        rng_key, step_key = jax.random.split(rng_key)
        state, info = step(step_key, state)
        dead.append(info)

        logX_prev = -i * num_delete / n_live
        logZ = jnp.logaddexp(
            logZ, jnp.max(info.particles.loglikelihood) + logX_prev + logdX
        )

        i += 1
        if print_progress and i % 100 == 0:
            print("At iteration %d, logZ = %.4f" % (i, logZ))

        # Remaining evidence if all the live points were at the peak
        logX_curr = -i * num_delete / n_live
        logZ_remain = jnp.max(state.particles.loglikelihood) + logX_curr

        if jnp.exp(logZ_remain - logZ) < dlogz:
            break

    if print_progress:
        print(
            "Stopped after {0} iterations, this took {1:.1f} seconds \n".format(
                i, time.perf_counter() - start
            )
        )

    # Combine the dead and (final) live particles
    finalised = ns_utils.finalise(state, dead)

    # Evidence estimate (and its uncertainty) from the importance weights
    rng_key, weight_key, sample_key = jax.random.split(rng_key, 3)
    logZ_samples = jax.scipy.special.logsumexp(
        ns_utils.log_weights(weight_key, finalised), axis=0
    )
    logZ_mean = jnp.mean(logZ_samples)
    logZ_std = jnp.std(logZ_samples)

    # Equally-weighted posterior samples, resampled from the importance weights
    samples = np.array(
        ns_utils.sample(
            sample_key, finalised, shape=n_posterior_samples
        ).position
    )

    if print_progress:
        print("log-evidence: %.4f +/- %.4f" % (logZ_mean, logZ_std))

    return samples, logZ_mean, logZ_std


def get_inference_data(
    regenerate_inference_data,
    T_obs_yrs=10.33,
    n_frequencies=30,
    signal_model=power_law_model,
    signal_parameters=SMBBH_parameters,
    realization=True,
    save_inference_data=True,
    path_to_inference_data="generated_data/inference_data.npz",
    get_tensors_kwargs={},
    generate_catalog_kwargs={},
):
    """
    Loads or regenerates the data used by either inference method (MCMC or
    nested sampling). If `regenerate_inference_data` is True or the data
    file is not found, the data is regenerated. Otherwise, it is loaded from
    the specified path. Data contain both signal and noise. If the data are
    generated, if `realization` is True, it generates a realization of the
    data. If `save_inference_data` is True, the generated data is saved to a
    specified path. Additional keyword arguments for get_tensors and
    generate_pulsars_catalog can be provided via get_tensors_kwargs and
    generate_catalog_kwargs.

    Parameters:
    -----------
    regenerate_inference_data : bool
        Whether to regenerate the data.
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
    save_inference_data : bool, optional
        Whether to save the generated data
        Default is True.
    path_to_inference_data : str, optional
        Path to save or load the generated data
        Default is "generated_data/inference_data.npz".
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
        if regenerate_inference_data:
            raise FileNotFoundError("Flag forces data regeneration")

        data = np.load(path_to_inference_data)
        frequency = jnp.asarray(data["frequency"])
        inference_data = jnp.asarray(data["data"])
        response_IJ = jnp.asarray(data["response_IJ"])
        strain_omega = jnp.asarray(data["strain_omega"])

    except FileNotFoundError:
        print("\nRegenerating inference data")

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

        # Generate the data
        frequency, inference_data, response_IJ, strain_omega = (
            generate_inference_data(
                realization,
                frequency,
                signal_std,
                strain_omega,
                response_IJ,
                HD_functions_IJ,
                HD_coeffs,
                save_inference_data=save_inference_data,
                path_to_inference_data=path_to_inference_data,
            )
        )

    return frequency, inference_data, response_IJ, strain_omega


def get_inference_ingredients(
    regenerate_inference_data,
    T_obs_yrs=10.33,
    n_frequencies=30,
    signal_model=power_law_model,
    signal_parameters=SMBBH_parameters,
    realization=True,
    save_inference_data=True,
    path_to_inference_data="generated_data/inference_data.npz",
    get_tensors_kwargs={},
    generate_catalog_kwargs={},
):
    """
    Shared first step of run_inference, for both method="mcmc" and
    method="nested_sampling": loads/generates the data (see
    get_inference_data), then diagonalizes the response w.r.t. the noise
    once (see prepare_log_likelihood), since both are fixed throughout any
    inference run and only the signal changes at every likelihood call.

    Parameters:
    -----------
    Same as get_inference_data.

    Returns:
    --------
    Tuple containing:
    - frequency: numpy.ndarray
        Array containing frequency bins.
    - eigenvalues: Array
        Generalized eigenvalues of response_IJ w.r.t. strain_omega, per
        frequency (see prepare_log_likelihood).
    - noise_logdet: Array
        Log determinant of strain_omega, per frequency.
    - data_eigenbasis: Array
        Data projected onto the generalized eigenbasis, per frequency.

    """

    frequency, inference_data, response_IJ, strain_omega = get_inference_data(
        regenerate_inference_data,
        T_obs_yrs=T_obs_yrs,
        n_frequencies=n_frequencies,
        signal_model=signal_model,
        signal_parameters=signal_parameters,
        realization=realization,
        save_inference_data=save_inference_data,
        path_to_inference_data=path_to_inference_data,
        get_tensors_kwargs=get_tensors_kwargs,
        generate_catalog_kwargs=generate_catalog_kwargs,
    )

    eigenvalues, noise_logdet, data_eigenbasis = prepare_log_likelihood(
        inference_data, response_IJ, strain_omega
    )

    return frequency, eigenvalues, noise_logdet, data_eigenbasis


def run_inference(
    priors,
    method="mcmc",
    T_obs_yrs=10.33,
    n_frequencies=30,
    signal_model=power_law_model,
    signal_parameters=SMBBH_parameters,
    regenerate_inference_data=False,
    realization=True,
    save_inference_data=True,
    path_to_inference_data="generated_data/inference_data.npz",
    print_progress=True,
    get_tensors_kwargs={},
    generate_catalog_kwargs={},
    # -- method="mcmc" only --
    initial=jnp.array([False]),
    sampler="emcee",
    sampler_kwargs={},
    i_max=i_max_default,
    R_convergence=R_convergence_default,
    R_criterion=R_criterion_default,
    burnin_steps=burnin_steps_default,
    MCMC_iteration_steps=MCMC_iteration_steps_default,
    path_to_MCMC_chains="generated_chains/MCMC_chains.npz",
    # -- method="nested_sampling" only --
    n_live=n_live_default,
    num_inner_steps=None,
    num_delete=1,
    dlogz=dlogz_default,
    max_iterations=max_iterations_default,
    n_posterior_samples=5000,
    path_to_NS_chains="generated_chains/NS_chains.npz",
):
    """
    Run either Markov Chain Monte Carlo (method="mcmc") or Nested Sampling
    (method="nested_sampling") to estimate the posterior distribution of the
    parameters given the observed data. Both methods share the data-loading
    and likelihood-preparation step (see get_inference_ingredients); only
    the parameter groups marked "-only" below, and the sampler itself,
    differ between the two.

    For method="mcmc": the initial points are generated randomly within the
    priors (or taken from `initial` if given). After a burn-in phase, the
    Gelman-Rubin statistic is used as a convergence diagnostic; several MCMC
    iterations are run until the chains reach convergence (see
    get_MCMC_samples). MCMC samples and log posterior probabilities are
    stored in path_to_MCMC_chains and returned.

    For method="nested_sampling": live points are drawn from the priors and
    evolved until their remaining evidence contribution is negligible (see
    get_nested_samples). Equally-weighted posterior samples and the
    Bayesian evidence are stored in path_to_NS_chains and returned.

    Parameters:
    -----------
    priors : prior object
        Object containing the prior probability density functions.
    method : str, optional
        Inference method to run, either "mcmc" or "nested_sampling"
        Default is "mcmc"
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
    regenerate_inference_data : bool, optional
        Flag indicating whether to regenerate the data
        Default is False
    realization : bool, optional
        Flag indicating whether to generate a data realization
        Default is True
    save_inference_data : bool, optional
        Flag indicating whether to save the data
        Default is True
    path_to_inference_data : str, optional
        Path to save/load the data
        Default is "generated_data/inference_data.npz"
    print_progress : bool, optional
        Whether to print progress of the run
        Default is True
    get_tensors_kwargs : dict, optional
        Additional keyword arguments for getting tensors
        Default is an empty dictionary
    generate_catalog_kwargs : dict, optional
        Additional keyword arguments for generating the catalog
        Default is an empty dictionary
    initial : list or numpy.ndarray, optional (method="mcmc" only)
        Initial parameter values for the MCMC walkers
        Default is empty (drawn from the priors)
    sampler : str, optional (method="mcmc" only)
        Sampler to use, either "emcee" or the name of a blackjax sampling
        algorithm (e.g. "nuts", "hmc", "normal_random_walk")
        Default is "emcee"
    sampler_kwargs : dict, optional (method="mcmc" only)
        Keyword arguments for the sampler: passed to emcee.EnsembleSampler
        if sampler is "emcee", otherwise to the blackjax sampling algorithm.
        For samplers in BlackjaxSampler.adaptive_samplers (e.g. "nuts",
        "hmc") may also set num_adaptation_steps, the number of window
        adaptation steps used to tune the step size and mass matrix
        Default is {}
    i_max : int, optional (method="mcmc" only)
        Maximum number of iterations for convergence
        Default is i_max_default
    R_convergence : float, optional (method="mcmc" only)
        Convergence threshold for the Gelman-Rubin statistic
        Default is R_convergence_default
    R_criterion : str, optional (method="mcmc" only)
        Criterion to calculate the Gelman-Rubin statistic
        Default is R_criterion_default
    burnin_steps : int, optional (method="mcmc" only)
        Number of burn-in steps for the MCMC sampler
        Default is burnin_steps_default
    MCMC_iteration_steps : int, optional (method="mcmc" only)
        Number of MCMC iteration steps
        Default is MCMC_iteration_steps_default
    path_to_MCMC_chains : str, optional (method="mcmc" only)
        Path to save MCMC chains
        Default is "generated_chains/MCMC_chains.npz"
    n_live : int, optional (method="nested_sampling" only)
        Number of live points
        Default is n_live_default
    num_inner_steps : int, optional (method="nested_sampling" only)
        Slice-sampling steps per new live point. If None, uses max(5, 2 *
        ndims), as recommended by blackjax
        Default is None
    num_delete : int, optional (method="nested_sampling" only)
        Live points deleted and replaced per iteration. Values above 1
        speed up the run but make the evidence estimate coarser, since the
        mean-shrinkage approximation used here is exact only for 1
        Default is 1
    dlogz : float, optional (method="nested_sampling" only)
        Stop once the live points' remaining evidence contribution drops
        below this fraction of the evidence found so far
        Default is dlogz_default
    max_iterations : int, optional (method="nested_sampling" only)
        Maximum number of iterations, reached only if dlogz is not met
        Default is max_iterations_default
    n_posterior_samples : int, optional (method="nested_sampling" only)
        Number of equally-weighted posterior samples to draw
        Default is 5000
    path_to_NS_chains : str, optional (method="nested_sampling" only)
        Path to save the posterior samples and evidence
        Default is "generated_chains/NS_chains.npz"

    Returns:
    --------
    For method="mcmc", tuple containing:
    - samples: numpy.ndarray
        Array containing MCMC samples.
    - pdfs: numpy.ndarray
        Array containing log posterior probabilities for MCMC samples.

    For method="nested_sampling", tuple containing:
    - samples: numpy.ndarray
        Array containing equally-weighted posterior samples.
    - logZ_mean: float
        Estimated log-evidence.
    - logZ_std: float
        Uncertainty (standard deviation) on the log-evidence estimate.

    """

    if method not in ("mcmc", "nested_sampling"):
        raise ValueError(
            'method must be "mcmc" or "nested_sampling", got %r' % (method,)
        )

    # Shared step: get the data and prepare the likelihood ingredients
    frequency, eigenvalues, noise_logdet, data_eigenbasis = (
        get_inference_ingredients(
            regenerate_inference_data,
            T_obs_yrs=T_obs_yrs,
            n_frequencies=n_frequencies,
            signal_model=signal_model,
            signal_parameters=signal_parameters,
            realization=realization,
            save_inference_data=save_inference_data,
            path_to_inference_data=path_to_inference_data,
            get_tensors_kwargs=get_tensors_kwargs,
            generate_catalog_kwargs=generate_catalog_kwargs,
        )
    )

    if method == "mcmc":
        ndims = len(priors.parameter_names)

        # Generate the initial points if not provided
        if not np.all(initial):
            nwalkers = max(2 * ndims, 5)
            initial = priors.sample(nwalkers)
        else:
            nwalkers = len(initial)

        # Args for the posterior
        log_posterior_args = [
            frequency,
            signal_model,
            eigenvalues,
            noise_logdet,
            data_eigenbasis,
            priors,
        ]

        # Samples and pdfs
        samples, pdfs = get_MCMC_samples(
            log_posterior,
            initial,
            log_posterior_args,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
            i_max=i_max,
            R_convergence=R_convergence,
            R_criterion=R_criterion,
            burnin_steps=burnin_steps,
            MCMC_iteration_steps=MCMC_iteration_steps,
            print_progress=print_progress,
        )

        print("Storing as", path_to_MCMC_chains)
        np.savez(path_to_MCMC_chains, samples=samples, pdfs=pdfs)

        return samples, pdfs

    # method == "nested_sampling"
    def logprior_fn(parameters):
        return priors.evaluate_log_priors(
            dict(zip(priors.parameter_names, parameters))
        )

    def loglikelihood_fn(parameters):
        signal_value = signal_model.template(frequency, parameters)
        return log_likelihood(
            signal_value, eigenvalues, noise_logdet, data_eigenbasis
        )

    # Draw the initial live points from the priors
    initial_live_points = priors.sample(n_live)

    # Samples and evidence
    samples, logZ_mean, logZ_std = get_nested_samples(
        logprior_fn,
        loglikelihood_fn,
        initial_live_points,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        dlogz=dlogz,
        max_iterations=max_iterations,
        n_posterior_samples=n_posterior_samples,
        print_progress=print_progress,
    )

    print("Storing as", path_to_NS_chains)
    np.savez(
        path_to_NS_chains,
        samples=samples,
        logZ_mean=np.array(logZ_mean),
        logZ_std=np.array(logZ_std),
    )

    return samples, logZ_mean, logZ_std
