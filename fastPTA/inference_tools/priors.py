# Global
import jax
from scipy import stats
import jax.numpy as jnp


# Local
import fastPTA.utils as ut
from fastPTA.compute_PBH_Abundance import get_PBH_abundance_from_interpolator


# Set the device
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


function_type = type(lambda x: x)


class Priors(object):
    """
    Class to define the prior probability density functions.

    """

    def __init__(
        self,
        priors_dictionary,
        get_PBH_abundance=False,
        check_PBH_abundance=True,
        use_PBH_interpolator=False,
        PBH_parameter_names=None,
        PBH_interpolator_kwargs=None,
    ):
        """
        Initialize the class.

        Parameters:
        -----------
        priors_dictionary : dictionary
            Dictionary containing the prior probability density functions.

        get_PBH_abundance : callable, optional
            Exact function to compute the PBH abundance from the full
            parameter vector. Default is False.

        check_PBH_abundance : bool, optional
            Check the PBH abundance using get_PBH_abundance. Default is True.

        use_PBH_interpolator : bool, optional
            If True, build an interpolator (see build_f_PBH_interpolator in
            compute_PBH_Abundance.py) from the priors' bounds and use it
            instead of get_PBH_abundance to speed up the PBH abundance
            check. Default is False (use the exact calculation).

        PBH_parameter_names : tuple of 3 str, optional
            Names, as they appear in priors_dictionary, of the log10
            amplitude, log10 width and log10 pivot [Hz] parameters, in
            that order. Required if use_PBH_interpolator is True. Assumes
            priors_dictionary's keys are ordered like the parameter vector
            get_PBH_abundance expects (as built from
            signal_model.parameter_names).

        PBH_interpolator_kwargs : dictionary, optional
            Extra keyword arguments passed to build_f_PBH_interpolator
            (e.g. n_grid). Default is None.

        """

        self.parameter_names = list(priors_dictionary.keys())
        self.priors = self.set_priors(priors_dictionary)
        self.get_PBH_abundance = get_PBH_abundance
        self.check_PBH_abundance = check_PBH_abundance
        self.use_PBH_interpolator = use_PBH_interpolator
        self.PBH_parameter_names = PBH_parameter_names

        if use_PBH_interpolator and check_PBH_abundance and get_PBH_abundance:
            # Interpolator (see build_f_PBH_interpolator) built once here
            # to speed up the many PBH abundance evaluations the sampler
            # will make during the run.
            self.get_PBH_abundance = get_PBH_abundance_from_interpolator(
                self.parameter_names,
                PBH_parameter_names,
                priors_dictionary,
                **(PBH_interpolator_kwargs or {}),
            )

    def set_priors(self, priors_dictionary):
        """

        Set the prior probability density functions from a dictionary.
        The keys of the dictionary should be the parameter names and the values
        should either be callables or dictionaries. If dictionaries, the keys
        should be the distribution names (in scipy.stats), and the values should
        be the keyword arguments for the distribution.

        Parameters:
        -----------
        priors_dictionary : dictionary
            Dictionary containing the prior probability density functions.

        Returns:
        --------
        dictionary
            Dictionary containing the prior probability density functions.

        """

        priors = {}

        for key, value in priors_dictionary.items():

            if type(value) is dict:
                for k, v in value.items():
                    priors[key] = {
                        "pdf": getattr(stats, k).pdf,
                        "rvs": getattr(stats, k).rvs,
                        "pdf_kwargs": v,
                    }

            elif type(value) is function_type:
                priors[key] = {
                    "pdf": value,
                    "rvs": None,
                    "pdf_kwargs": {},
                }

        return priors

    def evaluate_log_priors(self, parameters):
        """
        Evaluate the prior probability density functions.

        Parameters:
        -----------
        parameters : dictionary
            Dictionary containing the parameter names and values.

        Returns:
        --------
        jax.numpy.ndarray
            Prior probability density function values.

        """

        log_prior = 0.0

        if self.check_PBH_abundance and self.get_PBH_abundance:
            PBH_abundance = self.get_PBH_abundance(list(parameters.values()))

            if PBH_abundance > 1.0 or jnp.isnan(PBH_abundance):
                return -jnp.inf

        for k, v in parameters.items():
            p = self.priors[k]
            log_prior += jnp.log(p["pdf"](v, **p["pdf_kwargs"]))

        return log_prior
