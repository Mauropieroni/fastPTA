# Global
from scipy import stats
import jax.numpy as jnp

function_type = type(lambda: 0)


class Priors(object):
    """
    Class to define the prior probability density functions.

    """

    def __init__(
        self,
        priors_dictionary,
        get_PBH_abundance=None,
    ):
        """
        Initialize the class.

        Parameters:
        -----------
        priors_dictionary : dictionary
            Dictionary containing the prior probability density functions.

        get_PBH_abundance : callable, optional
            Function to compute the PBH abundance.

        """
        self.parameter_names = list(priors_dictionary.keys())
        self.priors = self.set_priors(priors_dictionary)
        self.get_PBH_abundance = get_PBH_abundance

    def set_priors(self, priors_dictionary):
        """
        Set the prior probability density functions.
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

        if self.get_PBH_abundance is function_type:
            PBH_abundance = self.get_PBH_abundance(list(parameters.values()))

            if PBH_abundance > 1 or jnp.isnan(PBH_abundance):
                return -jnp.inf

        for k, v in parameters.items():
            p = self.priors[k]
            log_prior += jnp.log(p["pdf"](v, **p["pdf_kwargs"]))

        return log_prior
