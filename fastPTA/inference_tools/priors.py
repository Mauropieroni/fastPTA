# Global
import jax
from scipy import stats
import jax.numpy as jnp


# Local
import fastPTA.utils as ut


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
    ):
        """
        Initialize the class.

        Parameters:
        -----------
        priors_dictionary : dictionary
            Dictionary containing the prior probability density functions.

        get_PBH_abundance : callable, optional
            Function to compute the PBH abundance. Default is False.

        check_PBH_abundance : bool, optional
            Check the PBH abundance using get_PBH_abundance. Default is True.

        """

        self.parameter_names = list(priors_dictionary.keys())
        self.priors = self.set_priors(priors_dictionary)
        self.get_PBH_abundance = get_PBH_abundance
        self.check_PBH_abundance = check_PBH_abundance

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
