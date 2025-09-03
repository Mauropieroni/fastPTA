# Global imports
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


# Current SMBBH SGWB log_amplitude best-fit
SMBBH_log_amplitude = -7.1995
SMBBH_tilt = 2

# Current SMBBH SGWB parameters
SMBBH_parameters = jnp.array([SMBBH_log_amplitude, SMBBH_tilt])


def build_gradient(npars, function, frequency, parameters, *args, **kwargs):
    """
    Get the vector of first derivatives of some signal model.

    Parameters:
    -----------
    npars : int
        Number of parameters in the signal model.
    function : callable
        Function to compute the first derivative of the signal model.
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the signal model.
    args : tuple
        Additional arguments to pass to the function.
    kwargs : dictionary
        Additional keyword arguments to pass to the function.

    Returns:
    --------
    Array
        Array containing the computed derivatives of the signal model.
        The shape of the array is (len(frequency), npars).

    """

    return jnp.array(
        [
            function(i, frequency, parameters, *args, **kwargs)
            for i in range(npars)
        ]
    ).T


def build_hessian(npars, function, frequency, parameters, *args, **kwargs):
    """
    Second derivative derivatives of some signal model.

    Parameters:
    -----------
    npars : int
        Number of parameters in the signal model.
    function : callable
        Function to compute the second derivative of the signal model.
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the signal model.
    args : tuple
        Additional arguments to pass to the function.
    kwargs : dictionary
        Additional keyword arguments to pass to the function.

    Returns:
    --------
    Array
        Array containing the second derivative of the spectrum.
        The shape of the array is (len(frequency), npars, npars).

    """

    return jnp.array(
        [
            [
                function(i, j, frequency, parameters, *args, **kwargs)
                for j in range(npars)
            ]
            for i in range(npars)
        ]
    ).T


class Signal_model(object):
    """
    Class to define the signal model and its derivatives.

    """

    def __init__(
        self,
        model_name,
        template,
        dtemplate=None,
        parameter_names=[],
        parameter_labels=[],
    ):
        """
        Initialize the class.

        Parameters:
        -----------
        model_name : str
            Label indicating the type of signal model.
        template : Callable
            Signal model.
        dtemplate : Callable, optional
            Derivative of the signal model.
            If not provided, it will be computed using auto diff.
        d2template : Callable, optional
            Second derivative of the signal model.
            If not provided, it will be computed using auto diff.
        parameter_names : list, optional
            List containing the names of the parameters for the signal model.
        parameter_labels : list, optional
            List containing the labels of the parameters for the signal model.

        """

        self.model_name = model_name
        self.parameter_names = parameter_names
        self.parameter_labels = parameter_labels

        self.template = template
        self.dtemplate = dtemplate

        # Set up the gradient and hessian functions correctly
        if dtemplate is None:
            self.gradient = self.dtemplate_forward
        else:
            # Create a wrapper function that already includes dtemplate
            def gradient_wrapper(frequency, parameters, *args, **kwargs):
                return build_gradient(
                    len(self.parameter_names),
                    self.dtemplate,
                    frequency,
                    parameters,
                    *args,
                    **kwargs,
                )

            self.gradient = gradient_wrapper

        self.hessian = self.d2template_forward

    def dtemplate_forward(self, frequency, parameters, *args, **kwargs):
        """
        Compute the derivative of the signal model using automatic
        differentiation.

        Parameters:
        -----------
        frequency : Array
            Array containing frequency bins.
        parameters : Array
            Array containing parameters for the signal model.
        args : tuple
            Additional arguments to pass to the signal model.
        kwargs : dict
            Additional keyword arguments to pass to the signal model.

        Returns:
        --------
        Array
            Array containing the derivatives of the signal model.
            The shape of the output will be frequency, number of parameters.

        """
        return jax.jacfwd(self.template, argnums=1)(
            frequency, parameters, *args, **kwargs
        )

    def d2template_forward(self, frequency, parameters, *args, **kwargs):
        """
        Compute the second derivative of the signal model using automatic
        differentiation.

        Parameters:
        -----------
        frequency : Array
            Array containing frequency bins.
        parameters : Array
            Array containing parameters for the signal model.
        args : tuple
            Additional arguments to pass to the signal model.
        kwargs : dict
            Additional keyword arguments to pass to the signal model.

        Returns:
        --------
        Array
            Array containing the second derivatives of the signal model.
            The shape of the output will be frequency, number of parameters.

        """

        return jax.jacfwd(self.gradient, argnums=1)(
            frequency, parameters, *args, **kwargs
        )

    @property
    def Nparams(self):
        """
        Number of parameters for the signal model.


        Returns:
        --------
        int
            Number of parameters for the signal model.
        """

        return len(self.parameter_names)
