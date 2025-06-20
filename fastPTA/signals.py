# Global
import jax
from jax import jacfwd
import jax.numpy as jnp

# Local
from fastPTA.utils import which_device
import fastPTA.signal_utils as s_ut
from fastPTA.compute_PBH_Abundance import f_PBH_NL_QCD_lognormal


# Set the device
jax.config.update("jax_default_device", jax.devices(which_device)[0])

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# Current SMBBH SGWB log_amplitude best-fit
SMBBH_log_amplitude = -7.1995
SMBBH_tilt = 2

# Current SMBBH SGWB parameters
SMBBH_parameters = jnp.array([SMBBH_log_amplitude, SMBBH_tilt])

# A value for a flat model
CGW_flat_parameters = jnp.array([-7.0])

# Some values for a LN model
LN_log_amplitude = -6.45167492
LN_log_width = -0.91240383
LN_log_pivot = -7.50455732
CGW_LN_parameters = jnp.array([LN_log_amplitude, LN_log_width, LN_log_pivot])

# Some values for a BPL model
BPL_log_amplitude = -5.8
BPL_log_width = -8.0
BPL_tilt_1 = 3
BPL_tilt_2 = 1.5
CGW_BPL_parameters = jnp.array(
    [BPL_log_amplitude, BPL_log_width, BPL_tilt_1, BPL_tilt_2]
)

# Some values for a Tanh model
Tanh_log_amplitude = -8.0
Tanh_tilt = 8.0
CGW_Tanh_parameters = jnp.array([Tanh_log_amplitude, Tanh_tilt])

# Some values for a SIGW model
SIGW_log_amplitude = -5
SIGW_log_width = jnp.log10(0.5)
SIGW_log_pivot = -8.45
CGW_SIGW_parameters = jnp.array(
    [SIGW_log_amplitude, SIGW_log_width, SIGW_log_pivot]
)


class Signal_model(object):
    """
    Class to define the signal model and its derivatives.

    """

    def __init__(
        self,
        model_name,
        template,
        dtemplate=None,
        d2template=None,
        parameter_names=[],
        parameter_labels=[],
    ):
        """
        Initialize the class.

        Parameters:
        -----------
        model_name : str
            Label indicating the type of signal model.
        template : function
            Signal model.
        dtemplate : function, optional
            Derivative of the signal model.
            If not provided, it will be computed using auto diff.
        d2template : function, optional
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

        self.d1 = self.dtemplate_forward if dtemplate is None else dtemplate
        self.d2 = self.d2template_forward if d2template is None else d2template

    def dtemplate_forward(self, frequency, parameters, *args, **kwargs):
        """
        Compute the derivative of the signal model using automatic
        differentiation.

        Parameters:
        -----------
        frequency : numpy.ndarray or jax.numpy.ndarray
            Array containing frequency bins.
        parameters : numpy.ndarray or jax.numpy.ndarray
            Array containing parameters for the signal model.
        args : tuple
            Additional arguments to pass to the signal model.
        kwargs : dict
            Additional keyword arguments to pass to the signal model.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            Array containing the derivatives of the signal model.
            The shape of the output will be frequency, number of parameters.

        """
        return jacfwd(self.template, argnums=1)(
            frequency, parameters, *args, **kwargs
        )

    def d2template_forward(self, frequency, parameters, *args, **kwargs):
        """
        Compute the second derivative of the signal model using automatic
        differentiation.

        Parameters:
        -----------
        frequency : numpy.ndarray or jax.numpy.ndarray
            Array containing frequency bins.
        parameters : numpy.ndarray or jax.numpy.ndarray
            Array containing parameters for the signal model.
        args : tuple
            Additional arguments to pass to the signal model.
        kwargs : dict
            Additional keyword arguments to pass to the signal model.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            Array containing the second derivatives of the signal model.
            The shape of the output will be frequency, number of parameters.

        """

        return jacfwd(self.d1, argnums=1)(
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

        return len(self.parameter_names.keys())


def get_signal_model(signal_label):
    """
    Retrieve signal and derivative models based on the specified label.

    Parameters:
    -----------
    signal_label : str
        Label indicating the type of signal model.

    Returns:
    --------
    dict
        Dictionary containing the signal model and its derivative model.

    Notes:
    ------
    Supported signal labels are:
        - "flat":
            Flat signal model.
        - "power_law":
            Power law signal model.
        - "lognormal":
            Log-normal signal model.
        - "bpl":
            Broken power law signal model.
        - "power_law_flat":
            Signal model combining SMBH and flat model.
        - "power_law_lognormal":
            Signal model combining SMBH and log-normal model.
        - "power_law_broken_power_law":
            Signal model combining SMBH and broken power law model.
        - "SIGW":
            Signal model for scalar induce GW
        - "power_law_SIGW";
            Signal model combining SMBH and SIGW

    """

    if signal_label == "flat":

        # wrapper for the analytic derivatives of the flat model
        def dflat(frequency, parameters, *args, **kwargs):
            """
            Function to compute the derivatives of the flat model.

            Parameters:
            -----------
            frequency : numpy.ndarray or jax.numpy.ndarray
                Array containing frequency bins.
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing parameters for the flat model.
            args : tuple
                Additional arguments to pass to the flat model.
            kwargs : dict
                Additional keyword arguments to pass to the flat model.

            Returns:
            --------
            numpy.ndarray or jax.numpy.ndarray
                Array containing the derivatives of the flat model.
                The shape of the output will be frequency, number of parameters.

            """
            return s_ut.get_gradient(
                1, s_ut.d1flat, frequency, parameters, *args, **kwargs
            )

        # wrapper for the second derivative of the flat model
        def d2flat(frequency, parameters, *args, **kwargs):
            """
            Function to compute the second derivatives of the flat model.

            Parameters:
            -----------
            frequency : numpy.ndarray or jax.numpy.ndarray
                Array containing frequency bins.
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing parameters for the flat model.
            args : tuple
                Additional arguments to pass to the flat
                model.
            kwargs : dict
                Additional keyword arguments to pass to the flat
                model.

            Returns:
            --------
            numpy.ndarray or jax.numpy.ndarray
                Array containing the second derivatives of the flat model.
                The shape of the output will be frequency, npars, npars.
            """
            return s_ut.get_hessian(
                1, s_ut.d2flat, frequency, parameters, *args, **kwargs
            )

        # Initialize the signal model
        signal_model = Signal_model(
            signal_label,
            s_ut.flat,
            dtemplate=dflat,
            d2template=d2flat,
            parameter_names=["log_amplitude"],
            parameter_labels=[r"$\alpha$"],
        )

    elif signal_label == "power_law":

        # wrapper for the analytic derivatives of the power law model
        def dpower_law(frequency, parameters, *args, **kwargs):
            """
            Function to compute the first derivatives of the power law model.

            Parameters:
            -----------
            frequency : numpy.ndarray or jax.numpy.ndarray
                Array containing frequency bins.
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing parameters for the power law model.
            args : tuple
                Additional arguments to pass to the power law model.
            kwargs : dict
                Additional keyword arguments to pass to the power law model.

            Returns:
            --------
            numpy.ndarray or jax.numpy.ndarray
                Array containing the derivatives of the power law model.
                The shape of the output will be frequency, number of parameters.

            """
            return s_ut.get_gradient(
                2, s_ut.d1power_law, frequency, parameters, *args, **kwargs
            )

        # wrapper for the analytic derivatives of the power law model
        def d2power_law(frequency, parameters, *args, **kwargs):
            """
            Function to compute the second derivatives of the power law model.

            Parameters:
            -----------
            frequency : numpy.ndarray or jax.numpy.ndarray
                Array containing frequency bins.
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing parameters for the power law model.
            args : tuple
                Additional arguments to pass to the power law model.
            kwargs : dict
                Additional keyword arguments to pass to the power law model.

            Returns:
            --------
            numpy.ndarray or jax.numpy.ndarray
                Array containing the second derivatives of the power law model.
                The shape of the output will be frequency, npars, npars.

            """
            return s_ut.get_hessian(
                2, s_ut.d2power_law, frequency, parameters, *args, **kwargs
            )

        # Initialize the signal model
        signal_model = Signal_model(
            signal_label,
            s_ut.power_law,
            dtemplate=dpower_law,
            d2template=d2power_law,
            parameter_names=["log_amplitude", "tilt"],
            parameter_labels=[r"$\alpha_{\rm PL}$", r"$n_{\rm T}$"],
        )

    elif signal_label == "lognormal":

        # wrapper for the analytic derivatives of the log-normal model
        def dlognormal(frequency, parameters, **kwargs):
            """
            Function to compute the derivatives of the log-normal model.

            Parameters:
            -----------
            frequency : numpy.ndarray or jax.numpy.ndarray
                Array containing frequency bins.
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing parameters for the log-normal model.
            kwargs : dict
                Additional keyword arguments to pass to the log-normal model.

            Returns:
            --------
            numpy.ndarray or jax.numpy.ndarray
                Array containing the derivatives of the log-normal model.
                The shape of the output will be frequency, number of parameters.
            """

            return s_ut.get_gradient(
                3, s_ut.d1lognormal, frequency, parameters, **kwargs
            )

        # Initialize the signal model
        signal_model = Signal_model(
            signal_label, s_ut.lognormal, dtemplate=dlognormal
        )

    elif signal_label == "bpl":

        # wrapper for the analytic derivatives of the broken power law model
        def dbroken_power_law(frequency, parameters, **kwargs):
            """
            Function to compute the derivatives of the broken power law
            model.

            Parameters:
            -----------
            frequency : numpy.ndarray or jax.numpy.ndarray
                Array containing frequency bins.
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing parameters for the broken power law model.
            kwargs : dict
                Additional keyword arguments to pass to the broken power law
                model.

            Returns:
            --------
            numpy.ndarray or jax.numpy.ndarray
                Array containing the derivatives of the broken power law
                model.
                The shape of the output will be frequency, number of parameters.
            """

            return s_ut.get_gradient(
                4, s_ut.d1broken_power_law, frequency, parameters, **kwargs
            )

        # Initialize the signal model
        signal_model = Signal_model(
            signal_label,
            s_ut.broken_power_law,
            dtemplate=dbroken_power_law,
        )

    elif signal_label == "power_law_flat":

        # wrapper for the analytic derivatives of the sum of a SMBH and flat
        # model
        def dSMBH_and_flat(frequency, parameters, **kwargs):
            """
            Function to compute the derivatives of the sum of a SMBH and flat
            model.

            Parameters:
            -----------
            frequency : numpy.ndarray or jax.numpy.ndarray
                Array containing frequency bins.
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing parameters for the SMBH + flat model.
            kwargs : dict
                Additional keyword arguments to pass to the SMBH + flat
                model.

            Returns:
            --------
            numpy.ndarray or jax.numpy.ndarray
                Array containing the derivatives of the SMBH + flat model.
                The shape of the output will be frequency, number of parameters.
            """
            return s_ut.get_gradient(
                3, s_ut.d1SMBH_and_flat, frequency, parameters, **kwargs
            )

        # Initialize the signal model
        signal_model = Signal_model(
            signal_label, s_ut.SMBH_and_flat, dtemplate=dSMBH_and_flat
        )

    elif signal_label == "power_law_lognormal":

        # wrapper for the analytic derivatives of the sum of a SMBH and
        # lognormal model
        def dSMBH_and_lognormal(frequency, parameters, **kwargs):
            """
            Function to compute the derivatives of the sum of a SMBH and
            lognormal model.

            Parameters:
            -----------
            frequency : numpy.ndarray or jax.numpy.ndarray
                Array containing frequency bins.
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing parameters for the SMBH + lognormal model.
            kwargs : dict
                Additional keyword arguments to pass to the SMBH
                + lognormal model.

            Returns:
            --------
            numpy.ndarray or jax.numpy.ndarray
                Array containing the derivatives of the SMBH + lognormal
                model.
                The shape of the output will be frequency, number of parameters.
            """
            return s_ut.get_gradient(
                5, s_ut.d1SMBH_and_lognormal, frequency, parameters, **kwargs
            )

        # Initialize the signal model
        signal_model = Signal_model(
            signal_label,
            s_ut.SMBH_and_lognormal,
            dtemplate=dSMBH_and_lognormal,
        )

    elif signal_label == "power_law_broken_power_law":

        # wrapper for the analytic derivatives of the sum of a SMBH and broken
        # power law model
        def dSMBH_and_broken_power_law(frequency, parameters, **kwargs):
            """
            Function to compute the derivatives of the sum of a SMBH and
            broken power law model.

            Parameters:
            -----------
            frequency : numpy.ndarray or jax.numpy.ndarray
                Array containing frequency bins.
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing parameters for the SMBH + broken power law
                model.
            kwargs : dict
                Additional keyword arguments to pass to the SMBH + broken power
                law model.

            Returns:
            --------
            numpy.ndarray or jax.numpy.ndarray
                Array containing the derivatives of the SMBH + broken power law
                model.
                The shape of the output will be frequency, number of parameters.
            """
            return s_ut.get_gradient(
                6,
                s_ut.d1SMBH_and_broken_power_law,
                frequency,
                parameters,
                **kwargs,
            )

        # Initialize the signal model
        signal_model = Signal_model(
            signal_label,
            s_ut.SMBH_and_broken_power_law,
            dtemplate=dSMBH_and_broken_power_law,
        )

    elif signal_label == "SIGW":

        # wrapper for the analytic derivatives of the SIGW model
        signal_model = Signal_model(
            signal_label,
            s_ut.SIGW_broad_approximated,
            parameter_names=[
                "log_amplitude_scalar",
                "log_width",
                "log_pivot",
            ],
            parameter_labels=[
                r"${\rm log}_{10}A_{\zeta}$",
                r"${\rm log}_{10} \Delta$",
                r"${\rm log}_{10} (f_*/{\rm Hz})$",
            ],
        )

        # wrapper for the function to compute the PBH abundance
        def f_PBH_wrapper(parameters):
            """
            Wrapper for the function to compute the PBH abundance.

            Parameters:
            -----------
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing the parameters for the SIGW model.

            Returns:
            --------
            float
                PBH abundance.
            """

            return f_PBH_NL_QCD_lognormal(
                10 ** parameters[0],
                10 ** parameters[1],
                10 ** parameters[2] * 2.0 * jnp.pi / (9.7156e-15),
            )

        signal_model.get_PBH_abundance = f_PBH_wrapper

    elif signal_label == "power_law_SIGW":

        # wrapper for the analytic derivatives of the sum of a power law and
        # SIGW model
        signal_model = Signal_model(
            signal_label,
            s_ut.power_law_SIGW_broad_approximated,
            parameter_names=[
                "log_amplitude_PL",
                "tilt",
                "log_amplitude_scalar",
                "log_width",
                "log_pivot",
            ],
            parameter_labels=[
                r"$\alpha_{\rm PL}$",
                r"$n_{\rm T}$",
                r"${\rm log}_{10}A_{\zeta}$",
                r"${\rm log}_{10} \Delta$",
                r"${\rm log}_{10} (f_*/{\rm Hz})$",
            ],
        )

        # wrapper for the function to compute the PBH abundance
        def f_PBH_wrapper(parameters):
            """
            Wrapper for the function to compute the PBH abundance.

            Parameters:
            -----------
            parameters : numpy.ndarray or jax.numpy.ndarray
                Array containing the parameters for the power law + SIGW
                model.

            Returns:
            --------
            float
                PBH abundance.
            """

            return f_PBH_NL_QCD_lognormal(
                10 ** parameters[2],
                10 ** parameters[3],
                10 ** parameters[4] * 2.0 * jnp.pi / (9.7156e-15),
            )

        signal_model.get_PBH_abundance = f_PBH_wrapper

    else:
        raise ValueError("Cannot use", signal_label)

    return signal_model
