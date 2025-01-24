# Global
import jax.numpy as jnp
from jax import jacfwd

# Local
import fastPTA.signal_utils as s_ut
from fastPTA.Compute_PBH_Abundance import f_PBH_NL_QCD


# Current SMBBH SGWB log_amplitude best-fit
SMBBH_log_amplitude = -7.1995
SMBBH_tilt = 2

# Current SMBBH SGWB parameters
SMBBH_parameters = jnp.array([SMBBH_log_amplitude, SMBBH_tilt])

# A value for a flat spectrum
CGW_flat_parameters = jnp.array([-7.0])

# Some values for a LN spectrum
LN_log_amplitude = -6.45167492
LN_log_width = -0.91240383
LN_log_pivot = -7.50455732
CGW_LN_parameters = jnp.array([LN_log_amplitude, LN_log_width, LN_log_pivot])

# Some values for a BPL spectrum
BPL_log_amplitude = -5.8
BPL_log_width = -8.0
BPL_tilt_1 = 3
BPL_tilt_2 = 1.5
CGW_BPL_parameters = jnp.array(
    [BPL_log_amplitude, BPL_log_width, BPL_tilt_1, BPL_tilt_2]
)

# Some values for a Tanh spectrum
Tanh_log_amplitude = -8.0
Tanh_tilt = 8.0
CGW_Tanh_parameters = jnp.array([Tanh_log_amplitude, Tanh_tilt])

# Some values for a SIGW spectrum
SIGW_log_amplitude = -2
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
        self.model_name = model_name
        self.parameter_names = parameter_names
        self.parameter_labels = parameter_labels

        self.template = template
        self.d1 = self.dtemplate_forward if dtemplate is None else dtemplate
        self.d2 = self.d2template_forward if d2template is None else d2template

    @property
    def Nparams(self):
        return len(self.parameter_names.keys())

    def dtemplate_forward(self, frequency, parameters, **kwargs):
        """
        Compute the derivative of the signal model using auto diff.

        Parameters:
        -----------
        frequency : numpy.ndarray or jax.numpy.ndarray
            Array containing frequency bins.
        parameters : numpy.ndarray or jax.numpy.ndarray
            Array containing parameters for the signal model.
        kwargs : dict
            Additional keyword arguments to pass to the signal model.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            Array containing the computed derivative of the signal model with
            respect to the specified parameter.
        """

        return jacfwd(self.template, argnums=1)(frequency, parameters, **kwargs)

    def d2template_forward(self, frequency, parameters, **kwargs):
        """
        Compute the derivative of the signal model using auto diff.

        Parameters:
        -----------
        frequency : numpy.ndarray or jax.numpy.ndarray
            Array containing frequency bins.
        parameters : numpy.ndarray or jax.numpy.ndarray
            Array containing parameters for the signal model.
        kwargs : dict
            Additional keyword arguments to pass to the signal model.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            Array containing the computed derivative of the signal model with
            respect to the specified parameter.
        """

        return jacfwd(self.d1, argnums=1)(frequency, parameters, **kwargs)


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
        - "power_law_flat":
            Signal model combining SMBH and flat spectrum.
        - "power_law_lognormal":
            Signal model combining SMBH and log-normal spectrum.
        - "bpl":
            Broken power law signal model.
        - "power_law_broken_power_law":
            Signal model combining SMBH and broken power law spectrum.
        - "tanh":
            Signal model with a Tanh.
        - "SIGW":
            Signal model for scalar induce GW
        - "power_law_SIGW";
            Signal model combining SMBH and SIGW

    """

    if signal_label == "flat":

        def dflat(frequency, parameters, *args, **kwargs):
            return s_ut.get_gradient(
                1, s_ut.d1flat, frequency, parameters, *args, **kwargs
            )

        signal_model = Signal_model(
            signal_label,
            s_ut.flat,
            dtemplate=dflat,
            parameter_names=["log_amplitude"],
            parameter_labels=[r"$\alpha$"],
        )

    elif signal_label == "power_law":

        def dpower_law(frequency, parameters, *args, **kwargs):
            return s_ut.get_gradient(
                2, s_ut.d1power_law, frequency, parameters, *args, **kwargs
            )

        signal_model = Signal_model(
            signal_label,
            s_ut.power_law,
            dtemplate=dpower_law,
            parameter_names=["log_amplitude", "tilt"],
            parameter_labels=[r"$\alpha_{\rm PL}$", r"$n_{\rm T}$"],
        )

    elif signal_label == "lognormal":

        def dlognormal(frequency, parameters, **kwargs):
            return s_ut.get_gradient(
                3, s_ut.d1lognormal, frequency, parameters, **kwargs
            )

        signal_model = Signal_model(
            signal_label, s_ut.lognormal, dtemplate=dlognormal
        )

    elif signal_label == "power_law_flat":

        def dSMBH_and_flat(frequency, parameters, **kwargs):
            return s_ut.get_gradient(
                3, s_ut.d1SMBH_and_flat, frequency, parameters, **kwargs
            )

        signal_model = Signal_model(
            signal_label, s_ut.SMBH_and_flat, dtemplate=dSMBH_and_flat
        )

    elif signal_label == "power_law_lognormal":

        def dSMBH_and_lognormal(frequency, parameters, **kwargs):
            return s_ut.get_gradient(
                5, s_ut.d1SMBH_and_lognormal, frequency, parameters, **kwargs
            )

        signal_model = Signal_model(
            signal_label,
            s_ut.SMBH_and_lognormal,
            dtemplate=dSMBH_and_lognormal,
        )

    elif signal_label == "bpl":

        def dbroken_power_law(frequency, parameters, **kwargs):
            return s_ut.get_gradient(
                4, s_ut.d1broken_power_law, frequency, parameters, **kwargs
            )

        signal_model = Signal_model(
            signal_label,
            s_ut.broken_power_law,
            dtemplate=dbroken_power_law,
        )

    elif signal_label == "power_law_broken_power_law":

        def dSMBH_and_broken_power_law(frequency, parameters, **kwargs):
            return s_ut.get_gradient(
                6,
                s_ut.d1SMBH_and_broken_power_law,
                frequency,
                parameters,
                **kwargs
            )

        signal_model = Signal_model(
            signal_label,
            s_ut.SMBH_and_broken_power_law,
            dtemplate=dSMBH_and_broken_power_law,
        )

    elif signal_label == "tanh":
        signal_model = Signal_model(signal_label, s_ut.tanh)

    elif signal_label == "SIGW":
        signal_model = Signal_model(
            signal_label,
            s_ut.SIGW,
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

        def f_PBH_wrapper(parameters):
            return f_PBH_NL_QCD(
                10 ** parameters[0],
                10 ** parameters[1],
                10 ** parameters[2] * 2.0 * jnp.pi / (9.7156e-15),
            )

        signal_model.get_PBH_abundance = f_PBH_wrapper

    elif signal_label == "power_law_SIGW":
        signal_model = Signal_model(
            signal_label,
            s_ut.power_law_SIGW,
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

        def f_PBH_wrapper(parameters):
            return f_PBH_NL_QCD(
                10 ** parameters[2],
                10 ** parameters[3],
                10 ** parameters[4] * 2.0 * jnp.pi / (9.7156e-15),
            )

        signal_model.get_PBH_abundance = f_PBH_wrapper

    else:
        raise ValueError("Cannot use", signal_label)

    return signal_model
