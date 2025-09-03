# Global
import jax.numpy as jnp

# Local
from fastPTA.signal_templates.flat_template import flat_model
from fastPTA.signal_templates.power_law_template import power_law_model
from fastPTA.signal_templates.broken_power_law_template import bpl_model
from fastPTA.signal_templates.lognormal_template import lognormal_model
from fastPTA.signal_templates.SIGWB_template import SIGWB_model
from fastPTA.signal_templates.SMBH_flat_template import SMBH_flat_model
from fastPTA.signal_templates.SMBH_lognormal_template import (
    SMBH_lognormal_model,
)
from fastPTA.signal_templates.SMBH_broken_power_law_template import (
    SMBH_bpl_model,
)
from fastPTA.signal_templates.SMBH_SIGWB_template import SMBH_SIGWB_model


from fastPTA.compute_PBH_Abundance import f_PBH_NL_QCD_lognormal


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
        - "SMBH_flat":
            Signal model combining SMBH and flat model.
        - "SMBH_lognormal":
            Signal model combining SMBH and log-normal model.
        - "SMBH_broken_power_law":
            Signal model combining SMBH and broken power law model.
        - "SIGW":
            Signal model for scalar induce GW
        - "power_law_SIGW";
            Signal model combining SMBH and SIGW

    """

    if signal_label == "flat":
        signal_model = flat_model

    elif signal_label == "power_law":
        signal_model = power_law_model

    elif signal_label == "lognormal":
        signal_model = lognormal_model

    elif signal_label == "bpl":
        signal_model = bpl_model

    elif signal_label == "SMBH_flat":
        signal_model = SMBH_flat_model

    elif signal_label == "SMBH_lognormal":
        signal_model = SMBH_lognormal_model

    elif signal_label == "SMBH_broken_power_law":
        signal_model = SMBH_bpl_model

    elif signal_label == "SIGW":
        signal_model = SIGWB_model

        def f_PBH_wrapper(parameters):
            """
            Wrapper for the function to compute the PBH abundance.

            Parameters:
            -----------
            parameters : Array
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

        SIGWB_model.get_PBH_abundance = f_PBH_wrapper

    elif signal_label == "power_law_SIGW":
        signal_model = SMBH_SIGWB_model

        # wrapper for the function to compute the PBH abundance
        def f_PBH_wrapper(parameters):
            """
            Wrapper for the function to compute the PBH abundance.

            Parameters:
            -----------
            parameters : Array
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

        SMBH_SIGWB_model.get_PBH_abundance = f_PBH_wrapper

    else:
        raise ValueError("Cannot use", signal_label)

    return signal_model
