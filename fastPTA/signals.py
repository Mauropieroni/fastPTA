r"""
Interface to the gwb_templates package
see https://github.com/Mauropieroni/GWB_templates.
All generic signal templates live there; this module adapts them to
fastPTA's calling convention and adds support for fastPTA-specific local
templates (see fastPTA/local_signal_templates/).
"""

# Global imports
import importlib
import inspect
import pkgutil
import warnings
from typing import Any

import jax.numpy as jnp

from gwb_templates import Template, get_template_from_registry

# Local imports
from fastPTA import utils as ut
from fastPTA.compute_PBH_Abundance import f_PBH_NL_QCD_lognormal

# Current SMBBH SGWB log_amplitude / tilt best-fit and default parameters
SMBBH_log_amplitude = -7.1995
SMBBH_tilt = 2
SMBBH_parameters = jnp.array([SMBBH_log_amplitude, SMBBH_tilt])


# Snapshot of the gwb_templates registry taken before any local template is
# imported. Defining a Template subclass auto-registers it in the shared
# package registry (overwriting on name collision), so this snapshot is what
# lets local templates be tracked separately without ever modifying the
# package registry.
_PACKAGE_REGISTRY: dict[str, type[Template]] = dict(Template._registry)

# Locally defined templates, keyed by class name. Checked before the
# package registry by get_template.
_LOCAL_REGISTRY: dict[str, type[Template]] = {}


def _detach_from_package_registry(name: str) -> None:
    """
    Undo the automatic registration of a local template in the shared
    gwb_templates registry: restore the package class if the name
    shadowed one, or drop the entry if the name is purely local.
    """
    if name in _PACKAGE_REGISTRY:
        Template._registry[name] = _PACKAGE_REGISTRY[name]
    else:
        Template._registry.pop(name, None)


def register_local_template(cls: type[Template]) -> type[Template]:
    """
    Register a locally defined template class so get_template can resolve
    it by class name. Returns the class, so it can be used as a decorator
    on templates defined outside fastPTA/local_signal_templates/ (e.g. in
    a notebook or an analysis script).

    The gwb_templates registry is left untouched: on a name collision
    get_template resolves to the local class, while
    gwb_templates.get_template_from_registry keeps returning the package
    one.
    """
    if not (isinstance(cls, type) and issubclass(cls, Template)):
        raise TypeError(
            f"register_local_template expects a Template subclass, got {cls!r}."
        )
    name = cls.__name__
    if name in _PACKAGE_REGISTRY:
        warnings.warn(
            f"Local template {name!r} shadows the gwb_templates template of "
            "the same name for fastPTA.signals.get_template; the package "
            "registry itself is left untouched.",
            stacklevel=2,
        )
    _LOCAL_REGISTRY[name] = cls
    _detach_from_package_registry(name)
    return cls


def _load_local_templates() -> None:
    """
    Import every module in fastPTA.local_signal_templates and register the
    concrete Template subclasses each one defines.
    """
    lt = importlib.import_module("fastPTA.local_signal_templates")

    for module_info in pkgutil.iter_modules(lt.__path__):
        module = importlib.import_module(f"{lt.__name__}.{module_info.name}")
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, Template)
                and not inspect.isabstract(obj)
                and obj.__module__ == module.__name__
            ):
                register_local_template(obj)


_load_local_templates()


def get_template(name: str, *args: Any, **kwargs: Any) -> Template:
    """
    Instantiate a template by class name, checking locally defined
    templates first (fastPTA/local_signal_templates/ or anything passed to
    register_local_template) and falling back to the gwb_templates
    registry (e.g. "PowerLaw"). Constructor arguments are forwarded, so
    template-level configuration such as the pivot frequency or prior
    overrides is set here:

        get_template("PowerLaw", pivot=ut.f_yr)
        get_template("SIGWB")
    """
    if name in _LOCAL_REGISTRY:
        return _LOCAL_REGISTRY[name](*args, **kwargs)
    if name in Template.registered_templates():
        return get_template_from_registry(name, *args, **kwargs)
    raise ValueError(
        f"Unknown template {name!r}. "
        f"Local templates: {sorted(_LOCAL_REGISTRY)}. "
        f"gwb_templates registry: {sorted(Template.registered_templates())}."
    )


class Signal_model:
    """
    Thin adapter exposing a gwb_templates Template instance through
    fastPTA's calling convention.

    Parameters:
    -----------
    template : Template
        A gwb_templates (or fastPTA local) Template instance.

    Attributes:
    -----------
    model_name : str
        Name of the wrapped template.
    parameter_names : list
        Names of the free parameters, in the order expected by
        template/gradient/hessian.
    parameter_labels : list
        Display labels for the free parameters, same order as
        parameter_names.
    """

    def __init__(self, template: Template):
        self._template = template
        self.model_name = template.model_name
        self.parameter_names = list(template.parameter_names)
        self.parameter_labels = [
            template.parameter_labels[name] for name in template.parameter_names
        ]

    def template(self, frequency, parameters):
        """
        Evaluate the signal model.

        Parameters:
        -----------
        frequency : Array
            Array containing frequency bins.
        parameters : Array
            Array containing parameters for the signal model.

        Returns:
        --------
        Array
            Array containing the computed spectrum.
        """
        return self._template.omega_gw_h2_from_parameters(frequency, parameters)

    def gradient(self, frequency, parameters):
        """
        Compute the gradient of the signal model with respect to its
        parameters.

        Parameters:
        -----------
        frequency : Array
            Array containing frequency bins.
        parameters : Array
            Array containing parameters for the signal model.

        Returns:
        --------
        Array
            Array of shape (len(frequency), Nparams) with the gradient.
        """
        return self._template.grad_theta_omega_gw_h2(frequency, parameters)

    def hessian(self, frequency, parameters):
        """
        Compute the Hessian of the signal model with respect to its
        parameters.

        Parameters:
        -----------
        frequency : Array
            Array containing frequency bins.
        parameters : Array
            Array containing parameters for the signal model.

        Returns:
        --------
        Array
            Array of shape (len(frequency), Nparams, Nparams) with the
            Hessian.
        """
        return self._template.hess_theta_omega_gw_h2(frequency, parameters)

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


def get_signal_model(signal_label):
    """
    Retrieve signal and derivative models based on the specified label.

    Parameters:
    -----------
    signal_label : str
        Name of the template class to use, matching the gwb_templates /
        fastPTA local template registry (see get_template).

    Returns:
    --------
    Signal_model
        Object containing the signal model and its derivatives.

    Notes:
    ------
    Supported signal labels are:
        - "Amplitude":
            Flat signal model.
        - "PowerLaw":
            Power law signal model.
        - "LognormalBump":
            Log-normal signal model.
        - "BrokenPowerLawFixedSmoothness":
            Broken power law signal model.
        - "SMBHFlat":
            Signal model combining SMBH and flat model.
        - "SMBHLognormal":
            Signal model combining SMBH and log-normal model.
        - "SMBHBrokenPowerLaw":
            Signal model combining SMBH and broken power law model.
        - "SIGWB":
            Signal model for scalar induced GW
        - "SMBHSIGWB":
            Signal model combining SMBH and SIGWB

    """

    if signal_label == "PowerLaw":
        # Pin the PTA pivot frequency (gwb_templates defaults to the
        # LISA-band pivot, which is meaningless at nHz frequencies).
        template = get_template(signal_label, pivot=ut.f_yr)
    else:
        template = get_template(signal_label)

    signal_model = Signal_model(template)

    if signal_label == "SIGWB":

        def f_PBH_wrapper(parameters):
            """
            Wrapper for the function to compute the PBH abundance.

            Parameters:
            -----------
            parameters : Array
                Array containing the parameters for the SIGWB model.

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

    elif signal_label == "SMBHSIGWB":

        def f_PBH_wrapper(parameters):
            """
            Wrapper for the function to compute the PBH abundance.

            Parameters:
            -----------
            parameters : Array
                Array containing the parameters for the power law + SIGWB
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

    return signal_model
