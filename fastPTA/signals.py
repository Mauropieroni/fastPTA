# Global
import jax.numpy as jnp

# Local
from fastPTA.signal_templates.flat_template import flat_model
from fastPTA.signal_templates.power_law_template import power_law_model
from fastPTA.signal_templates.broken_power_law_template import bpl_model
from fastPTA.signal_templates.lognormal_template import lognormal_model

# Interface to the gwb_templates package. All shipped signal templates
# live there; this module adapts them to the calling convention and adds
# support for locally defined templates (see gwb_lisa/local_templates/).

# Global imports
import importlib
import inspect
import pkgutil
import warnings
from typing import Any

from gwb_templates import Template, get_template_from_registry

# Local imports
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

# Snapshot of the gwb_templates registry taken before any local template
# is imported. Defining a Template subclass auto-registers it in the
# shared package registry (overwriting on name collision), so this
# snapshot is what lets local templates be tracked separately without
# ever modifying the package registry.
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
    Register a locally defined template class so get_template can
    resolve it by class name. Returns the class, so it can be used as a
    decorator on templates defined outside gwb_lisa/local_templates/
    (e.g. in a notebook or an analysis script).

    The gwb_templates registry is left untouched: on a name collision
    get_template resolves to the local class, while
    gwb_templates.get_template_from_registry keeps returning the
    package one.
    """
    if not (isinstance(cls, type) and issubclass(cls, Template)):
        raise TypeError(
            f"register_local_template expects a Template subclass, got {cls!r}."
        )
    name = cls.__name__
    if name in _PACKAGE_REGISTRY:
        warnings.warn(
            f"Local template {name!r} shadows the gwb_templates template of "
            "the same name for gwb_lisa.templates.get_template; the package "
            "registry itself is left untouched.",
            stacklevel=2,
        )
    _LOCAL_REGISTRY[name] = cls
    _detach_from_package_registry(name)
    return cls


def _load_local_templates() -> None:
    """
    Import every module in gwb_lisa.local_templates and register the
    concrete Template subclasses each one defines.
    """
    lt = importlib.import_module("gwb_lisa.local_templates")

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
    templates first (gwb_lisa/local_templates/ or anything passed to
    register_local_template) and falling back to the gwb_templates
    registry (e.g. "PowerLaw"). Constructor arguments are forwarded, so
    template-level configuration such as the pivot frequency or prior
    overrides is set here:

        get_template("PowerLaw", pivot=1e-3)
        get_template("ExtragalacticWd2", prior_by_param={...})
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
