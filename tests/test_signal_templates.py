# Global
import unittest
import jax
import jax.numpy as jnp

# Local
import utils as tu
from fastPTA.signals import get_signal_model, get_template, Signal_model

# Define test parameters for different models
FLAT_PARAMS = jnp.array([-7.0])
POWER_LAW_PARAMS = jnp.array([-7.1995, 2])
BPL_PARAMS = jnp.array([-5.8, -8.0, 3, 1.5])
LN_PARAMS = jnp.array([-3, -8.0, -1.5])
SMBH_FLAT_PARAMS = jnp.array([-7.1995, 2, -7.0])
SMBH_LN_PARAMS = jnp.array([-7.1995, 2, -3, -8.0, -1.5])
SMBH_BPL_PARAMS = jnp.array([-7.1995, 2, -5.8, -8.0, 3, 1.5])


@tu.not_a_test
def test_dmodel(model, parameters):
    """
    Given a signal model and parameters, checks the analytical derivatives
    against the numerical derivatives computed with the forward diff.

    Parameters:
    -----------
    model : Signal_model
        The signal model object to test
    parameters : Array
        Parameters for the signal model

    Returns:
    --------
    tuple
        Tuple containing the analytical and numerical derivatives
    """
    fvec = jnp.geomspace(1e-9, 1e-7, 100)

    # Use model's gradient method for analytical derivatives
    gradient = model.gradient(fvec, parameters)

    # Calculate numerical derivatives using forward differences for comparison
    gradientj = jax.jacfwd(model.template, argnums=1)(fvec, parameters)

    return gradient, gradientj


class TestSignals(unittest.TestCase):

    def test_dflat(self):
        """
        Test function for the first derivative of a flat signal model.
        """
        model = get_signal_model("Amplitude")
        gradient, gradientj = test_dmodel(model, FLAT_PARAMS)
        self.assertAlmostEqual(jnp.sum(gradient - gradientj), 0.0, places=5)

    def test_dpower_law(self):
        """
        Test function for the first derivative of a power law signal model.
        """
        model = Signal_model(get_template("PowerLaw", pivot=1e-8))
        gradient, gradientj = test_dmodel(model, POWER_LAW_PARAMS)
        self.assertAlmostEqual(jnp.sum(gradient - gradientj), 0.0, places=5)

    def test_dbroken_power_law(self):
        """
        Test function for the first derivative of a broken power law signal
        model.
        """
        model = get_signal_model("BrokenPowerLawFixedSmoothness")
        gradient, gradientj = test_dmodel(model, BPL_PARAMS)
        self.assertAlmostEqual(jnp.sum(gradient - gradientj), 0.0, places=5)

    def test_dlognormal(self):
        """
        Test function for the first derivative of a lognormal signal model.
        """
        model = get_signal_model("LognormalBump")
        gradient, gradientj = test_dmodel(model, LN_PARAMS)
        self.assertAlmostEqual(jnp.sum(gradient - gradientj), 0.0, places=5)

    def test_dSMBH_and_flat(self):
        """
        Test function for the first derivative of a SMBH + flat signal model.
        """
        model = get_signal_model("SMBHFlat")
        gradient, gradientj = test_dmodel(model, SMBH_FLAT_PARAMS)
        self.assertAlmostEqual(jnp.sum(gradient - gradientj), 0.0, places=5)

    def test_dSMBH_and_lognormal(self):
        """
        Test function for the first derivative of a SMBH + lognormal signal
        model.
        """
        model = get_signal_model("SMBHLognormal")
        gradient, gradientj = test_dmodel(model, SMBH_LN_PARAMS)
        self.assertAlmostEqual(jnp.sum(gradient - gradientj), 0.0, places=5)

    def test_dSMBH_and_broken_power_law(self):
        """
        Test function for the first derivative of a SMBH + broken power law
        signal model.
        """
        model = get_signal_model("SMBHBrokenPowerLaw")
        gradient, gradientj = test_dmodel(model, SMBH_BPL_PARAMS)
        self.assertAlmostEqual(jnp.sum(gradient - gradientj), 0.0, places=5)

    def test_hessian_flat(self):
        """
        Test function for the second derivative of a flat signal model.
        """

        fvec = jnp.geomspace(1e-9, 1e-7, 100)

        flat_model = get_signal_model("Amplitude")

        model = flat_model.template(fvec, FLAT_PARAMS)

        hessian_analytic = model * jnp.log(10) ** 2

        hessian = flat_model.hessian(fvec, FLAT_PARAMS)

        diff_sum = jnp.sum(jnp.abs(hessian[:, 0, 0] - hessian_analytic))
        self.assertAlmostEqual(diff_sum, 0.0, places=5)

    def test_hessian_power_law(self):
        """
        Test function for the second derivative of a power law signal model.
        """

        fvec = jnp.geomspace(1e-9, 1e-7, 100)

        power_law_model = Signal_model(get_template("PowerLaw", pivot=1e-8))

        model = power_law_model.template(fvec, POWER_LAW_PARAMS)

        hessian = power_law_model.hessian(fvec, POWER_LAW_PARAMS)

        hessian_analytic = model[None, None, :] * jnp.array(
            [
                [
                    jnp.log(10) ** 2 * fvec**0,
                    jnp.log(10) * jnp.log(fvec / 1e-8),
                ],
                [
                    jnp.log(10) * jnp.log(fvec / 1e-8),
                    jnp.log(fvec / 1e-8) ** 2,
                ],
            ]
        )

        diff_sum = jnp.sum(jnp.abs(hessian - hessian_analytic.T))
        self.assertAlmostEqual(diff_sum, 0.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
