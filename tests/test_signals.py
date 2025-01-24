# Global
import unittest

import numpy as np
import jax.numpy as jnp


# Local
import utils as tu
from fastPTA.signals import SMBBH_parameters, get_signal_model


@tu.not_a_test
def test_model(model_name="power_law", parameters=SMBBH_parameters, **kwargs):
    """
    Add some text here here

    """

    fvec = jnp.geomspace(1e-9, 1e-7, 100)
    model = get_signal_model(model_name)

    d1 = model.d1(fvec, parameters, **kwargs)
    d1j = model.dtemplate_forward(fvec, parameters, **kwargs)

    np.allclose(d1, d1j, rtol=1e-5, atol=1e-5)


class TestSignals(unittest.TestCase):

    def test_flat(self):
        """
        Test function for the flat signal model.

        """

        test_model(model_name="flat", parameters=jnp.array([-7.0]))

    def test_power_law(self):
        """
        Test function for the power law signal model.

        """

        test_model(
            model_name="power_law", parameters=SMBBH_parameters, pivot=1e-8
        )

    def test_lognormal(self):
        """
        Test function for the lognormal signal model.

        """

        test_model(
            model_name="lognormal",
            parameters=jnp.array([-3, -1.5, -8.0]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


# --- Power law test
# Check analytical and numerical derivative
# To use grad() we need a function of the parameters


# def function_powerlaw(log_amplitude, tilt, pivot):
#     return 10**log_amplitude * (frequency_point / pivot) ** tilt


# X = np.arange(1, 10, 0.01)
# Y = []
# Ynum = []
# Z = []
# Znum = []
# for i in range(len(X)):
#     Y.append(dpower_law(0, frequency_point, [X[i], SMBBH_tilt], ut.f_yr))
#     Ynum.append(grad(function_powerlaw, argnums=0)(X[i], SMBBH_tilt, ut.f_yr))
#     Z.append(
#         dpower_law(1, frequency_point, [SMBBH_log_amplitude, X[i]], ut.f_yr)
#     )
#     Znum.append(
#         grad(function_powerlaw, argnums=1)(SMBBH_log_amplitude, X[i], ut.f_yr)
#     )
# plt.loglog(X, Y, label="$LogAmplitude - analytic$")
# plt.loglog(
#     X,
#     Ynum,
#     color="yellow",
#     linestyle="dashed",
#     label="$LogAmplitude - numeric$",
# )
# plt.loglog(X, Z, label="$Tilt - analytic$")
# plt.loglog(X, Znum, color="cyan", linestyle="dashed",
# label="$Tilt - numeric$")
# plt.legend(loc="center left", bbox_to_anchor=(0.1, 0.8), prop={"size": 10})
# plt.show()


# --- Lognormal test


# Check analytical and numerical derivative
# To use grad() we need a function of the parameters


# def function_lognormal(log_amplitude, log_width, log_pivot):
#     return 10**log_amplitude * jnp.exp(
#         -0.5 * (jnp.log(frequency_point / (10**log_pivot)) / 10**log_width
# ) ** 2
#     )


# X = np.arange(1, 10, 0.01)
# Y = []
# Ynum = []
# Z = []
# Znum = []
# W = []
# Wnum = []
# for i in range(len(X)):
#     Y.append(
# dlognormal(0, frequency_point, [X[i], LN_log_width, LN_log_pivot]))
#     Ynum.append(
#         grad(function_lognormal, argnums=0)(X[i], LN_log_width, LN_log_pivot)
#     )
#     Z.append(
#         dlognormal(1, frequency_point, [LN_log_amplitude, X[i], LN_log_pivot])
#     )
#     Znum.append(
#         grad(function_lognormal, argnums=1)(
#             LN_log_amplitude, X[i], LN_log_pivot
#         )
#     )
#     W.append(
#         dlognormal(
# 2, frequency_point, [LN_log_amplitude, LN_log_width, -X[i]])
#     )
#     Wnum.append(
#         grad(function_lognormal, argnums=2)(
#             LN_log_amplitude, LN_log_width, -X[i]
#         )
#     )
# plt.loglog(X, Y, label="$LogAmplitude - Analytic$")
# plt.loglog(
#     X,
#     Ynum,
#     color="yellow",
#     linestyle="dashed",
#     label="$LogAmplitude - Numeric$",
# )
# plt.loglog(X, Z, label="$LogWidth - Analytic$")
# plt.loglog(
#     X, Znum, color="cyan", linestyle="dashed", label="$LogWidth - Numeric$"
# )
# plt.legend(loc="center left", bbox_to_anchor=(0.1, 0.2), prop={"size": 10})
# plt.show()
# plt.plot(-X, W, label="$LogPivot - Analytic$")
# plt.plot(
#     -X, Wnum, color="yellow", linestyle="dashed", label="$LogPivot - Numeric$"
# )
# plt.legend(loc="center left", bbox_to_anchor=(0.5, 0.2), prop={"size": 10})
# plt.show()


# Check analytical and numerical derivative
# Already done for the two separate functions


# Check analytical and numerical derivative
# To use grad() we need a function of the parameters

# def function_bpl(alpha, gamma, a, b):

#     smoothing = 1.5
#     x = frequency_point / 10**gamma

#     return (
#         10**alpha
#         * (jnp.abs(a) + jnp.abs(b)) ** smoothing
#         / ((
#             jnp.abs(b) * x ** (-a / smoothing)
#             + jnp.abs(a) * x ** (b / smoothing)
#         )
#         ** smoothing)
#     )

# X = np.arange(1, 10, 0.01)
# Y = []
# Ynum = []
# Z = []
# Znum = []
# W = []
# Wnum = []
# K = []
# Knum = []
# for i in range(len(X)):
#     Y.append(
#         dbroken_power_law(
#             0, frequency_point, [X[i], BPL_log_width, BPL_tilt_1, BPL_tilt_2]
#         )
#     )
#     Ynum.append(
#         grad(function_bpl, argnums=0)(
#             X[i], BPL_log_width, BPL_tilt_1, BPL_tilt_2
#         )
#     )
#     Z.append(
#         dbroken_power_law(
#             1,
#             frequency_point,
#             [BPL_log_amplitude, X[i], BPL_tilt_1, BPL_tilt_2],
#         )
#     )
#     Znum.append(
#         grad(function_bpl, argnums=1)(
#             BPL_log_amplitude, X[i], BPL_tilt_1, BPL_tilt_2
#         )
#     )
#     W.append(
#         dbroken_power_law(
#             2,
#             frequency_point,
#             [BPL_log_amplitude, BPL_log_width, X[i], BPL_tilt_2],
#         )
#     )
#     Wnum.append(
#         grad(function_bpl, argnums=2)(
#             BPL_log_amplitude, BPL_log_width, X[i], BPL_tilt_2
#         )
#     )
#     K.append(
#         dbroken_power_law(
#             3,
#             frequency_point,
#             [BPL_log_amplitude, BPL_log_width, BPL_tilt_1, X[i]],
#         )
#     )
#     Knum.append(
#         grad(function_bpl, argnums=3)(
#             BPL_log_amplitude, BPL_log_width, BPL_tilt_1, X[i]
#         )
#     )
# plt.plot(X, Y, label="$LogAmplitude - Analytic$")
# plt.plot(
#     X,
#     Ynum,
#     color="yellow",
#     linestyle="dashed",
#     label="$LogAmplitude - Numeric$",
# )
# plt.legend(loc="center left", bbox_to_anchor=(0.1, 0.2), prop={"size": 10})
# plt.show()
# plt.plot(X, Z, label="$LogWidth - Analytic$")
# plt.plot(
#     X, Znum, color="cyan", linestyle="dashed", label="$LogWidth - Numeric$"
# )
# plt.legend(loc="center left", bbox_to_anchor=(0.1, 0.2), prop={"size": 10})
# plt.show()
# plt.plot(X, W, label="$Tilt_1 - Analytic$")
# plt.plot(
#     X, Wnum, color="yellow", linestyle="dashed", label="$Tilt_1 - Numeric$"
# )
# plt.plot(X, K, label="$Tilt_2 - Analytic$")
# plt.plot(X, Knum, color="blue", linestyle="dashed",
# label="$Tilt_2 - Numeric$")
# plt.legend(loc="center left", bbox_to_anchor=(0.1, 0.2), prop={"size": 10})
# plt.show()


# --- SMBH + BPL


# Check analytical and numerical derivative
# To use grad() we need a function of the parameters

# def function_bpl(alpha, gamma, a, b):

#     smoothing = 1.5
#     x = frequency_point / 10**gamma

#     return (
#         10**alpha
#         * (jnp.abs(a) + jnp.abs(b)) ** smoothing
#         / ((
#             jnp.abs(b) * x ** (-a / smoothing)
#             + jnp.abs(a) * x ** (b / smoothing)
#         )
#         ** smoothing)
#     )

# X = np.arange(1, 10, 0.01)
# Y = []
# Ynum = []
# Z = []
# Znum = []
# W = []
# Wnum = []
# K = []
# Knum = []
# for i in range(len(X)):
#     Y.append(
#         dbroken_power_law(
#             0, frequency_point, [X[i], BPL_log_width, BPL_tilt_1, BPL_tilt_2]
#         )
#     )
#     Ynum.append(
#         grad(function_bpl, argnums=0)(
#             X[i], BPL_log_width, BPL_tilt_1, BPL_tilt_2
#         )
#     )
#     Z.append(
#         dbroken_power_law(
#             1,
#             frequency_point,
#             [BPL_log_amplitude, X[i], BPL_tilt_1, BPL_tilt_2],
#         )
#     )
#     Znum.append(
#         grad(function_bpl, argnums=1)(
#             BPL_log_amplitude, X[i], BPL_tilt_1, BPL_tilt_2
#         )
#     )
#     W.append(
#         dbroken_power_law(
#             2,
#             frequency_point,
#             [BPL_log_amplitude, BPL_log_width, X[i], BPL_tilt_2],
#         )
#     )
#     Wnum.append(
#         grad(function_bpl, argnums=2)(
#             BPL_log_amplitude, BPL_log_width, X[i], BPL_tilt_2
#         )
#     )
#     K.append(
#         dbroken_power_law(
#             3,
#             frequency_point,
#             [BPL_log_amplitude, BPL_log_width, BPL_tilt_1, X[i]],
#         )
#     )
#     Knum.append(
#         grad(function_bpl, argnums=3)(
#             BPL_log_amplitude, BPL_log_width, BPL_tilt_1, X[i]
#         )
#     )
# plt.plot(X, Y, label="$LogAmplitude - Analytic$")
# plt.plot(
#     X,
#     Ynum,
#     color="yellow",
#     linestyle="dashed",
#     label="$LogAmplitude - Numeric$",
# )
# plt.legend(loc="center left", bbox_to_anchor=(0.1, 0.2), prop={"size": 10})
# plt.show()
# plt.plot(X, Z, label="$LogWidth - Analytic$")
# plt.plot(
#     X, Znum, color="cyan", linestyle="dashed", label="$LogWidth - Numeric$"
# )
# plt.legend(loc="center left", bbox_to_anchor=(0.1, 0.2), prop={"size": 10})
# plt.show()
# plt.plot(X, W, label="$Tilt_1 - Analytic$")
# plt.plot(
#     X, Wnum, color="yellow", linestyle="dashed", label="$Tilt_1 - Numeric$"
# )
# plt.plot(X, K, label="$Tilt_2 - Analytic$")
# plt.plot(X, Knum, color="blue", linestyle="dashed",
# label="$Tilt_2 - Numeric$")
# plt.legend(loc="center left", bbox_to_anchor=(0.1, 0.2), prop={"size": 10})
# plt.show()

# --- PL + SIGW

# Check analytical and numerical derivative
# Already done for the two separate functions

# filedata = np.load('MCMC_data_pl_SIGW_SKA200.npz')
# filechains = np.load('MCMC_chains_pl_SIGW_SKA200.npz')
# frequency = np.logspace(-9, -7, num = 200)
# print(len(frequency))

# dataplSIGW = np.zeros(shape=(200, 5000))
# datapl = np.zeros(shape=(200, 5000))
# dataSIGW = np.zeros(shape=(200, 5000))
# for j in range(5000):
#     y = random.randint(0, len(filechains['samples'])-1)
#     dataplSIGW[:,j] = power_law_SIGW(frequency, filechains['samples'][y][:])
#     datapl[:,j] = power_law(frequency, filechains['samples'][y][:2])
#     dataSIGW[:,j] = SIGW(frequency, filechains['samples'][y][2:])
# quant025plSIGW = np.quantile(dataplSIGW, 0.025, axis=1)
# quant16plSIGW = np.quantile(dataplSIGW, 0.16, axis=1)
# quant84plSIGW = np.quantile(dataplSIGW, 0.84, axis=1)
# quant975plSIGW = np.quantile(dataplSIGW, 0.975, axis=1)

# quant025pl = np.quantile(datapl, 0.025, axis=1)
# quant16pl = np.quantile(datapl, 0.16, axis=1)
# quant84pl = np.quantile(datapl, 0.84, axis=1)
# quant975pl = np.quantile(datapl, 0.975, axis=1)

# quant025SIGW = np.quantile(dataSIGW, 0.025, axis=1)
# quant16SIGW = np.quantile(dataSIGW, 0.16, axis=1)
# quant84SIGW = np.quantile(dataSIGW, 0.84, axis=1)
# quant975SIGW = np.quantile(dataSIGW, 0.975, axis=1)
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.size"] = "18"
# plt.fill_between(
#     frequency, quant025SIGW, quant975SIGW, color="orange", alpha=0.4
# )
# plt.fill_between(
#     frequency, quant16SIGW, quant84SIGW, color="darkorange", alpha=0.4
# )
# plt.loglog(
#     frequency,
#     SIGW(frequency, [-1.5, np.log10(0.5), -7.8]),
#     color="chocolate",
#     alpha=0.8,
# )
# plt.fill_between(
#     frequency, quant025plSIGW, quant975plSIGW, color="mediumpurple", alpha=0.7
# )
# plt.fill_between(
#     frequency, quant16plSIGW, quant84plSIGW, color="rebeccapurple", alpha=0.7
# )
# plt.loglog(
#     frequency,
#     power_law_SIGW(frequency, [-7.1995, 2, -1.5, np.log10(0.5), -7.8]),
#     color="indigo",
#     alpha=0.8,
# )
# plt.fill_between(
#     frequency, quant025pl, quant975pl, color="limegreen", alpha=0.4
# )
# plt.fill_between(
#     frequency, quant16pl, quant84pl, color="forestgreen", alpha=0.4
# )
# plt.loglog(
#     frequency, power_law(frequency, [-7.1995, 2]), color="darkgreen",
# alpha=0.8
# )
# plt.xlabel(r"$f \mathrm{[Hz]}$")
# plt.ylabel(r"$h^2 \Omega_{GW}$")
# plt.ylim((1e-19, 1e-5))
# plt.show()


# --- Tanh test

# def dtanh(index, frequency, parameters, pivot=ut.f_yr):
#     """
#     Derivative of the tanh spectrum.

#     Parameters:
#     -----------
#     index : int
#         Index of the parameter with respect to which the derivative is
# computed.
#     frequency : numpy.ndarray or jax.numpy.ndarray
#         Array containing frequency bins.
#     parameters : numpy.ndarray or jax.numpy.ndarray
#         Array containing parameters for the tanh spectrum.

#     Returns:
#     --------
#     numpy.ndarray or jax.numpy.ndarray
#         Array containing the computed derivative of the tanh spectrum with
#         respect to the specified parameter.
#     """

#     # --- unpack parameters
#     log_amplitude, tilt = parameters

#     model = tanh(frequency, parameters, pivot=ut.f_yr)

#     if index == 0:
#         dlog_model = jnp.log(10)

#     elif index == 1:
#         dlog_model = (
#             jnp.log(1 + jnp.tanh(frequency/pivot))
#         )

#     else:
#         raise ValueError("Cannot use that for this signal")

#     return model * dlog_model

# def function_tanh(frequency, log_amplitude, tilt):
#     return (10**log_amplitude) * (1+jnp.tanh(frequency/ut.f_yr))**tilt

# The following one is if you want ot compute the derivative numerically


# Check analytical and numerical derivative
# To use grad() we need a function of the parameters

# def function_tanh(log_amplitude, tilt):
#     return (10**log_amplitude) * (1+jnp.tanh(frequency_point/ut.f_yr))**tilt

# X = np.arange(1, 10, 0.01)
# Y = []
# Ynum = []
# Z = []
# Znum = []
# for i in range(len(X)):
#     Y.append(dtanh(0, frequency_point, [X[i], Tanh_tilt], ut.f_yr))
#     Ynum.append(grad(function_tanh, argnums=0)(X[i], Tanh_tilt))
#     Z.append(dtanh(1, frequency_point, [Tanh_log_amplitude, X[i]], ut.f_yr))
#     Znum.append(grad(function_tanh, argnums=1)(Tanh_log_amplitude, X[i]))
# plt.loglog(X, Y, label='$LogAmplitude - analytic$')
# plt.loglog(X, Ynum, color='yellow', linestyle='dashed',
# label='$LogAmplitude - numeric$')
# plt.loglog(X, Z, label='$Tilt - analytic$')
# plt.loglog(X,Znum, color='cyan', linestyle='dashed', label='$Tilt - numeric$')
# plt.legend(loc='center left', bbox_to_anchor=(.05, .85), prop={'size': 10})
# plt.show()
