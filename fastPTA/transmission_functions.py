# Global
import jax
import jax.numpy as jnp

# Local
import fastPTA.utils as ut


@jax.jit
def transmission_function_approx(frequencies, T_obs):
    """
    Compute the transmission function (see Eq. 3 of 2404.02864), which
    represents the attenuation of signals, for some frequencies given the
    observation time.

    Parameters:
    -----------
    frequencies : Array
        Array of frequencies (in Hz).
    T_obs : float
        Observation time (in seconds).

    Returns:
    --------
    transmission : Array
        Array of transmission values computed for the given frequencies and
        observation time.

    """

    return 1 / (1 + 1 / (frequencies * T_obs) ** 6)


@jax.jit
def transmission_function_quadratic(frequencies, T_obs):
    """
    Compute the transmission function, which gives the attenuation of signals,
    for a quadratic spindown timing model. The function is based on the
    derivation in Appendix C of 2507.18593, in particular Eq. C5 using C7.

    Parameters:
    -----------
    frequencies : Array
        Array of frequencies (in Hz).
    T_obs : float
        Observation time (in seconds).

    Returns:
    --------
    transmission : Array
        Array of transmission values computed for the given frequencies and
        observation time.

    """

    # Compute dimensionless quantity
    x = frequencies * T_obs

    # Multiply by pi
    pix = jnp.pi * x

    first_term = (jnp.sin(pix) / (pix)) ** 2

    second_term = (
        jnp.sqrt(3)
        / (2 * jnp.pi)
        * (-2 * jnp.sin(pix) / (jnp.pi * x**2) + 2 * jnp.cos(pix) / x)
    ) ** 2

    third_term = (
        jnp.sqrt(5)
        / (-4 * jnp.pi**2)
        * (
            12 * jnp.sin(pix) / (jnp.pi * x**3)
            - 12 * jnp.cos(pix) / x**2
            - 4 * jnp.pi * jnp.sin(pix) / x
        )
    ) ** 2

    return 1.0 - first_term - second_term - third_term


@jax.jit
def transmission_function_quadratic_1yr_peak(frequencies, T_obs):
    """
    Compute the transmission function (see App. C of 2507.18593), which
    gives the attenuation of signals, for a quadratic spindown timing model,
    reproducing the one year peak due to the Earth orbit.

    Parameters:
    -----------
    frequencies : Array
        Array of frequencies (in Hz).
    T_obs : float
        Observation time (in seconds).

    Returns:
    --------
    transmission : Array
        Array of transmission values computed for the given frequencies and
        observation time.

    """

    transmission = transmission_function_quadratic(frequencies, T_obs)

    year_peak = (0.99 * jnp.sinc((frequencies - ut.f_yr) * T_obs)) ** 2

    return transmission - year_peak


@jax.jit
def transmission_function_matrix(f, t, Mmat):
    """
    Compute the transmission function matrix given frequencies, times, and
    design matrix. The transmission function matrix is computed according to
    the procedure described in 1907.04341.

    Parameters:
    -----------
    f : Array
        Array of frequencies (in Hz). Shape (n_frequencies,).
    t : Array
        Array of times (in seconds).
        Shape (n_times,) for a single pulsar, or
        (n_pulsars, n_times) for multiple pulsars.
    Mmat : Array
        Design matrix of the timing model.
        Shape (n_times, n_params) for a single pulsar, or
        (n_pulsars, n_times, n_params) for multiple pulsars.

    Returns:
    --------
    Array
        Transmission function values.
        Shape (n_frequencies,) for a single pulsar, or
        (n_pulsars, n_frequencies) for multiple pulsars.
    """

    N, m = Mmat.shape[-2], Mmat.shape[-1]
    U, _, _ = jnp.linalg.svd(Mmat)
    G = U[..., m:]

    exp = jnp.exp(jnp.einsum("f,...t->...tf", f, 2.0j * jnp.pi * t))

    G_exp = jnp.einsum("...ta,...tf->...taf", G, exp)

    GG = jnp.real(jnp.einsum("...taf,...waf->...f", G_exp, jnp.conj(G_exp))) / N

    return GG
