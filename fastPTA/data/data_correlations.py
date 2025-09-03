# Global imports
import jax
import jax.numpy as jnp

# Local imports
import fastPTA.data.datastream as gds


@jax.jit
def get_correlation_IJ(s_I):
    """
    Computes the covariance matrix D_IJ for all pulsar pairs at the measured
    frequencies. The covariance is computed as the outer product of the signal
    s_I for each pulsar pair.

    NB: Since s_I respects the reality condition, we will work with positive
    frequencies only. Thus, we return 2 x the real part of the outer product
    to account for the complex conjugate symmetry.


    Parameters:
    -----------
    s_I : Array
        The projected signal for the pulsars I at frequencies f.
        Should have shape (Np, Nfi), where Np is the number of pulsars and
        Nfi is the number of frequencies.

    Returns:
    --------
    D_IJ : Array
        The covariance matrix for all pulsar pairs at the measured frequencies.
        Will have shape (Nfi, Np, Np), where Nfi is the number of frequencies
        and Np is the number of pulsars.

    """

    return 2.0 * jnp.real(jnp.einsum("If,Jf->fIJ", s_I, jnp.conjugate(s_I)))


@jax.jit
def get_D_IJ(
    fi,
    h_tilde,
    distances,
    p_vec,
    theta_k,
    phi_k,
):
    """
    Computes the covariance matrix D_IJ at the measured frequencies fi
    for all pulsar pairs.

    Parameters:
    -----------
    fi : Array
        Array of frequencies at which to compute the signal.
        Should have shape (Nfi,).
    h_tilde : Array
        The GW realization in frequency domain.
        Should have shape (2, Npix, Nff), where 2 are the polarizations.
    distances : Array
        Distances to the pulsars in meters.
        Should have shape (Np,).
    p_vec : Array
        Unit vectors pointing to the pulsars (works for 1 pulsar too).
        Should have shape (..., 3).
    theta_k : Array
        Polar angles (co-latitudes) of the pixel directions in radians.
        Should have shape (..., Npix).
    phi_k : Array
        Azimuthal angles (longitudes) of the pixel directions in radians.
        Should have shape (..., Npix).

    Returns:
    --------
    D_IJ : Array
        The covariance matrix for the pulsar signals at frequencies fi.
        Will have shape (Nfi, Np, Np), where Np is the number of pulsars.

    """

    # Get the response functions for the pulsars at frequencies fi
    # The shapes of R_p and R_c will be (Np, Npix, Nfi)
    R_p, R_c = gds.get_R_pc(fi, distances, p_vec, theta_k, phi_k)

    # Compute the signal s_I for each pulsar at frequencies fi, shape (Np, Nfi)
    s_I = gds.get_s_I(h_tilde[0], h_tilde[1], R_p, R_c)

    # Compute the tensor product of s_I with its conjugate shape (Nfi, Np, Np)
    # it is computed for positive frequencies, so it is 2x the real part
    D_IJ = get_correlation_IJ(s_I)

    return D_IJ


@jax.jit
def get_D_IJ_fi(
    Tspan,
    fi,
    ff,
    h_tilde,
    distances,
    p_vec,
    theta_k,
    phi_k,
):
    """
    Computes the covariance matrix D_IJ at the measured frequencies fi
    for all pulsar pairs.

    Parameters:
    -----------
    Tspan : float
        The time span of the observation in seconds.
    fi : Array
        Array of frequencies at which to compute the signal.
        Should have shape (Nfi,).
    ff : Array
        Array of internal frequencies for the integration (more dense than fi).
        Should have shape (Nff,).
    h_tilde : Array
        The GW realization in frequency domain.
        Should have shape (2, Npix, Nff), where 2 are the polarizations.
    distances : Array
        Distances to the pulsars in meters.
        Should have shape (Np,).
    p_vec : Array
        Unit vectors pointing to the pulsars (works for 1 pulsar too).
        Should have shape (..., 3).
    theta_k : Array
        Polar angles (co-latitudes) of the pixel directions in radians.
        Should have shape (..., Npix).
    phi_k : Array
        Azimuthal angles (longitudes) of the pixel directions in radians.
        Should have shape (..., Npix).

    Returns:
    --------
    D_IJ : Array
        The covariance matrix for the pulsar signals at frequencies fi.
        Will have shape (Nfi, Np, Np), where Np is the number of pulsars.

    """

    # Get the response functions for the pulsars at frequencies ff
    # The shapes of R_p and R_c will be (Np, Npix, Nff)
    R_p, R_c = gds.get_R_pc(ff, distances, p_vec, theta_k, phi_k)

    # Compute the signal s_I for each pulsar at frequencies fi, shape (Np, Nfi)
    s_I = gds.get_s_I_fi(Tspan, fi, ff, h_tilde[0], h_tilde[1], R_p, R_c)

    # Compute the tensor product of s_I with its conjugate shape (Nfi, Np, Np)
    # it is computed for positive frequencies, so it is 2x the real part
    D_IJ = get_correlation_IJ(s_I)

    return D_IJ


@jax.jit
def get_D_IJ_fi_normalization(Tspan, fi, ff, H_p_ff):
    """
    Computes the normalization factor for the covariance matrix D_IJ to
    recover the Hellings and Downs correlations.

    Parameters:
    -----------
    Tspan : float
        The time span of the observation in seconds.
    fi : Array
        Array of frequencies at which to compute the signal.
        Should have shape (Nfi,).
    ff : Array
        Array of internal frequencies for the integration (more dense than fi).
        Should have shape (Nff,).
    H_p_ff : Array
        The GW power spectrum for the pulsars at frequencies ff.
        Should have shape (Npix, Nff).

    Returns:
    --------
    normalization : Array
        The normalization factor for the covariance matrix D_IJ.
        Will have shape (Nfi,).

    """

    # Compute the sinc functions for the internal (Nff) and external frequencies
    # (Nfi), the shape is (Nfi, Nff)
    sinc_minus = jnp.sinc((fi[:, None] - ff[None, :]) * Tspan)
    sinc_plus = jnp.sinc((fi[:, None] + ff[None, :]) * Tspan)

    # Compute the spectrum by integrating over pixels
    H_fj = 4 * jnp.pi * jnp.mean(H_p_ff, axis=0)

    # Spectrum term, the normalization is as in eq. 21 (or 22) of xxxx.yyyyy.
    spectrum = H_fj / (2.0 * jnp.pi * ff) ** 2

    # Compute the normalization factor for the covariance matrix D_IJ
    normalization = (
        2.0  # 2 polarizations
        * 2.0  # sum positive and negative frequencies no need to take real
        / 3.0
        * jnp.sum(
            (sinc_minus**2 + sinc_plus**2) * spectrum[None, :],
            axis=-1,
        )
    )

    return normalization


@jax.jit
def get_D_IJ_fifj(
    Tspan,
    fi,
    ff,
    h_tilde,
    distances,
    p_vec,
    theta_k,
    phi_k,
):
    """
    Computes the covariance matrix D_IJ at the measured frequencies (fi,fj)
    for all pulsar pairs.

    Parameters:
    -----------
    Tspan : float
        The time span of the observation in seconds.
    fi : Array
        Array of frequencies at which to compute the signal.
        Should have shape (Nfi,).
    ff : Array
        Array of internal frequencies for the integration (more dense than fi).
        Should have shape (Nff,).
    h_tilde : Array
        The GW realization in frequency domain.
        Should have shape (2, Npix, Nff), where 2 are the polarizations.
    distances : Array
        Distances to the pulsars in meters.
        Should have shape (Np,).
    p_vec : Array
        Unit vectors pointing to the pulsars (works for 1 pulsar too).
        Should have shape (..., 3).
    theta_k : Array
        Polar angles (co-latitudes) of the pixel directions in radians.
        Should have shape (..., Npix).
    phi_k : Array
        Azimuthal angles (longitudes) of the pixel directions in radians.
        Should have shape (..., Npix).

    Returns:
    --------
    D_IJ : Array
        The covariance matrix for the pulsar signals at frequencies (fi,fj).
        Will have shape (Nfi, Nfj, Np, Np), where Np is the number of pulsars.

    """

    # Get the response functions for the pulsars at frequencies ff
    # The shapes of R_p and R_c will be (Np, Npix, Nff)
    R_p, R_c = gds.get_R_pc(ff, distances, p_vec, theta_k, phi_k)

    # Compute the signal s_I for each pulsar at frequencies fi, shape (Np, Nfi)
    s_I = gds.get_s_I_fi(Tspan, fi, ff, h_tilde[0], h_tilde[1], R_p, R_c)

    # Compute the tensor product of s_I with its conjugate shape (Nfi, Np, Np)
    D_IJ = jnp.einsum("If,Jg->fgIJ", s_I, jnp.conjugate(s_I))

    # The covariance matrix for positive frequencies only is 2 time the real
    # part of the tensor product
    return 2.0 * jnp.real(D_IJ)


@jax.jit
def get_D_IJ_fifj_normalization(Tspan, fi, ff, H_p_ff):
    """
    Computes the normalization factor for the covariance matrix D_IJ to
    recover the Hellings and Downs correlations.

    Parameters:
    -----------
    Tspan : float
        The time span of the observation in seconds.
    fi : Array
        Array of frequencies at which to compute the signal.
        Should have shape (Nfi,).
    ff : Array
        Array of internal frequencies for the integration (more dense than fi).
        Should have shape (Nff,).
    H_p_ff : Array
        The GW power spectrum for the pulsars at frequencies ff.
        Should have shape (Npix, Nff).

    Returns:
    --------
    normalization : Array
        The normalization factor for the covariance matrix D_IJ.
        Will have shape (Nfi,Nfj).

    """

    # Compute the sinc functions for the internal (Nff) and external frequencies
    # (Nfi), the shape is (Nfi, Nff)
    sinc_minus = jnp.sinc((fi[:, None] - ff[None, :]) * Tspan)
    sinc_plus = jnp.sinc((fi[:, None] + ff[None, :]) * Tspan)

    # Compute the spectrum by integrating over pixels
    H_fj = 4 * jnp.pi * jnp.mean(H_p_ff, axis=0)

    # Spectrum term, the normalization is as in eq. 21 (or 22) of xxxx.yyyyy.
    spectrum = H_fj / (2.0 * jnp.pi * ff) ** 2

    # Compute the normalization factor for the covariance matrix D_IJ
    normalization = (
        2.0  # 2 polarizations
        * 2.0  # sum positive and negative frequencies no need to take real
        / 3.0
        * jnp.sum(
            (
                sinc_minus[:, None] * sinc_minus[None, :]
                + sinc_plus[:, None] * sinc_plus[None, :]
            )
            * spectrum[None, None, :],
            axis=-1,
        )
    )

    return normalization
