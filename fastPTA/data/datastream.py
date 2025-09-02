# Global imports
import jax
import jax.numpy as jnp

# Local imports
import fastPTA.utils as ut


@jax.jit
def unit_vector(theta, phi):
    """
    Compute the unit vector in 3D Cartesian coordinates given spherical
    coordinates. Theta is the polar angle (co-latitude) and phi is the
    azimuthal angle (longitude). The input angles theta and phi should be given
    in radians.

    Parameters:
    -----------
    theta : Array
        Array of angles in radians representing the polar angle (co-latitude).
    phi : Array
        Array of angles in radians representing the azimuthal angle (longitude).

    Returns:
    --------
    unit_vec : jax.numpy.ndarray
        2D array of unit vectors in 3D Cartesian coordinates corresponding to
        the given spherical coordinates. The shape will be (N, 3), where N is
        the number of unit vectors.

    """

    # Compute the x component of the unit vector
    x_term = jnp.sin(theta) * jnp.cos(phi)

    # Compute the y component of the unit vector
    y_term = jnp.sin(theta) * jnp.sin(phi)

    # Compute the z component of the unit vector
    z_term = jnp.cos(theta)

    # Assemble the unit vector and return it with the right shape
    return jnp.array([x_term, y_term, z_term]).T


@jax.jit
def get_u(theta, phi):
    """
    Computes the derivative of the unit vector with respect to theta. This
    vector is orthogonal to the unit vector defined by unit_vector(theta, phi).

    Parameters:
    -----------
    theta : (Array)
        Array of polar angles (co-latitudes) in radians.
    phi : (Array)
        Array of azimuthal angles (longitudes) in radians.

    Returns:
    --------
    u : jax.numpy.ndarray
        2D array of unit vectors with the derivative of the unit vector with
        respect to theta.
        The shape will be (N, 3), where N is the number of unit vectors.

    """

    # Compute the x component of the unit vector
    x_term = jnp.cos(theta) * jnp.cos(phi)

    # Compute the y component of the unit vector
    y_term = jnp.cos(theta) * jnp.sin(phi)

    # Compute the z component of the unit vector
    z_term = -jnp.sin(theta)

    # Assemble the unit vector and return it with the right shape
    return jnp.array([x_term, y_term, z_term]).T


@jax.jit
def get_v(theta, phi):
    """
    Computes the derivative of the unit vector with respect to phi. This vector
    is orthogonal to the unit vector defined by unit_vector(theta, phi).

    Parameters:
    -----------
    theta : (Array)
        Array of polar angles (co-latitudes) in radians.
    phi : (Array)
        Array of azimuthal angles (longitudes) in radians.

    Returns:
    --------
    v : jax.numpy.ndarray
        2D array of unit vectors with the derivative of the unit vector with
        respect to phi.
        The shape will be (N, 3), where N is the number of unit vectors.

    """

    # Compute the x component of the unit vector
    x_term = -jnp.sin(phi)

    # Compute the y component of the unit vector
    y_term = jnp.cos(phi)

    # Compute the z component of the unit vector
    z_term = 0.0 * phi

    # Assemble the unit vector and return it with the right shape
    return jnp.array([x_term, y_term, z_term]).T


@jax.jit
def get_plus_cross(theta, phi):
    """
    Computes the plus and cross GW polarization tensors given
    the direction(s) on the sky defined by spherical coordinates theta and phi.

    Parameters:
    -----------
    theta : Array
        Array of polar angles (co-latitudes) in radians.

    phi : Array
        Array of azimuthal angles (longitudes) in radians.

    Returns:
    --------
    plus : jax.numpy.ndarray
        3D array of plus polarization tensors.
        The shape will be (..., 3, 3), where ... represents any number of
        leading dimensions, and the 3 are the spatial dimension components.
    cross : jax.numpy.ndarray
        3D array of cross polarization tensors.
        The shape will be (..., 3, 3), where ... represents any number of
        leading dimensions, and the 3 are the spatial dimension components.

    """

    # Get the u and v vectors
    u = get_u(theta, phi)
    v = get_v(theta, phi)

    # Compute the plus polarization tensors = uu - vv
    plus = jnp.einsum("...i,...j->...ij", u, u) - jnp.einsum(
        "...i,...j->...ij", v, v
    )

    # Compute the cross polarization tensors = uv + vu
    cross = jnp.einsum("...i,...j->...ij", u, v) + jnp.einsum(
        "...i,...j->...ij", v, u
    )

    return plus, cross


@jax.jit
def get_F_pc(p_vec, k_vec, e_p_k, e_c_k):
    """
    Computes the pattern function for a set of pulsars and for some directions
    in the sky.

    p_vec is the unit vector pointing to the pulsars
    k_vec is the unit vector specfying some sky direction
    e_p_k is the plus polarization tensor in the sky unit vectors
    e_c_k is the cross polarization tensor in the sky unit vectors

    Parameters:
    -----------
    p_vec: Array
        Unit vectors pointing to the pulsars. Should have shape (..., 3).
    k_vec: Array
        Unit vectors specifying a set of (n_pixel) sky directions.
        Should have shape (n_pixels, 3).
    e_p_k: Array
        Plus polarization tensors for k = k_vec.
        Should have shape (n_pixels, 3, 3).
    e_c_k: Array
        Cross polarization tensors for k = k_vec.
        Should have shape (n_pixels, 3, 3).

    Returns:
    --------
    F_p: jax.numpy.ndarray
        The plus polarization pattern function.
        Will have shape (..., n_pixels).
    F_c: jax.numpy.ndarray
        The cross polarization pattern function.
        Will have shape (..., n_pixels).

    """

    # Tensor product of the pulsar unit vectors
    pp = jnp.einsum("...i,...j->...ij", p_vec, p_vec)

    # Contract the polarization tensors with the pulsar-pulsar tensor
    e_p_pp = jnp.einsum("pij,...ij->...p", e_p_k, pp)
    e_c_pp = jnp.einsum("pij,...ij->...p", e_c_k, pp)

    # Compute the denominator in eq. 9 of 2407.14460
    den = 2.0 * (1.0 + jnp.einsum("pi,...i->...p", k_vec, p_vec))

    # Return the plus and cross polarization pattern functions
    return e_p_pp / den, e_c_pp / den


@jax.jit
def get_R_pc(f_vec, distances, p_vec, theta_k, phi_k):
    """
    Computes the linear response function for a set of pulsars for some given
    sky directions and frequencies. This is eq 9 of 2407.14460.

    Parameters:
    -----------
    f_vec: Array
        Frequency vector.
        Should have shape (n_frequencies).
    distances: Array
        Distances to the pulsars in meters.
        Should have shape (n_pulsars).
    p_vec: Array
        Unit vectors pointing to the pulsars (works for 1 pulsar too).
        Should have shape (..., 3).
    theta_k: Array
        Polar angles (co-latitudes) of the pixel directions in radians.
        Should have shape (..., n_pixels).
    phi_k: Array
        Azimuthal angles (longitudes) of the pixel directions in radians.
        Should have shape (..., n_pixels).

    Returns:
    --------
    R_p_f: jax.numpy.ndarray
        The plus polarization response function.
        Will have shape (..., n_pixels, n_frequencies).
    R_c_f: jax.numpy.ndarray
        The cross polarization response function.
        Will have shape (..., n_pixels, n_frequencies).

    """

    # Compute the unit vector in the direction of the pixels
    k_vec = unit_vector(theta_k, phi_k)

    # Get the plus and cross polarization tensors for the pixel directions
    e_p_k, e_c_k = get_plus_cross(theta_k, phi_k)

    # Get the plus and cross pattern functions, the shapes are (..., n_pixels)
    F_p, F_c = get_F_pc(p_vec, k_vec, e_p_k, e_c_k)

    # Compute the dot product in the exponent
    one_plus = 1.0 + jnp.einsum("pi,...i->...p", k_vec, p_vec)

    # Compute the factor (1 - exponential) in the response function
    exponential = 1 - jnp.exp(
        -2.0j
        * jnp.pi
        * jnp.einsum(
            "f,...,...p->...pf", f_vec, distances / ut.light_speed, one_plus
        )
    )

    # Compute the response function for the plus and cross polarizations
    # The shapes of R_p_f and R_c_f will be (..., n_pixels, n_frequencies)
    R_p_f = jnp.einsum(
        "...,...f,f->...f", F_p, exponential, 1.0 / (2j * jnp.pi * f_vec)
    )
    R_c_f = jnp.einsum(
        "...,...f,f->...f", F_c, exponential, 1.0 / (2j * jnp.pi * f_vec)
    )

    return R_p_f, R_c_f


@jax.jit
def get_s_I(h_p, h_c, R_p, R_c):
    """
    Projects a GWB realization (h_p, h_c) onto the response functions (R_p, R_c)
    to get the signal (s_I) for a set of pulsars. The projection operation
    involves integrating over pixels to get s_I in the frequency domain.

    NB: Reality in time domain imposes s_I(f) = s_I^*(-f)

    Parameters:
    -----------
    h_p : jax.numpy.ndarray
        Plus polarization GW realization in frequency domain.
        Should have shape (Npix, Nfi), where Npix is the number of pixels and
        Nfi is the number of frequencies.
    h_c : jax.numpy.ndarray
        Cross polarization GW realization in frequency domain.
        Should have shape (Npix, Nfi), where Npix is the number of pixels and
        Nfi is the number of frequencies.
    R_p : jax.numpy.ndarray
        Plus polarization response function.
        Should have shape (Np, Npix, Nfi), where Np is the number of pulsars
        and Npix is the number of pixels.
    R_c : jax.numpy.ndarray
        Cross polarization response function.
        Should have shape (Np, Npix, Nfi), where Np is the number of pulsars
        and Npix is the number of pixels.

    Returns:
    --------
    s_I : jax.numpy.ndarray
        The projected signal for the pulsars at (external) frequencies fi.
        Will have shape (Np, Nfi), where Np is the number of pulsars and
        Nfi is the number of external frequencies.

    """

    # Contract GW (Npix, Nfi) and response functions (Np, Npix, Nfi) for plus
    sp = jnp.einsum("p...,Ip...->I...", h_p, R_p)

    # Contract GW (Npix, Nfi) and response functions (Np, Npix, Nfi) for cross
    sc = jnp.einsum("p...,Ip...->I...", h_c, R_c)

    # Combine the plus and cross signals to get the total signal
    return sp + sc


@jax.jit
def get_s_I_fi(Tspan, fi, ff, h_p, h_c, R_p, R_c):
    """
    Projects a GW realization (h_tilde) onto the response functions (R_p, R_c)
    to get the signal (s_I) for a set of pulsars. The projection operation
    involves integrating over the internal frequencies (ff) to get s_I in fi.
    The procedure is consistent with what's written in eq 19 of xxxx.yyyyy.

    Parameters:
    -----------
    Tspan : float
        The time span of the observation in seconds.
    fi : jax.numpy.ndarray
        Array of (external/measured) frequencies at which to compute the signal.
        Should have shape (Nfi,).
    ff : jax.numpy.ndarray
        Array of internal frequencies for the integration (more dense than fi).
        Should have shape (Nff,).
    h_p : jax.numpy.ndarray
        Plus polarization GW realization in frequency domain.
        Should have shape (Npix, Nff), where Npix is the number of pixels and
        Nff is the number of (internal) frequencies.
    h_c : jax.numpy.ndarray
        Cross polarization GW realization in frequency domain.
        Should have shape (Npix, Nff), where Npix is the number of pixels and
        Nff is the number of (internal) frequencies.
    R_p : jax.numpy.ndarray
        Plus polarization response function.
        Should have shape (npulsar, Npix, Nff).
    R_c : jax.numpy.ndarray
        Cross polarization response function.
        Should have shape (npulsar, Npix, Nff).

    Returns:
    --------
    s_I : jax.numpy.ndarray
        The projected signal for the pulsars at frequencies fi.
        Will have shape (Np, Nfi), where Np is the number of pulsars.

    """

    # Compute the sinc functions for the internal (Nff) and external frequencies
    # (Nfi), the shape is (Nfi, Nff)
    sinc_minus = jnp.sinc((fi[:, None] - ff[None, :]) * Tspan)
    sinc_plus = jnp.sinc((fi[:, None] + ff[None, :]) * Tspan)

    # Contract GW (2, Npix, Nff) and response functions (npulsar, Npix, Nff)
    # Since the power used to generate h_tilde was normalized to have the right
    # power in each pixel, the integration over the pixels is just a sum
    h_R_tot = get_s_I(h_p, h_c, R_p, R_c)

    # Convolve h_R_tot (npulsar, Nff) with sinc functions (Nfi, Nff)
    to_int1 = jnp.einsum("If,gf->Igf", h_R_tot, sinc_minus)
    to_int2 = jnp.einsum("If,gf->Igf", jnp.conj(h_R_tot), sinc_plus)

    # Sum over the internal frequencies (Nff) to get the signal for each pulsar
    # at the external frequencies (Nfi), the output has shape (Np, Nfi)
    s_I = jnp.sum((to_int1 - to_int2), axis=-1)

    return s_I
