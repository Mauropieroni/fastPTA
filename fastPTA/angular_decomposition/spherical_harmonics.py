# Global imports
import functools
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax_healpy as jhp


def _double_factorial_odd(m):
    """
    Compute (2m - 1)!! (with the convention (-1)!! = 1), via the closed form

        ```(2m - 1)!! = (2m)! / (2^m * m!)```

    and jax.scipy.special.factorial. Returns a (possibly traced) jax scalar, so
    it can either be called eagerly or from within a jax.jit trace.

    """

    return jsp.factorial(2 * m) / (2.0**m * jsp.factorial(m))


def _associated_legendre_table(l_max, x, sin_theta):
    """
    Compute the Legendre functions P_l^m(x) (Condon-Shortley phase included, see
    https://en.wikipedia.org/wiki/Spherical_harmonics) for every (l_, m) with
    0 <= m <= l <= l_max, via the standard stable three-term recurrence, stacked
    into a single array. l_max is always a concrete Python int.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.
    x : Array
        cos(theta), any shape.
    sin_theta : Array
        sin(theta), same shape as x.

    Returns:
    --------
    P : Array
        Array with shape (l_max + 1, l_max + 1, *x.shape). The (l, m) entry
        is P_l^m(x) for m <= l, and 0 otherwise.

    """

    P = [[None] * (l_max + 1) for _ in range(l_max + 1)]
    zeros = jnp.zeros_like(x)

    # Compute the Legendre functions via the three-term recurrence relation.
    for m in range(l_max + 1):
        p_mm = ((-1.0) ** m) * _double_factorial_odd(m) * sin_theta**m
        P[m][m] = p_mm

        if m + 1 <= l_max:
            P[m + 1][m] = x * (2 * m + 1) * p_mm

        for ell in range(m + 2, l_max + 1):
            P[ell][m] = (
                x * (2 * ell - 1) * P[ell - 1][m]
                - (ell + m - 1) * P[ell - 2][m]
            ) / (ell - m)

        for ell in range(m):
            P[ell][m] = zeros

    return jnp.stack([jnp.stack(row) for row in P])


@functools.partial(jax.jit, static_argnames=["l_max"])
def _spherical_harmonics_table(l_max, theta, phi):
    """
    Compute the complex spherical harmonics Y_l^m(theta, phi) (orthonormal
    convention, Condon-Shortley phase, matching scipy.special.sph_harm_y) for
    every 0 <= m <= l <= l_max, at every point of theta, phi.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.
    theta : Array
        Polar angle (colatitude), any shape.
    phi : Array
        Azimuthal angle, same shape as theta.

    Returns:
    --------
    Y : Array
        Complex array with shape (l_max + 1, l_max + 1, *theta.shape). The
        (l, m) entry is Y_l^m(theta, phi) for m <= l, and 0 otherwise.

    """

    P = _associated_legendre_table(l_max, jnp.cos(theta), jnp.sin(theta))

    ell_idx = jnp.arange(l_max + 1)
    ell_grid, m_grid = jnp.meshgrid(ell_idx, ell_idx, indexing="ij")
    valid = m_grid <= ell_grid
    diff = jnp.where(valid, ell_grid - m_grid, 0)

    norm = jnp.where(
        valid,
        jnp.sqrt(
            (2 * ell_grid + 1)
            / (4 * jnp.pi)
            * jsp.factorial(diff)
            / jsp.factorial(ell_grid + m_grid)
        ),
        0.0,
    )

    extra_dims = (1,) * theta.ndim
    norm = norm.reshape(norm.shape + extra_dims)
    phase = jnp.exp(1j * m_grid.reshape(m_grid.shape + extra_dims) * phi)

    return norm * P * phase


def sph_harm_y(ell, m, theta, phi):
    """
    Jaxified version of scipy.special.sph_harm_y(ell, m, theta, phi). Supports
    the same broadcasting of ell, m, theta, phi used throughout this module.

    Parameters:
    -----------
    ell : int or Array
        Degree(s).
    m : int or Array
        Order(s).
    theta : float or Array
        Polar angle (colatitude).
    phi : float or Array
        Azimuthal angle.

    Returns:
    --------
    Y : Array
        Complex spherical harmonics, with the broadcast shape of ell, m,
        theta, phi.

    """

    ell, m, theta, phi = jnp.broadcast_arrays(
        jnp.array(ell), jnp.array(m), jnp.array(theta), jnp.array(phi)
    )

    l_max = int(jnp.max(jnp.abs(ell)))
    table = _spherical_harmonics_table(l_max, theta, phi).reshape(
        l_max + 1, l_max + 1, -1
    )

    point_index = jnp.arange(ell.size)
    values = table[ell.reshape(-1), jnp.abs(m).reshape(-1), point_index]

    sign = jnp.where(m.reshape(-1) < 0, (-1.0) ** jnp.abs(m).reshape(-1), 1.0)
    values = jnp.where(m.reshape(-1) < 0, sign * jnp.conj(values), values)

    return values.reshape(ell.shape)


def get_l_max_real(real_spherical_harmonics):
    """
    Given the real spherical harmonics coefficients, this function returns the
    maximum ell value.

    Parameters:
    -----------
    real_spherical_harmonics : Array
        Array of real spherical harmonics coefficients. If dimension is > 1, lm
        must be the first index

    Returns:
    --------
    l_max : int
        Maximum ell value.

    """

    return int(len(real_spherical_harmonics) ** 0.5 - 1)


def get_l_max_complex(complex_spherical_harmonics):
    """
    Given the complex spherical harmonics coefficients, this function returns
    the maximum ell value.

    Parameters:
    -----------
    complex_spherical_harmonics : Array
        Array of complex spherical harmonics coefficients. If dimension is > 1,
        lm must be the first index

    Returns:
    --------
    l_max : int
        Maximum ell value.

    """

    return int((1.0 + 8.0 * len(complex_spherical_harmonics)) ** 0.5 / 2 - 1.5)


def get_n_coefficients_complex(l_max):
    """
    Given the maximum ell value, this function returns the number of spherical
    harmonics coefficients for the complex representation.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.

    Returns:
    --------
    n_coefficients : int
        Number of spherical harmonics coefficients.

    """

    return int((l_max + 1) * (l_max + 2) / 2)


def get_n_coefficients_real(l_max):
    """
    Given the maximum ell value, this function returns the number of spherical
    harmonics coefficients for the real representation.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.

    Returns:
    --------
    n_coefficients : int
        Number of spherical harmonics coefficients.

    """

    return int((l_max + 1) ** 2)


def get_sort_indexes(l_max):
    """
    Given the maximum ell value, this function returns the indexes to sort the
    indexes of the spherical harmonics coefficients when going from real to
    complex representation and viceversa.

    The complex representation is assumed to be sorted as in map2alm of healpy
    (see https://healpy.readthedocs.io/en/latest/), i.e., according to (m, l).
    The ouput allows to sort the real representation according to (l, m).

    Parameters:
    -----------
    l_max : int
        Maximum ell value.

    Returns:
    --------
    l_grid : Array
        Array of l values.
    m_grid : Array
        Array of m values.
    ll : Array
        Array of l values corresponding to the sorted indexes.
    mm : Array
        Array of m values corresponding to the sorted indexes.
    sort_indexes : Array
        Array of indexes to sort the spherical harmonics coefficients.

    """

    # This function only depends on a concrete l_max (never traced data), and
    # its outputs are concrete, statically-sized. ensure_compile_time_eval is
    # used to ensure that even if this function is called from inside a jax.jit
    # trace, the outputs are concrete arrays and not tracers.
    with jax.ensure_compile_time_eval():
        l_values = jnp.arange(l_max + 1)
        m_values = jnp.arange(l_max + 1)

        # Create a grid of all possible (l, m) pairs
        l_grid, m_grid = jnp.meshgrid(l_values, m_values, indexing="xy")

        # Flatten the grid
        l_flat = l_grid.flatten()
        m_flat = m_grid.flatten()

        # Select only the m values that are allowed for a given ell
        l_grid = l_flat[jnp.abs(m_flat) <= l_flat]
        m_grid = m_flat[jnp.abs(m_flat) <= l_flat]

        # Create a vector with all the m<0 and then all the m>=0
        mm = jnp.append(-jnp.flip(m_grid[m_grid > 0]), m_grid)

        # Create a vector with all the ls corresponding to mm
        ll = jnp.append(jnp.flip(l_grid[m_grid > 0]), l_grid)

        # Return the sorted indexes
        return l_grid, m_grid, ll, mm, jnp.lexsort((mm, ll))


@functools.lru_cache(maxsize=None)
def get_projection_matrix(nside, l_max):
    """
    Precomputes the complex spherical harmonics evaluated on every pixel of a
    HEALPix grid of a given Nside, for l, m in the (m, l) order used throughout
    this module and the folding weight (1 for m = 0, 2 for m > 0) used to
    reconstruct a real map from the m >= 0 coefficients via conjugate symmetry.
    Cached to reuse (and not recompute) the same matrices in repeated calls.

    Parameters:
    -----------
    nside : int
        HEALPix Nside parameter.
    l_max : int
        Maximum ell value.

    Returns:
    --------
    spherical_harmonics : Array
        Complex spherical harmonics Y_lm(theta_pix, phi_pix), with shape
        (lm, Npix).
    spherical_harmonics_conj : Array
        Complex conjugate of spherical_harmonics.
    weight : Array
        Folding weight for the m > 0 terms, with shape (lm,).

    """

    # This function's result is cached (by nside, l_max) with a plain
    # functools.lru_cache, not jax.jit. If it were ever invoked for the
    # first time (a cache miss) from inside someone else's active jax.jit
    # trace, the cache would be populated with tracers rather than concrete
    # arrays -- values that are invalid once that trace ends, causing a
    # confusing UnexpectedTracerError on a later, unrelated eager call.
    # ensure_compile_time_eval forces genuine eager evaluation here
    # regardless of the calling context, so the cache only ever holds
    # concrete data.

    # the jax.ensure_compile_time_eval ensures that the code inside is evaluated
    # at compile time, which avoiding tracers issues and allows caching.
    with jax.ensure_compile_time_eval():
        npix = jhp.nside2npix(nside)
        theta, phi = jhp.pix2ang(nside, jnp.arange(npix))
        l_grid, m_grid, _, _, _ = get_sort_indexes(l_max)

        table = _spherical_harmonics_table(l_max, theta, phi)
        spherical_harmonics = table[l_grid, m_grid]
        weight = jnp.where(m_grid == 0, 1.0, 2.0)

        return spherical_harmonics, jnp.conj(spherical_harmonics), weight


@jax.jit
def _project_onto_spherical_harmonics(quantity, spherical_harmonics_conj, npix):
    """
    One-shot (non-iterative) quadrature projection of one or several HEALPix
    maps onto the complex spherical harmonics, using the uniform pixel
    weighting 4 * pi / Npix.

    Parameters:
    -----------
    quantity : Array
        HEALPix map(s), with shape (..., Npix).
    spherical_harmonics_conj : Array
        Complex conjugate spherical harmonics, with shape (lm, Npix).
    npix : int
        Number of pixels in the map.

    Returns:
    --------
    alm : Array
        Complex spherical harmonics coefficients, with shape (..., lm).

    """

    return (4 * jnp.pi / npix) * jnp.einsum(
        "lp,...p->...l", spherical_harmonics_conj, quantity
    )


@jax.jit
def _synthesize_from_spherical_harmonics(alm, spherical_harmonics, weight):
    """
    Synthesize one or several real HEALPix maps from complex spherical
    harmonics coefficients (m >= 0 only), folding in the m < 0 contribution
    through conjugate symmetry.

    Parameters:
    -----------
    alm : Array
        Complex spherical harmonics coefficients, with shape (..., lm).
    spherical_harmonics : Array
        Complex spherical harmonics, with shape (lm, Npix).
    weight : Array
        Folding weight for the m > 0 terms, with shape (lm,).

    Returns:
    --------
    quantity : Array
        HEALPix map(s), with shape (..., Npix).

    """

    return jnp.real(
        jnp.einsum("...l,lp->...p", weight * alm, spherical_harmonics)
    )


def spherical_harmonics_projection(quantity, l_max, n_iter=3):
    """
    Compute the spherical harmonics projection of a given quantity. Quantity
    should be an array (or batch of arrays) in HEALPix pixel space. The
    projection is computed via a quadrature sum against the spherical harmonics
    with uniform pixel weighting 4 * pi / Npix, refined with n_iter Jacobi
    iterations to correct for the HEALPix quadrature bias -- matching the
    algorithm and default iter=3 of healpy's map2alm. The spherical harmonics
    coefficients are sorted as described in the get_sort_indexes function.

    All the helper functions used below are jax.jit-compiled, but this function
    itself is not jax.jit-compiled by default.

    Parameters:
    -----------
    quantity : Array
        Array (or batch of arrays, leading axes) of quantities to project on
        spherical harmonics, with shape (..., Npix).
    l_max : int
        Maximum ell value.
    n_iter : int, optional
        Number of Jacobi refinement iterations. Defaults to 3.

    Returns:
    --------
    real_alm : Array
        Array of real spherical harmonics coefficients, with shape
        (..., lm), where lm = (l_max + 1)**2 is the number of spherical
        harmonics coefficients.

    """

    npix = quantity.shape[-1]
    nside = jhp.npix2nside(npix)
    spherical_harmonics, spherical_harmonics_conj, weight = (
        get_projection_matrix(nside, l_max)
    )

    alm = _project_onto_spherical_harmonics(
        quantity, spherical_harmonics_conj, npix
    )

    for _ in range(n_iter):
        residual = quantity - _synthesize_from_spherical_harmonics(
            alm, spherical_harmonics, weight
        )
        alm = alm + _project_onto_spherical_harmonics(
            residual, spherical_harmonics_conj, npix
        )

    return jnp.moveaxis(
        complex_to_real_conversion(jnp.moveaxis(alm, -1, 0)), 0, -1
    )


def project_correlation_spherical_harmonics(quantity, l_max):
    """
    Compute the spherical harmonics projection of the correlation matrix (in the
    pulsar-pulsar axes) of a given quantity.

    TBD: This function should use the fact that quantity is symmetric in the
    pulsar pulsar indexes to reduce computations and increase efficiency.

    Parameters:
    -----------
    quantity : Array
        3D array to be projected on spherical harmonics. Should have shape
        (N, N, P), where N is the number of pulsars and P is the number of
        HEALPix pixels.
    l_max : int
        Maximum ell value.

    Returns:
    --------
    real_alm : Array
        3D array of real spherical harmonics coefficients.
        It has shape (lm, N, N), where N is the number of pulsars and
        lm = (l_max + 1)**2 is the number of spherical harmonics coefficients.

    """

    # Get the shape of the quantity to project on spherical harmonics
    shape = list(quantity.shape)

    # Reshape quantity so that the batch of maps can be projected in one shot
    qquantity = jnp.reshape(quantity, (int(shape[0] ** 2), shape[-1]))

    # Get all the alms
    real_alm = spherical_harmonics_projection(qquantity, l_max)

    # Reshape to get the same shape as before
    return jnp.reshape(real_alm, (shape[0], shape[0], real_alm.shape[-1])).T


@jax.jit
def complex_to_real_conversion(spherical_harmonics):
    """
    Converts the complex spherical harmonics (or the coefficients) to real
    spherical harmonics (or the coefficients).

    Parameters:
    -----------
    spherical_harmonics : Array
        2D (or 1D) array of complex spherical harmonics coefficients.
        If 2D, the shape is (lm, pp), where lm runs over l,m (with m > 0), and
        pp is the number of theta and phi values. If 1D, the shape is (lm,).

    Returns:
    --------
    all_spherical_harmonics : Array
        2D (or 1D) array of real spherical harmonics coefficients. If 2D, the
        shape is (lm, pp), where lm runs over l,m (with -l <= m <= l), and pp
        is the number of theta and phi values. If 1D, the shape is (lm,).

    """

    # Get the right value of l_max from the input complex coefficients
    l_max = get_l_max_complex(spherical_harmonics)

    # Create the m == 0 / m > 0 boolean masks and the sign for m > 0. These
    # depend on the concrete l_max, but ensure_compile_time_eval keeps them
    # concrete and thus this can be called from inside a jax.jit trace.
    with jax.ensure_compile_time_eval():
        _, m_grid, _, _, sort_indexes = get_sort_indexes(l_max)
        zero_mask = m_grid == 0.0
        positive_mask = m_grid > 0
        sign = (-1.0) ** m_grid[positive_mask]

    # Pick only m = 0
    zero_m = spherical_harmonics[zero_mask].real

    # The m != 0 values are multiplied by sqrt(2) and then take real/imag part
    positive_spherical = jnp.sqrt(2.0) * spherical_harmonics[positive_mask]

    # Build the m > 0 values
    positive_m = jnp.einsum("i,i...->i...", sign, positive_spherical.real)

    # Build the m < 0 values
    negative_m = jnp.einsum("i,i...->i...", sign, positive_spherical.imag)

    # Concatenate the negative, zero and positive m values
    all_spherical_harmonics = jnp.concatenate(
        (jnp.flip(negative_m, axis=0), zero_m, positive_m), axis=0
    )

    # Return spherical harmonics (coefficients) sorted by l and m
    return all_spherical_harmonics[sort_indexes]


@jax.jit
def real_to_complex_conversion(real_spherical_harmonics):
    """
    Converts the real spherical harmonics (or the coefficients) back to complex
    spherical harmonics (or the coefficients).

    Parameters:
    -----------
    real_spherical_harmonics : Array
        1D array of real spherical harmonics coefficients.
        The shape is (lm,), where lm runs over l,m (with -l <= m <= l).
    l_max : int
        Maximum ell value.

    Returns:
    --------
    complex_spherical_harmonics : Array
        1D array of complex spherical harmonics coefficients.
        The shape is (lm,), where lm runs over l,m (with m >= 0).
    """

    # Get the right value of l_max from the input real coefficients
    l_max = get_l_max_real(real_spherical_harmonics)

    # Get sort indexes and the m == 0 / m > 0 / m < 0 boolean masks. As in
    # complex_to_real_conversion ensure_compile_time_eval keeps things concrete.
    with jax.ensure_compile_time_eval():
        _, _, _, mm, sort_indexes = get_sort_indexes(l_max)
        zero_mask = mm == 0
        positive_mask = mm > 0
        negative_mask = mm < 0
        m_positive = mm[positive_mask]

    # Reorder the input real coefficients to the original order
    ordered_real_spherical_harmonics = (
        jnp.zeros_like(real_spherical_harmonics)
        .at[sort_indexes]
        .set(real_spherical_harmonics)
    )

    # Split the ordered real coefficients into negative, zero, and positive
    # m values
    zero_m = ordered_real_spherical_harmonics[zero_mask]
    positive_m = ordered_real_spherical_harmonics[positive_mask]
    negative_m = ordered_real_spherical_harmonics[negative_mask]

    # Reconstruct the complex coefficients
    complex_positive_m = (positive_m + 1j * negative_m[::-1]) / (
        jnp.sqrt(2.0) * (-1.0) ** m_positive
    )

    # Combine zero and positive m values to form the full complex coefficients
    complex_spherical_harmonics = jnp.concatenate(
        (zero_m, complex_positive_m), axis=0
    )

    return complex_spherical_harmonics


def get_real_spherical_harmonics(l_max, theta, phi):
    """
    Compute the real spherical harmonics for a given maximum ell value and for
    a given set of theta and phi values.

    Parameters:
    -----------
    l_max : int
        Maximum ell value.
    theta : Array
        Array of polar angles (co-latitudes).
    phi : Array
        Array of azimuthal angles (longitudes).

    Returns:
    --------
    all_spherical_harmonics : Array
        2D array of spherical harmonics computed for the given maximum ell
        value, and theta and phi values. The shape will be (lm, pp), where pp
        is the number of theta and phi values, and lm = (l_max + 1)**2 is the
        number of spherical harmonics coefficients.

    """

    # Create arrays the m_values and the indexes to sort
    inds = get_sort_indexes(l_max)

    # Unpack m_grid and sorted_indexes
    l_grid = inds[0]
    m_grid = inds[1]

    # Compute all the spherical harmonics
    spherical_harmonics = _spherical_harmonics_table(l_max, theta, phi)[
        l_grid, m_grid
    ]

    # Return sorted
    return complex_to_real_conversion(spherical_harmonics)


def get_map_from_real_clms(clms_real, Nside, l_max=None):
    """
    Get the HEALPix map from the real spherical harmonic coefficients.

    Parameters:
    -----------
    clms_real : Array
        Real spherical harmonic coefficients with shape as described above a
        vector order according to l and m
    Nside : int
        HEALPix Nside parameter.
    l_max : int, optional
        Maximum multipole moment to consider. If None, defaults to the maximum
        value based on the input coefficients.

    Returns:
    --------
    my_map : Array
        HEALPix map with shape (Npix,).
    """

    # If not provided, get the maximum ell value from the input coefficients
    l_max = get_l_max_real(clms_real) if l_max is None else l_max

    # Get the complex spherical harmonics coefficients
    clms_complex = real_to_complex_conversion(clms_real)

    # Convert the complex coefficients to a map
    spherical_harmonics, _, weight = get_projection_matrix(Nside, l_max)

    return _synthesize_from_spherical_harmonics(
        clms_complex, spherical_harmonics, weight
    )


@functools.partial(jax.jit, static_argnames=["l_max"])
def _segment_mean_by_ell(values, l_max):
    """
    Compute, for each ell, the mean over the m-axis segment of an (l, m) ordered
    array (segments of length 2*ell + 1 starting at ell**2, as produced by
    get_sort_indexes), used to build the angular power spectrum (and its
    uncertainty) from real spherical harmonics coefficients.

    Parameters:
    -----------
    values : Array
        Array with shape (lm, ...), where lm = (l_max + 1)**2 runs over l, m.
    l_max : int
        Maximum ell value.

    Returns:
    --------
    means : Array
        Array with shape (l_max + 1, ...), the per-ell mean of values.

    """

    ell = jnp.arange(l_max + 1)
    counts = 2 * ell + 1
    ell_labels = jnp.repeat(ell, counts, total_repeat_length=(l_max + 1) ** 2)
    counts = counts.reshape((-1,) + (1,) * (values.ndim - 1))

    sums = jnp.zeros((l_max + 1,) + values.shape[1:]).at[ell_labels].add(values)

    return sums / counts


@jax.jit
def get_CL_from_real_clm(clm_real):
    """
    Compute the angular power spectrum from the spherical harmonics
    coefficients.

    Parameters:
    -----------
    clm_real : Array
        Array of spherical harmonics coefficients, if dimension > 1 the first
        axis must run over the coefficients.

    Returns:
    --------
    CL : Array
        Array of angular power spectrum.

    """

    l_max = get_l_max_real(clm_real)

    return _segment_mean_by_ell(clm_real**2, l_max)


@jax.jit
def get_dCL_from_real_clm(clm_real, dclm_real):
    """
    Compute the uncertainty on the angular power spectrum from the spherical
    harmonics coefficients using linear error propagation

    Parameters:
    -----------
    clm_real : Array
        Array of spherical harmonics coefficients, if dimension > 1 the first
        axis must run over the coefficients.
    dclm_real : Array
        Array of uncertainties on the spherical harmonics coefficients, must
        have the same shape as clm_real.

    Returns:
    --------
    dCL : Array
        Array of uncertainties on the angular power spectrum.

    """

    l_max = get_l_max_real(clm_real)

    return 2 * _segment_mean_by_ell(jnp.abs(clm_real * dclm_real), l_max)


@functools.partial(jax.jit, static_argnames=["shape_params"])
def _cl_quantile(data, shape_params, limit_cl):
    """
    Compute the quantile of the angular power spectrum (l >= 1) from samples of
    real spherical harmonics coefficients. Fuses the slicing, the CL computation
    and the quantile into a single compiled call, used by get_Cl_limits.

    Parameters:
    -----------
    data : Array
        Samples, with shape (n_points, lm).
    shape_params : int
        Number of leading coefficients excluded from the angular power
        spectrum.
    limit_cl : float
        Quantile to compute.

    Returns:
    --------
    Cl_limit : Array
        Quantile of the angular power spectrum, with shape (l_max,).

    """

    correlations_lm = get_CL_from_real_clm(data.T[shape_params - 1 :])[1:]
    return jnp.quantile(correlations_lm, limit_cl, axis=-1)


@functools.partial(jax.jit, static_argnames=["n_draw", "shape_params"])
def _sample_and_prior_mask(key, means, cov, n_draw, shape_params, prior):
    """
    Draw n_draw samples from a multivariate normal(means, cov) and the boolean
    mask of samples within the flat prior on the coefficients.

    Parameters:
    -----------
    key : Array
        A jax.random PRNG key.
    means : Array
        Mean vector.
    cov : Array
        Covariance matrix.
    n_draw : int
        Number of samples to draw.
    shape_params : int
        Number of leading coefficients excluded from the prior check.
    prior : float
        Prior bound on the coefficients.

    Returns:
    --------
    data : Array
        Samples, with shape (n_draw, len(means)).
    mask : Array
        Boolean array with shape (n_draw,), True where the sample is within
        the prior.

    """

    # Draw samples from the multivariate normal distribution. Uses eigh since
    # the covariance matrix is symmetric and positive semi-definite.
    data = jax.random.multivariate_normal(
        key, means, cov, shape=(n_draw,), method="eigh"
    )
    mask = jnp.max(jnp.abs(data[:, shape_params:]), axis=-1) <= prior

    return data, mask


def get_Cl_limits(
    seed,
    means,
    cov,
    shape_params,
    n_points=int(1e4),
    limit_cl=0.95,
    max_iter=100,
    prior=5.0 / (4.0 * jnp.pi),
):
    """
    Compute the upper limit on the angular power spectrum from the means and
    covariance matrix of the spherical harmonics coefficients.

    Parameters:
    -----------
    seed : int
        Seed for the PRNG key used to generate the samples. Uses the "rbg" PRNG
        implementation rather than jax's default "threefry2x32" since it is
        faster despite being less secure (should be fine for this use case).
    means : Array
        Array of means for the spherical harmonics coefficients.
    cov : Array
        Array of covariance matrix for the spherical harmonics coefficients.
    shape_params : int
        Number of parameters for the SGWB shape.
    n_points : int, optional
        Number of points to generate.
    limit_cl : float, optional
        Quantile to compute the upper limit.
    max_iter : int, optional
        Maximum number of iterations to generate points.
    prior : float, optional
        Prior value to restrict the points.

    Returns:
    --------
    Cl_limits : Array
        Array of upper limits on the angular power spectrum from the covariance
    Cl_limits_prior : Array
        Array of upper limits on the angular power spectrum including the prior

    """

    key = jax.random.key(seed, impl="rbg")

    # Generate gaussian data from the covariance matrix
    key, subkey = jax.random.split(key)
    data, mask = _sample_and_prior_mask(
        subkey, means, cov, n_points, shape_params, prior
    )

    # Select only the points that are within the prior
    data_prior = data[mask]

    # Initialize the counter and the length of the data
    i_add = 0
    len_restricted = len(data_prior)

    # Use a while loop to generate enough points
    while len_restricted < n_points and i_add < max_iter:

        # Generate more points and select those within the prior
        key, subkey = jax.random.split(key)
        add_data, add_mask = _sample_and_prior_mask(
            subkey, means, cov, 10 * n_points, shape_params, prior
        )
        data_prior = jnp.append(data_prior, add_data[add_mask], axis=0)

        # Update the counter and the length of the data
        len_restricted = len(data_prior)
        i_add += 1

    # Compute the upper limits without the prior
    Cl_limits = _cl_quantile(data, shape_params, limit_cl)

    # And with the prior if there are enough points
    if len_restricted == 0:
        Cl_limits_prior = jnp.nan * Cl_limits

    else:
        Cl_limits_prior = _cl_quantile(data_prior, shape_params, limit_cl)

    return Cl_limits, Cl_limits_prior
