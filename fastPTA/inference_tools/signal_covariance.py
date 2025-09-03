# Global imports
import jax
import jax.numpy as jnp


# Local imports
import fastPTA.utils as ut


@jax.jit
def get_signal_covariance(signal, response_IJ):
    """

    Parameters:
    -----------
    signal : Array
        Array containing signal data.
    response_IJ : Array
        3D array containing response tensor for all the pulsars pairs.
        It has shape (F, N, N), where F is the number of frequencies and N is
        the number of pulsars.

    Returns:
    --------
    Array
        The signal covariance tensor.
    """

    return jnp.einsum("f,fIJ->fIJ", signal, response_IJ)


@jax.jit
def get_covariance_diagonal(signal_lm, gamma_IJ_lm, ff, S_f):
    """
    Compute the diagonal covariance in frequency domain.
    Consistent with eq 21 of 2508.21131.

    Parameters:
    -----------
    signal_lm : Array
        Array containing the lm coefficients of the signal.
    gamma_IJ_lm : Array
        3D array containing the response tensor for all pulsar pairs
        decomposed into spherical harmonics. It has shape (L, N, N) where
        L is the number of spherical harmonic coefficients and N is the
        number of pulsars.
    ff : Array
        Array containing the frequencies.
    S_f : Array
        Array containing the spectral density values at each frequency.

    Returns:
    --------
    Array
        The frequency-diagonal covariance tensor with shape (F, N, N) where
        F is the number of frequencies and N is the number of pulsars.
    """

    # Multiply the spectrum and the response (squared) to get the whole
    # frequency structure
    frequency_part = 4.0 / 3.0 * S_f / (2.0 * jnp.pi * ff) ** 2

    # Contract c_lms with Gamma_lm_IJ to get the full pulsar-pulsar covariance
    IJ_part = jnp.einsum("p,pIJ->IJ", signal_lm, gamma_IJ_lm)

    return jnp.einsum("f,IJ->fIJ", frequency_part, IJ_part)


@jax.jit
def get_dcovariance_diagonal(signal_lm, gamma_IJ_lm, ff, S_f):
    """
    Compute the derivative of diagonal covariance with respect to signal_lm.
    Consistent with eq 21 of 2508.21131.

    Parameters:
    -----------
    signal_lm : Array
        Array containing the lm coefficients of the signal.
    gamma_IJ_lm : Array
        3D array containing the response tensor for all pulsar pairs
        decomposed into spherical harmonics. It has shape (L, N, N) where
        L is the number of spherical harmonic coefficients and N is the
        number of pulsars.
    ff : Array
        Array containing the frequencies.
    S_f : Array
        Array containing the spectral density values at each frequency.

    Returns:
    --------
    Array
        The derivative of frequency-diagonal covariance tensor with respect to
        signal_lm. Has shape (L, F, N, N) where L is the number of spherical
        harmonic coefficients, F is the number of frequencies, and N is the
        number of pulsars.
    """

    # The derivatives of the clms with respect to the signal parameters is
    # just a kroneker delta function
    identity = jnp.eye(len(signal_lm))

    # Multiply the spectrum and the response (squared) to get the whole
    # frequency structure
    frequency_part = 4.0 / 3.0 * S_f / (2.0 * jnp.pi * ff) ** 2

    # Contract the derivatives with Gamma_lm_IJ
    d_IJ_part = jnp.einsum("pq,qIJ->pIJ", identity, gamma_IJ_lm)

    # Contract the derivatives with the frequency part
    return jnp.einsum("f,pIJ->pfIJ", frequency_part, d_IJ_part)


@jax.jit
def get_inverse_covariance_full(signal_lm, gamma_IJ_lm, C_ff):
    """
    Compute the inverse of the full covariance matrix.
    Consistent with eq 21 of 2508.21131.

    Parameters:
    -----------
    signal_lm : Array
        Array containing the lm coefficients of the signal.
    gamma_IJ_lm : Array
        3D array containing the response tensor for all pulsar pairs
        decomposed into spherical harmonics. It has shape (L, N, N) where
        L is the number of spherical harmonic coefficients and N is the
        number of pulsars.
    C_ff : Array
        2D array containing the frequency-frequency correlation matrix.
        It has shape (F, F) where F is the number of frequencies.

    Returns:
    --------
    Array
        The inverse of the full covariance tensor. Has shape (F, F, N, N)
        where F is the number of frequencies and N is the number of pulsars.
    """

    # Invert the frequency part of the covariance
    inv_ff = ut.compute_inverse(C_ff)

    # Contract c_lms with Gamma_lm_IJ to get the full pulsar-pulsar covariance
    IJ_part = jnp.einsum("p,pIJ->IJ", signal_lm, gamma_IJ_lm)

    # Invert the pulsar-pulsar covariance
    inv_IJ_part = ut.compute_inverse(IJ_part)

    # Combine the inverses
    return jnp.einsum("fg,IJ->fgIJ", inv_ff, inv_IJ_part)


@jax.jit
def get_covariance_full(signal_lm, gamma_IJ_lm, C_ff):
    """
    Compute the full covariance matrix including frequency-frequency
    correlations. Consistent with eq 21 of 2508.21131.

    Parameters:
    -----------
    signal_lm : Array
        Array containing the lm coefficients of the signal.
    gamma_IJ_lm : Array
        3D array containing the response tensor for all pulsar pairs
        decomposed into spherical harmonics. It has shape (L, N, N) where
        L is the number of spherical harmonic coefficients and N is the
        number of pulsars.
    C_ff : Array
        2D array containing the frequency-frequency correlation matrix.
        It has shape (F, F) where F is the number of frequencies.

    Returns:
    --------
    Array
        The full covariance tensor. Has shape (F, F, N, N) where F is the
        number of frequencies and N is the number of pulsars.
    """

    # Contract c_lms with Gamma_lm_IJ to get the full pulsar-pulsar covariance
    IJ_part = jnp.einsum("p,pIJ->IJ", signal_lm, gamma_IJ_lm)

    # tensor product with the frequency-frequency correlation matrix
    return jnp.einsum("fg,IJ->fgIJ", C_ff, IJ_part)


@jax.jit
def get_dcovariance_full(signal_lm, gamma_IJ_lm, C_ff):
    """
    Compute the derivative of the full covariance matrix with respect to
    signal_lm. Consistent with eq 21 of 2508.21131.

    Parameters:
    -----------
    signal_lm : Array
        Array containing the lm coefficients of the signal.
    gamma_IJ_lm : Array
        3D array containing the response tensor for all pulsar pairs
        decomposed into spherical harmonics. It has shape (L, N, N) where
        L is the number of spherical harmonic coefficients and N is the
        number of pulsars.
    C_ff : Array
        2D array containing the frequency-frequency correlation matrix.
        It has shape (F, F) where F is the number of frequencies.

    Returns:
    --------
    Array
        The derivative of the full covariance tensor with respect to signal_lm.
        Has shape (L, F, F, N, N) where L is the number of spherical harmonic
        coefficients, F is the number of frequencies, and N is the number of
        pulsars.
    """

    # The derivative of the clms wrt parameters is a Kronecker delta
    identity = jnp.eye(len(signal_lm))

    # The derivative of the pulsar-pulsar covariance with respect to signal_lm
    d_IJ_part = jnp.einsum("pq,qIJ->pIJ", identity, gamma_IJ_lm)

    # Combine the derivatives with the frequency-frequency correlation matrix
    return jnp.einsum("fg,pIJ->pfgIJ", C_ff, d_IJ_part)


@jax.jit
def get_signal_dsignal_tensors_lm_spherical_harmonics_basis(
    signal_lm, signal, dsignal, response_IJ
):
    """
    Compute the signal and its derivatives in the spherical harmonics basis

    Parameters:
    -----------
    signal_lm: Array
        Array containing the lm coefficients.
    signal : Array
        Array containing the signal evaluated at all frequencies.
    dsignal : Array
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, F)
        where P is the number of parameters, F is the number of frequency bins
    response_IJ : Array
        4D array containing response tensor for all the pulsars pairs.
        It has shape (lm, F, N, N), where lm is the number of coefficients for
        the anisotropy decomposition (spherical harmonics or sqrt basis), F is
        the number of frequencies, N is the number of pulsars.

    Returns:
    --------
    Tuple containing:
    - signal_tensor: Array
        the signal tensor in the spherical harmonics basis.
    - dsignal_tensor_frequency_shape: Array
        the derivative of the signal tensor with respect to the signal
        parameters in frequency space.
    - dsignal_tensor_anisotropies: Array
        the derivative of the signal tensor with respect to the lm coefficients

    """

    # Assemble the signal tensor, signal_lm are the lm coefficients of the
    # signal and signal is the signal in frequency space (with len = F)
    signal_lm_f = jnp.einsum("a,f->af", signal_lm, signal)

    # Assemble the signal tensor, the response_IJ tensor has shape
    # (lm, N, pulsars, pulsars)
    signal_tensor = jnp.einsum("afIJ,af->fIJ", response_IJ, signal_lm_f)

    # Derivatives of signal_lm_f in parameters dsignal is the derivative of the
    # signal in frequency space shape (P, N)
    dsignal_lm_f = jnp.einsum("a,pf->paf", signal_lm, dsignal)

    # Assemble the tensor with signal derivatives with respect to the signal
    # parameters, we thus sum over the lm coefficients
    dsignal_tensor_frequency_shape = jnp.einsum(
        "afIJ,paf->pfIJ", response_IJ, dsignal_lm_f
    )

    # Just build a kronecher delta in (P, lm) space
    delta = jnp.eye(len(signal_lm_f))

    # Derivatives of signal_lm_f with respect to the lm coefficients
    dsignal_lm_f_anisotropies = jnp.einsum("pq,f->pqf", delta, signal)

    # Assemble the tensor with signal derivatives with respect to the lm
    # coefficients by contracting dsignal_lm_f_anisotropies with the response
    dsignal_tensor_anisotropies = jnp.einsum(
        "pfIJ,pqf->pfIJ", response_IJ, dsignal_lm_f_anisotropies
    )

    return (
        signal_tensor,
        dsignal_tensor_frequency_shape,
        dsignal_tensor_anisotropies,
    )


@jax.jit
def get_signal_dsignal_tensors_lm_sqrt_basis(
    signal_lm, signal, dsignal, response_IJ
):
    """
    Compute the signal and its derivatives in the sqrt basis

    Parameters:
    -----------
    signal_lm: Array
        Array containing the lm coefficients.
    signal : Array
        Array containing the signal evaluated at all frequencies.
    dsignal : Array
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, F)
        where P is the number of parameters, F is the number of frequency bins
    response_IJ : Array
        4D array containing response tensor for all the pulsars pairs.
        It has shape (lm, N, pulsars, pulsars), where lm is the number of
        coefficients for the anisotropy decomposition (spherical harmonics or
        sqrt basis), N is the number of frequencies, and pulsars is the number
        of pulsars.

    Returns:
    --------
    Tuple containing:
    - signal_tensor: Array
        the signal tensor in the sqrt basis.
    - dsignal_tensor_frequency_shape: Array
        the derivative of the signal tensor with respect to the signal
        parameters in frequency space.
    - dsignal_tensor_anisotropies: Array
        the derivative of the signal tensor with respect to the lm coefficients

    """

    # Assemble the signal tensor, signal_lm are the lm coefficients of the
    # signal and signal is the signal in frequency space (with len = F)
    # response_IJ tensor has shape (lm, lm, F, N, N)
    signal_tensor = jnp.einsum(
        "f,abfde,a,b->fde", signal, response_IJ, signal_lm, signal_lm
    )

    # Assemble the tensor with signal derivatives with respect to the signal
    # parameters, we thus sum over the lm coefficients and multiply by dsignal
    # with axes introduced for pulsars
    dsignal_tensor_frequency_shape = jnp.einsum(
        "pf,abfde,a,b->pfde", dsignal, response_IJ, signal_lm, signal_lm
    )

    # Assemble the tensor with signal derivatives with respect to the lm
    # coefficients. Note that signal_lm enters quadratically in signal_tensor
    dsignal_tensor_anisotropies = jnp.einsum(
        "f,abfde,b->afde", 2.0 * signal, response_IJ, signal_lm
    )

    return (
        signal_tensor,
        dsignal_tensor_frequency_shape,
        dsignal_tensor_anisotropies,
    )


lm_basis_list = [
    get_signal_dsignal_tensors_lm_spherical_harmonics_basis,
    # get_signal_dsignal_tensors_lm_sqrt_basis,
]


lm_basis_map = {
    "spherical_harmonics_basis": 0,
    # "sqrt_basis": 1,
}


@jax.jit
def get_signal_dsignal_tensors_lm(
    lm_basis_idx, signal_lm, signal, dsignal, response_IJ
):
    """
    Compute the signal and its derivatives in the spherical harmonics basis

    Parameters:
    -----------
    lm_basis_idx: int
        Index indicating the basis to use for the anisotropy decomposition
    signal_lm: Array
        Array containing the lm coefficients.
    signal : Array
        Array containing the signal evaluated at all frequencies.
    dsignal : Array
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, F)
        where P is the number of parameters, F is the number of frequency bins
    response_IJ : Array
        4D array containing response tensor for all the pulsars pairs.
        It has shape (lm, F, N, N), where lm is the number of coefficients for
        the anisotropy decomposition (spherical harmonics or sqrt basis), F is
        the number of frequencies, N is the number of pulsars.

    Returns:
    --------
    Tuple containing:
    - signal_tensor: Array
        the signal tensor in the chosen basis.
    - dsignal_tensor_frequency_shape: Array
        the derivative of the signal tensor with respect to the signal
        parameters in frequency space.
    - dsignal_tensor_anisotropies: Array
        the derivative of the signal tensor with respect to the lm coefficients

    """

    return jax.lax.switch(
        lm_basis_idx, lm_basis_list, signal_lm, signal, dsignal, response_IJ
    )
