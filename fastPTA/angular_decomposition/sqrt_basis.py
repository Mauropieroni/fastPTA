# Global imports
import numpy as np
from wigners import clebsch_gordan

# Local imports
from fastPTA.angular_decomposition import spherical_harmonics as spha


def sqrt_to_lin_conversion(gLM_grid, l_max_lin=-1, real_basis_input=False):
    """
    Convert the sqrt basis to the linear basis.

    Parameters:
    -----------
    gLM_grid : numpy.ndarray
        Array of sqrt basis coefficients.
    l_max_lin : int, optional
        Maximum ell value for the linear basis. Default is -1.
    real_basis_input : bool, optional
        If True, the input is in the real basis. Default is False.

    Returns:
    --------
    clm_real : numpy.ndarray
        Array of real coefficients in the linear basis.

    """

    # If gLM are in the real basis convert the complex basis
    if real_basis_input:
        gLM_complex = spha.real_to_complex_conversion(gLM_grid)
    else:
        gLM_complex = gLM_grid

    # Get the maximum ell value for the sqrt basis
    l_max_sqrt = spha.get_l_max_complex(gLM_complex)

    # Get the grid of all possible L values for the sqrt basis
    L_grid_all = np.arange(l_max_sqrt + 1)

    # If the maximum ell value for the linear basis is not provided, set it
    if l_max_lin < 0:
        l_max_lin = 2 * l_max_sqrt

    # Get the number of coefficients for the linear basis
    n_coefficients = spha.get_n_coefficients_complex(l_max_lin)

    # Initialize the array for the linear basis
    clm_complex = np.zeros(n_coefficients, dtype=np.cdouble)

    # Get the indexes for the linear and sqrt basis
    l_lin, m_lin, _, _, _ = spha.get_sort_indexes(l_max_lin)
    l_sqrt, m_sqrt, _, _, _ = spha.get_sort_indexes(l_max_sqrt)

    # Compute the sign for the gLM with m < 0
    gLnegM_complex = (-1) ** np.abs(m_sqrt) * np.conj(gLM_complex)

    for ind_linear in range(len(m_lin)):
        # Get the values of ell and m
        m = m_lin[ind_linear]
        ell = l_lin[ind_linear]

        for L1 in L_grid_all:

            # Build a mask using the conditions from the selection rules
            mask_L2 = (np.abs(L1 - L_grid_all) <= ell) * (
                L_grid_all >= ell - L1
            )

            # Run over the L2 allowed by the mask
            for L2 in L_grid_all[mask_L2]:
                # Compute the Clebsch-Gordan coefficient for all ms = 0
                cg0 = clebsch_gordan(L1, 0, L2, 0, ell, 0)

                # If the coefficient is not zero compute the prefactor
                if cg0 != 0.0:
                    prefac = np.sqrt(
                        (2.0 * L1 + 1.0)
                        * (2.0 * L2 + 1.0)
                        / (4.0 * np.pi * (2.0 * ell + 1.0))
                    )

                    # These are all the values of M1 to use
                    M1_grid_all = np.arange(-L1, L1 + 1)

                    # Enforce m +M1 + M2 = 0
                    M2_grid_all = m - M1_grid_all

                    # Check that the values of M2 are consistent with L2
                    mask_M = np.abs(M2_grid_all) <= L2

                    # Apply the mask
                    M1_grid = M1_grid_all[mask_M]
                    M2_grid = M2_grid_all[mask_M]

                    for iM in range(len(M1_grid)):
                        # Get the values of M1 and M2
                        M1 = M1_grid[iM]
                        M2 = M2_grid[iM]

                        # Compute the Clebsch-Gordan coefficient for ms neq 0
                        cg1 = clebsch_gordan(L1, M1, L2, M2, ell, m)

                        # Mask to get the corresponding value of gLM_complex
                        b1_mask = (l_sqrt == L1) & (m_sqrt == np.abs(M1))
                        b2_mask = (l_sqrt == L2) & (m_sqrt == np.abs(M2))

                        # Get the values of gLM_complex for the given L and M
                        b1 = (
                            gLM_complex[b1_mask]
                            if M1 >= 0
                            else gLnegM_complex[b1_mask]
                        )

                        b2 = (
                            gLM_complex[b2_mask]
                            if M2 >= 0
                            else gLnegM_complex[b2_mask]
                        )

                        # Multiply everything and sum to the right index
                        clm_complex[ind_linear] += prefac * cg0 * cg1 * b1 * b2

    return spha.complex_to_real_conversion(clm_complex)
