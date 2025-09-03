# Global
import numpy as np
from scipy.stats import norm

# Local
from fastPTA.utils import (
    save_table,
    default_pulsar_parameters,
    path_to_default_NANOGrav_positions,
)


def generate_distance(
    n_pulsars, loc=3.1356587021094077, scale=0.2579495260515389
):
    """
    Generate distances for a given number of pulsars based on a normal
    distribution. The loc and scale are fitted from the NANOGrav set of pulsars.

    Args:
        n_pulsars (int): Number of pulsars to generate distances for.
        loc (float, optional): Mean of the normal distribution.
            Defaults to 3.1356587021094077.
        scale (float, optional): Standard deviation of the normal distribution.
            Defaults to 0.2579495260515389.

    Returns:
        numpy.ndarray: Array of generated distances in parsecs (pc).
    """

    # Generate log10 of the distance in parsecs
    log10d = norm.rvs(size=n_pulsars, loc=loc, scale=scale)

    # The fit was done for log10 of the distance in parsecs
    return 10**log10d


def generate_parameter(n_pulsars, parameter_dict):
    """
    Generate the parameters for a given number of pulsars based on specified
    info in parameter_dict. Supported distribution types are 'gaussian' and
    'uniform'.

    Parameters:
    -----------
    n_pulsars : int
        Number of pulsars for which parameters will be generated.

    parameter_dict : dict
        Dictionary containing parameters for generating data.

        It should have the following keys:
            - "which_distribution": str, the specific distribution to use.
            - "mean": float, mean value for Gaussian distribution.
            - "std": float, standard deviation for Gaussian distribution.
            - "min": float, minimum value for uniform distribution.
            - "max": float, maximum value for uniform distribution.

    Returns:
    --------
    data : Array
        Generated parameters for the catalog of pulsars.

    """

    if parameter_dict["which_distribution"] == "gaussian":
        data = np.random.normal(
            parameter_dict["mean"], parameter_dict["std"], n_pulsars
        )
    elif parameter_dict["which_distribution"] == "uniform":
        data = np.random.uniform(
            parameter_dict["min"], parameter_dict["max"], n_pulsars
        )
    else:
        raise ValueError(
            "Cannot use that pdf", parameter_dict["which_distribution"]
        )
    return data


def generate_noise_parameter(
    n_pulsars,
    noise_parameters,
    noise_probabilities,
    do_filter=False,
):
    """
    Generate noise parameters for a given number of pulsars based on specified
    distributions. Each noise parameter is generated independently from a
    uniform distribution defined by its min and max values. If do_filter is
    True, noise parameters are filtered based on noise_probabilities.

    Parameters:
    -----------
    n_pulsars : int
        Number of pulsars for which noise parameters needs to be generated.
    noise_parameters : list of dict
        List of dictionaries containing parameters for generating noise data.
        Each dictionary should have keys:
            - "min": float, minimum value for noise parameter.
            - "max": float, maximum value for noise parameter.
    noise_probabilities : Array
        Array of probabilities corresponding to each noise parameter.
    do_filter : bool, optional
        Flag indicating whether to apply filtering to noise parameters,
        default is False.

    Returns:
    --------
    noise_vals : Array
        Generated noise parameters for the pulsars.

    """

    # Unpack minimal and maximal values
    mins = np.array([n["min"] for n in noise_parameters])
    maxs = np.array([n["max"] for n in noise_parameters])

    # Generate the noise parameters
    noise_vals = np.random.uniform(mins, maxs, size=(n_pulsars, len(mins)))

    if do_filter:
        # Generate acceptance probabilities
        acceptance_prob = np.random.uniform(0, 1, size=(n_pulsars, len(mins)))
        noise_prob_filter = np.zeros(shape=(n_pulsars, len(mins)))
        noise_prob_filter[acceptance_prob < noise_probabilities[None, :]] = 1

        # Accepts/rejects depending on the acceptance probabilities
        for i in range(len(noise_prob_filter)):
            if np.all(
                noise_prob_filter[i] == np.zeros(len(noise_prob_filter[i]))
            ):
                val = np.random.uniform(0, 1)

                for i in range(len(mins)):

                    if val < noise_probabilities[0]:
                        noise_prob_filter[i, 0] = 1.0
                        break

        noise_prob_filter = np.log10(noise_prob_filter + 1e-30)
        noise_vals += noise_prob_filter

    return noise_vals


def generate_pulsars_catalog(
    n_pulsars=30,
    dt_dict=default_pulsar_parameters["dt_dict"],
    T_span_dict=default_pulsar_parameters["T_span_dict"],
    wn_dict=default_pulsar_parameters["wn_dict"],
    dm_noise_log_10_A_dict=default_pulsar_parameters["dm_noise_log_10_A_dict"],
    red_noise_log_10_A_dict=default_pulsar_parameters[
        "red_noise_log_10_A_dict"
    ],
    sv_noise_log_10_A_dict=default_pulsar_parameters["sv_noise_log_10_A_dict"],
    dm_noise_g_dict=default_pulsar_parameters["dm_noise_g_dict"],
    red_noise_g_dict=default_pulsar_parameters["red_noise_g_dict"],
    sv_noise_g_dict=default_pulsar_parameters["sv_noise_g_dict"],
    noise_probabilities_dict=default_pulsar_parameters[
        "noise_probabilities_dict"
    ],
    save_catalog=False,
    outname="pulsar_configurations/new_pulsars_catalog.txt",
    use_ng_positions=False,
    verbose=False,
):
    """
    Generate a catalog of pulsars with specified parameters (mostly timing and
    noise characteristics). The generated catalog is returned as a dictionary.
    If save_catalog is True, the generated catalog is saved to the
    specified output file.

    Parameters:
    -----------
    n_pulsars : int, optional
        Number of pulsars to generate in the catalog, default is 30.
    dt_dict : dict, optional
        Dictionary containing parameters for generating pulsar dt.
    T_span_dict : dict, optional
        Dictionary containing parameters for generating pulsar T_span.
    wn_dict : dict, optional
        Dictionary containing parameters for generating pulsar white noise.
    dm_noise_log_10_A_dict : dict, optional
        Dictionary containing parameters for generating log10_A.for DM noise
    red_noise_log_10_A_dict : dict, optional
        Dictionary containing parameters for generating log10_A.for red noise
    sv_noise_log_10_A_dict : dict, optional
        Dictionary containing parameters for generating log10_A for sv noise
    dm_noise_g_dict : dict, optional
        Dictionary containing parameters for generating the tilt for DM noise.
    red_noise_g_dict : dict, optional
        Dictionary containing parameters for generating the tilt for red noise.
    sv_noise_g_dict : dict, optional
        Dictionary containing parameters for generating the tilt for sv noise.
    noise_probabilities_dict : dict, optional
        Dictionary containing noise probabilities.
    save_catalog : bool, optional
        Flag indicating whether to save the generated catalog to a file,
        default is False.
    outname : str, optional
        Name of the output file to save the generated catalog,
        default is "pulsar_configurations/test_pulsars.txt".
    use_ng_positions : bool, optional
        Flag indicating whether to use NANOGrav positions for pulsars,
        default is False.
    verbose : bool, optional
        Flag indicating whether to print verbose output during catalog
        generation, default is False.

    Returns:
    --------
    DataFrame
        Generated catalog of pulsars with specified parameters.

    """

    noise_probabilities = np.array(list(noise_probabilities_dict.values()))
    normed = noise_probabilities / np.sum(noise_probabilities)

    catalog = {}

    if use_ng_positions:
        data_ng = np.loadtxt(
            path_to_default_NANOGrav_positions, skiprows=1, dtype=np.str_
        )
        n_pulsars = len(data_ng)
        catalog["names"] = data_ng[:, 0]
        catalog["phi"] = data_ng[:, 1].astype(float)
        catalog["theta"] = data_ng[:, 2].astype(float)
    else:
        catalog["names"] = [
            "pulsar_" + str(ell) for ell in (1 + np.arange(n_pulsars))
        ]
        catalog["phi"] = np.random.uniform(0.0, 2 * np.pi, n_pulsars)
        catalog["theta"] = np.arccos(np.random.uniform(-1, 1, n_pulsars))

    catalog["dt"] = 10 ** generate_parameter(n_pulsars, dt_dict)
    catalog["Tspan"] = 10 ** generate_parameter(n_pulsars, T_span_dict)
    catalog["wn"] = 10 ** generate_parameter(n_pulsars, wn_dict)

    catalog["log10_A_dm"], catalog["log10_A_red"], catalog["log10_A_sv"] = (
        generate_noise_parameter(
            n_pulsars,
            [
                dm_noise_log_10_A_dict,
                red_noise_log_10_A_dict,
                sv_noise_log_10_A_dict,
            ],
            normed,
            do_filter=True,
        ).T
    )

    catalog["g_dm"], catalog["g_red"], catalog["g_sv"] = (
        generate_noise_parameter(
            n_pulsars,
            [
                dm_noise_g_dict,
                red_noise_g_dict,
                sv_noise_g_dict,
            ],
            normed,
            do_filter=False,
        ).T
    )

    if save_catalog:
        # Save to .txt with headers
        save_table(
            outname,
            catalog,
            verbose=verbose,
        )

    return catalog
