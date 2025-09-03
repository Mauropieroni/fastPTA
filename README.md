<table border="10">
  <tr>
    <td>
      <img src="fast_PTA.png" alt="fast_PTA" width="700"/>
    </td>
    <td>
      <h1>fastPTA</h1>
      <h2>
      A jit-enhanced Python code to forecast the sensitivity of Pulsar Timing Array (PTA) configurations and assess constraints on Stochastic Gravitational Wave Background (SGWB) parameters. 
 </h2>
      <p>
The code can generate mock PTA catalogs with noise levels compatible with current and future PTA experiments.
These catalogs can then be used to perform Fisher forecasts of MCMC simulations.
      </p>
    </td>
  </tr>
</table>

# Installation
- Clone from this repository:
```
https://github.com/Mauropieroni/fastPTA
```
- Install using the following command (you can delete the `fastPTA` folder afterwards).
```
cd fastPTA/
pip install .
``` 
  alternatively, to have changes in the code propagate instantaneously (do not delete the `fastPTA` folder in this case!):
```
cd fastPTA/
pip install -e .
```

# To test:
After installation (see above), you can run the following command
```
pytest $(cd fastPTA/)
```
a series of tests will run to check that everything works fine.

# Repository structure:
```
fastPTA/
├── __init__.py                                   # Package initialization, version, and 
│                                                 # top-level imports
├── compute_PBH_Abundance.py                      # Calculation of primordial black hole 
│                                                 # abundance
├── Fisher_code.py                                # Fisher matrix calculation for parameter 
│                                                 # estimation
├── generate_new_pulsar_configuration.py          # Generation of pulsar configurations for 
│                                                 # simulations
├── get_tensors.py                                # Tensor calculations for PTA correlation 
│                                                 # patterns
├── MCMC_code.py                                  # MCMC sampling methods for posterior 
│                                                 # inference
├── plotting_functions.py                         # Functions for visualizing results and 
│                                                 # diagnostics
├── pulsar_noises.py                              # Models for pulsar intrinsic noise sources
├── signals.py                                    # Signal models for gravitational wave 
│                                                 # backgrounds
├── utils.py                                      # General utility functions for various 
│                                                 # calculations
├── angular_decomposition/                        # Module for angular power decomposition
│   ├── spherical_harmonics.py                    # Spherical harmonics implementation for 
│   │                                             # sky maps
│   └── sqrt_basis.py                             # Square-root basis for positive-definite 
│                                                 # sky maps
├── data/                                         # Data generation and handling utilities
│   ├── data_correlations.py                      # Tools for calculating data correlations
│   ├── datastream.py                             # Data stream handling and processing
│   └── generate_data.py                          # Functions to generate mock PTA data
├── defaults/                                     # Default configuration and parameters
│   ├── cgvals.txt                                # Cosmic growth values for calculations
│   ├── data_f_PBH_MH.txt                         # Data for PBH mass-fraction relationships
│   ├── default_catalog.txt                       # Default pulsar catalog
│   ├── default_pulsar_parameters.yaml            # Default pulsar parameters in YAML format
│   ├── fvals.txt                                 # Frequency values for default calculations
│   ├── gstar_T.txt                               # Temperature-dependent g* values
│   ├── NANOGrav_positions.txt                    # NANOGrav pulsar positions
│   ├── SIGWB_prefactor_data.txt                  # Prefactor data for SIGWB calculations
│   └── T_data.txt                                # Temperature data for various calculations
├── inference_tools/                              # Statistical inference utilities
│   ├── iterative_estimation.py                   # Iterative parameter estimation methods
│   ├── likelihoods.py                            # Likelihood functions for PTA data
│   ├── priors.py                                 # Prior distributions for Bayesian inference
│   └── signal_covariance.py                      # Signal covariance matrix calculations
└── signal_templates/                             # GW background spectral templates
    ├── broken_power_law_template.py              # Broken power law spectrum model
    ├── flat_template.py                          # Flat (white) spectrum model
    ├── lognormal_template.py                     # Lognormal spectrum model
    ├── power_law_template.py                     # Power law spectrum model
    ├── signal_utils.py                           # Utilities for signal model creation
    ├── SIGWB_template.py                         # Stochastic inflationary GW background 
    │                                             # template
    ├── SMBH_broken_power_law_template.py         # SMBH and broken power law model
    ├── SMBH_flat_template.py                     # Combined SMBH and flat spectrum model
    ├── SMBH_lognormal_template.py                # SMBH and lognormal spectrum model
    └── SMBH_SIGWB_template.py                    # SMBH and SIGWB combined model

examples/
├── examples_utils.py                             # Utility functions for example notebooks
├── MCMC_Fisher_future.ipynb                      # Notebook for MCMC and Fisher analysis 
│                                                 # with future PTA data
├── scan_parameter.ipynb                          # Parameter scanning and sensitivity 
│                                                 # analysis
├── examples_first_paper/                         # Examples from the first fastPTA paper
│   ├── HD_constraints.ipynb                      # Analysis of Hellings-Downs correlation 
│   │                                             # constraints, reproduces Fig. 5 (right) of 
│   │                                             # 2404.02864
│   ├── MCMC_Fisher_EPTA.ipynb                    # MCMC and Fisher analysis with EPTA data
│   ├── N_scaling_figure.ipynb                    # Scaling behavior with number of pulsars, 
│   │                                             # reproduces Fig. 4 and 7 of 2404.02864
│   ├── T_scaling_figure.ipynb                    # Scaling behavior with observation time, 
│                                                 # reproduces Fig. 3 of 2404.02864
├── examples_paper_anisotropies/                  # Examples from the anisotropy paper
│   ├── linear_basis_figure.ipynb                 # Linear basis visualization for anisotropy, 
│   │                                             # reproduces Fig. 3 (left) of 2407.14460
│   ├── sqrt_basis_figure.ipynb                   # Square-root basis visualization for 
│   │                                             # anisotropy, reproduces Fig. 3 (right) of 
│   │                                             # 2407.14460
│   ├── strong_signal_limit.ipynb                 # Analysis of anisotropies in strong signal 
│   │                                             # limit, reproduces Fig. 1 (left) of 
│   │                                             # 2407.14460
│   ├── data_paper_2/                             # Data from the second paper
│   │   ├── limits_Cl_powerlaw_lin_ng15.dat       # Angular power spectrum limits (linear 
│   │   │                                         # basis)
│   │   └── limits_Cl_powerlaw_sqrt_ng15.dat      # Angular power spectrum limits (sqrt 
│   │                                             # basis)
├── examples_paper_cosmic_variance/               # Examples from the cosmic variance paper
│   ├── HD_normalization.ipynb                    # Hellings-Downs normalization analysis, 
│   │                                             # reproduces Fig. 1 of 2508.21131
│   ├── Histograms_PP_plots.ipynb                 # Posterior and P-P plots, reproduces Fig. 3 
│   │                                             # of 2508.21131
│   ├── plot_all_results.ipynb                    # Plot all Cls histograms, reproduces Fig. 4 
│   │                                             # of 2508.21131
│   ├── Plot_Cls.ipynb                            # Plotting of angular power spectra, 
│   │                                             # reproduces Fig. 2 of 2508.21131
│   ├── run_many_diagonal.ipynb                   # Multiple runs with diagonal approximation
│   ├── run_many_full.ipynb                       # Multiple runs with full correlation matrix
│   ├── run_many.py                               # Script for batch processing multiple runs
│   ├── Run_MCMC.ipynb                            # MCMC analysis for cosmic variance
├── examples_paper_SIGWB/                         # Examples from the SIGWB paper
│   ├── Fisher_forecasts_PL_SIGW.ipynb            # Fisher forecasts for PL and SIGW models, 
│   │                                             # reproduces Fig. 3 of 2503.10805
│   ├── MCMC_SIGW_PL_PBHs.ipynb                   # MCMC for SIGW, power law, and PBH models, 
│   │                                             # reproduces Fig. 7 (left) of 2503.10805
│   ├── N_pulsars.ipynb                           # Analysis of scaling with number of pulsars, 
│   │                                             # reproduces (a panel of) Fig. 5 of 
│   │                                             # 2503.10805
│   ├── Plot2D.ipynb                              # 2D visualization of results, reproduces 
│                                                 # Fig. 6 (left) of 2503.10805
└── pulsar_configurations/                        # Common pulsar configurations for all 
    │                                             # examples
    ├── EPTAlike_pulsar_parameters_noiseless.yaml # EPTA-like configuration without 
    │                                             # noise
    ├── EPTAlike_pulsar_parameters.yaml           # EPTA-like configuration with realistic 
    │                                             # noise
    └── mockSKA10_pulsar_parameters.yaml          # Mock SKA configuration with 10 pulsars

```

# Some examples:
- Navigate to examples for some scripts and jupyter notebooks explaining how to use the code. <br> See above for the description of the example folder.
    
# How to cite this code:
If you use fastPTA, please cite
- [2404.02864](https://arxiv.org/pdf/2404.02864), the original `fastPTA' paper 
and if appropriate:
- [2407.14460](https://arxiv.org/pdf/2407.14460), [2508.21131](https://arxiv.org/abs/2508.21131), for anisotropies searches,
- [2503.10805](https://arxiv.org/pdf/2503.10805) for SIGW searches. 

There's also a [Zenodo](https://zenodo.org/records/12820730) entry associated with this code.
