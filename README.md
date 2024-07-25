# fastPTA
A jit-enhanced Python code to forecast the sensitivity of future Pulsar Timing Array (PTA) configurations and assess constraints on Stochastic Gravitational Wave Background (SGWB) parameters. 
The code can generate mock PTA catalogs with noise levels compatible with current and future PTA experiments.
These catalogs can then be used to perform Fisher forecasts of MCMC runs.

# Installation
- Clone from `https://github.com/Mauropieroni/fastPTA`
- `python3 -m pip install ./fastPTA` -- you can delete the `fastPTA` folder now.
- [alternatively `python3 -m pip install -e ./fastPTA`, to have changes in the code propagate instantaneously; do not delete `fastPTA` in this case!]

# To test:
- After installation (see above) you can run `$pytest $(cd fastPTA/)'. A series of tests will run to check everything works fine.
 
# Some examples:
- Navigate to examples for some scripts and jupyter notebooks explaining how to use the code
    
# How to cite this
If you use fastPTA, please cite [2404.02864](https://arxiv.org/pdf/2404.02864) and [2407.14460](https://arxiv.org/pdf/2407.14460). 
There's also a [Zenodo](https://zenodo.org/records/12820730) entry associated with this code.
