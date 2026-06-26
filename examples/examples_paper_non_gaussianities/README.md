# Paper-figure notebooks — *Are PTA measurements sensitive to gravitational wave non-Gaussianities?*

Reproduces every figure of

> **Are PTA measurements sensitive to gravitational wave non-Gaussianities?**
> C. Cecchini, J. El Gammal, G. Franciolini, M. Pieroni
> [arXiv:2605.05157](https://arxiv.org/abs/2605.05157)
>
> *(Note: the arXiv version is being updated; the figures here track the current
> manuscript and may differ slightly from the first arXiv posting.)*

Five self-contained Jupyter notebooks, one per figure. This example ships
inside [`fastPTA`](https://github.com/Mauropieroni/fastPTA) (for the PTA
response) and additionally needs
[`fastropop`](https://github.com/jonaselgammal/fastropop) (for the
semi-analytic SMBHB population). Everything else is `numpy`, `scipy`,
`healpy`, `jax`, and `matplotlib`.

## Layout

```
.
├── fig1_whitening.ipynb          Whitening QQ plot                          (Fig 1)
├── fig2_cosmo_KS_HDandR.ipynb    Isotropic-injection KS rejection, R vs HD  (Fig 2)
├── fig3_ks_rejection.ipynb       KS rejection sweep, toy + astro            (Fig 3)
├── fig4_clt_and_scale.ipynb      CLT + scale-estimation histograms          (Fig 4)
├── fig5_antenna_pattern.ipynb    Mollweide antenna patterns  (Fig 5; SM)    (Fig 5)
├── helpers/
│   ├── kolmogorov_smirnov.py     Bootstrap-calibrated KS test
│   └── pta_helpers.py            Shared PTA / response / whitening / population code
├── data/                         Cached intermediate results (.npz)
└── plots/                        Saved figures (.pdf)
```

`fig5` (the single-pulsar antenna patterns) appears in the Supplemental
Material of the Letter, not the main text, but is generated the same way.

Each notebook imports the local `helpers/` package, so **run the notebooks
from this directory** (so that `import helpers...` resolves).

## Flags

Every notebook has two flags at the top:

| flag              | default | effect                                                                                  |
|-------------------|---------|-----------------------------------------------------------------------------------------|
| `REGENERATE_DATA` | `False` | load the cached `.npz` from `data/`. Set to `True` to recompute (slow for Figs 2 and 3). |
| `SAVE_PLOT`       | `True`  | write the figure to `plots/`. Set to `False` to only show it inline.                      |

## Requirements / setup

```bash
# from the fastPTA repository root
pip install -e .                      # or: pip install fastPTA once published

# the SMBHB population code (required to import helpers, used for the astro row)
pip install fastropop                 # or: pip install -e /path/to/fastropop

# everything else
pip install healpy numpy scipy matplotlib jax jaxlib tqdm

# optional: register a kernel for the notebooks
python -m ipykernel install --user --name fastpta-examples
```

`fastropop` is imported unconditionally by `helpers/pta_helpers.py`, so it
must be installed even for the figures that do not use the astrophysical
population.

## Notes on the figures

(Throughout, $R_{IJ}$ is the response-integral covariance — the *correct*
whitening basis — and $\Gamma_{IJ}$ is the analytic Hellings–Downs curve, the
approximation available in practice. $w_I = (\Lambda^{-1/2}M^\dagger)_{IK}d_K$
are the whitened data and $|w_I|^2$ the whitened "diagonal powers".)

- **Fig 1 — `fig1_whitening.ipynb`.** Generates 1000 multivariate-Gaussian
  realizations on a fixed PTA ($N_p=67$, `Nside=16`) and computes KS
  $p$-values against an $\mathrm{Exp}(1)$ reference for the raw diagonal powers
  $|d_I|^2/R_{II}$ and the whitened powers $|w_I|^2$, plotting their empirical
  quantiles as a QQ plot. Without whitening the inter-pulsar correlations cause
  spurious rejection; after whitening the curve tracks the diagonal. Fast.

- **Fig 2 — `fig2_cosmo_KS_HDandR.ipynb`.** The *isotropic* test: per-pixel
  complex amplitudes are drawn from three distributions (uniform, exponential,
  normal) with no discrete sources, propagated through the PTA response,
  whitened, and KS-tested. Reports the rejection fraction (`p < 0.05`) vs
  $N_p \in \{10, 50, 67, 100, 200, 500\}$ over 20000 realizations. Three panels
  compare whitening with the response $R$ (known scale, and scale estimated)
  and with the HD curve $\Gamma$ (scale estimated); all sit at the nominal
  5%. **Slow** (large `Nreal` × 50000-sample bootstrap); cached data ≈ 11 MB.

- **Fig 3 — `fig3_ks_rejection.ipynb`.** The anisotropic / sparse-source
  sweep, whitened with the response $R$. Rejection fraction vs source count
  $N_s \in \{1, 5, 10, 50, 100, 300, 500, 1000\}$ and pulsar count
  $N_p \in \{10, 20, 50, 67, 100, 200, 500\}$, 5000 realizations per cell, for
  two toy populations (uniform, exponential) and the astrophysical `lown0`
  SMBHB model, with two KS variants (known scale; bootstrap-calibrated with
  scale estimated). **Slow**; cached toy + astro data ≈ 25 MB total. The astro
  sweep (which uses `fastropop`) is cached separately so it can be regenerated
  independently.

- **Fig 4 — `fig4_clt_and_scale.ipynb`.** For $N_p=300$ and
  $N_s \in \{1, 2, 10\}$, histograms of the stacked real and imaginary parts
  of $w_I$ ($2N_p$ samples per realization) for three source-amplitude
  distributions (uniform, exponential, log-normal), shown before and after
  scale normalization, for the true PTA response and for two synthetic top-hat
  patch responses ($30^\circ$ and $15^\circ$ half-angle, synthesized in the
  notebook). Illustrates that scale normalization collapses everything onto
  $\mathcal{N}(0,1)$ for the broad PTA response, while the narrower patches
  retain visible deviations. Fast.

- **Fig 5 — `fig5_antenna_pattern.ipynb`.** (Supplemental Material.)
  Per-pulsar Mollweide maps at `Nside=256`: the true PTA antenna pattern
  alongside the two top-hat patches ($30^\circ$, $15^\circ$) used in Fig 4.
  Moderate cost; cached map data ≈ 18 MB.

## LaTeX

`helpers.pta_helpers.set_paper_rcparams()` enables `text.usetex` if a working
LaTeX install is detected on `$PATH`; otherwise it falls back to matplotlib's
built-in mathtext renderer and prints a one-line warning. Plots remain readable
in the fallback mode, but small notational details may differ.
