import numpy as np
from scipy.stats import chi2, kstest


def kolmogorov_smirnov_statistic_chi2(data, df, scale):
    """Compute the one-sample KS statistic against a chi2(df, scale) null."""
    stat, _ = kstest(data, "chi2", args=(df, 0.0, scale))
    return float(stat)


def _ecdf_pvalue(observed_stat, bootstrap_stats):
    """
    Interpolated p-value from a sorted bootstrap null distribution.

    This mirrors the ECDF interpolation used for the AD bootstrap caches so the
    returned p-values are smooth rather than being quantized at multiples of
    1 / n_bootstrap.
    """
    B = len(bootstrap_stats)
    rank = np.searchsorted(bootstrap_stats, observed_stat, side="left")
    if rank == 0:
        return 1.0 - 0.5 / B
    if rank >= B:
        return 0.5 / B
    lower = bootstrap_stats[rank - 1]
    upper = bootstrap_stats[rank]
    frac = (observed_stat - lower) / (upper - lower) if upper > lower else 0.5
    p = 1.0 - (rank - 1 + frac) / B
    return float(np.clip(p, 0.5 / B, 1.0 - 0.5 / B))


def _bootstrap_stat_chi2(n, df, scale):
    """
    Generate one chi2 sample, estimate the scale from the same sample, and
    compute the corresponding KS statistic.
    """
    sample = chi2.rvs(df, loc=0.0, scale=scale, size=n)
    estimated_scale = np.mean(sample) / df
    return kolmogorov_smirnov_statistic_chi2(sample, df=df, scale=estimated_scale)


class KolmogorovSmirnovBootstrapCache:
    """
    Bootstrap cache for the KS statistic with the scale estimated from the data.

    The null distribution depends only on the sample size n and the chi2 degrees
    of freedom df because the scale is profiled out in each bootstrap sample.
    """

    def __init__(self, n, df=2, n_bootstrap=100000, scale=1.0):
        self.n = int(n)
        self.df = int(df)
        self.n_bootstrap = int(n_bootstrap)
        self.scale = float(scale)

        print(
            f"Precomputing {self.n_bootstrap} bootstrap KS samples for chi2 "
            f"(n={self.n}, df={self.df})..."
        )
        self.bootstrap_stats = np.sort(
            np.array(
                [
                    _bootstrap_stat_chi2(self.n, self.df, self.scale)
                    for _ in range(self.n_bootstrap)
                ]
            )
        )
        print(
            "  Done: "
            f"mean={np.mean(self.bootstrap_stats):.4f}, "
            f"std={np.std(self.bootstrap_stats):.4f}"
        )

    def compute_test(self, data, interpolate=True):
        """
        Compute the KS test against chi2 with df degrees of freedom and the
        scale estimated from the input sample.
        """
        data = np.asarray(data)
        if data.shape[0] != self.n:
            raise ValueError(f"Data length {data.shape[0]} does not match cache n={self.n}")

        estimated_scale = np.mean(data) / self.df
        observed_stat = kolmogorov_smirnov_statistic_chi2(
            data, df=self.df, scale=estimated_scale
        )
        if interpolate:
            p_value = _ecdf_pvalue(observed_stat, self.bootstrap_stats)
        else:
            p_value = float(np.mean(self.bootstrap_stats >= observed_stat))
        return observed_stat, p_value
