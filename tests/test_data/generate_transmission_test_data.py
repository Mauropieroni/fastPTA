import numpy as np
import fastPTA.transmission_functions as tf

yr = 365.25 * 24 * 3600

# Init time and frequencies arrays
Tobs = np.random.uniform(10, 15, size=25) * yr
Tobs[0] = 15 * yr
t = np.linspace(0.0, Tobs[0], 500)
frequencies = np.arange(1, 51) / (15 * yr)

# Init design matrix (quadratic spin-down model + one year peak)
Mmat = np.array(
    [
        np.ones(len(t)),
        t,
        0.5 * t**2,
        np.sin(2 * np.pi * (1 / yr) * t),
        np.cos(2 * np.pi * (1 / yr) * t),
    ]
).T
Mmat /= np.sqrt(np.sum(Mmat**2, axis=0))

np.savez(
    "transmission_data.npz",
    T_obs=Tobs,
    t=t,
    Mmat=Mmat,
    frequencies=frequencies,
    transmission_approx_single=tf.transmission_function_approx(
        frequencies, Tobs[0]
    ),
    transmission_quadratic_single=tf.transmission_function_quadratic(
        frequencies, Tobs[0]
    ),
    transmission_quadratic_1yr_peak_single=tf.transmission_function_quadratic_1yr_peak(
        frequencies, Tobs[0]
    ),
    transmission_approx_tensor=tf.transmission_function_approx(
        frequencies[:, None], Tobs[None, :]
    ),
    transmission_quadratic_tensor=tf.transmission_function_quadratic(
        frequencies[:, None], Tobs[None, :]
    ),
    transmission_quadratic_1yr_peak_tensor=tf.transmission_function_quadratic_1yr_peak(
        frequencies[:, None], Tobs[None, :]
    ),
    transmission_matrix_single=tf.transmission_function_matrix(
        frequencies, t, Mmat
    ),
)
