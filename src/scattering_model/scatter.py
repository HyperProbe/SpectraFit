import numpy as np
from numpy.typing import NDArray


def compute_delta_A_error(X, *arg):
    """
    Compute the error between the measured delta attenuation and the modelled delta attenuation with scattering.

    Compute: M∆c + S - b

    Weird parameters in the function signature to match scipy.optimize.least_squares etc.

    Args:
        X (NDArray): Vector of concentrations and scattering parameters [_mol_1,...,mol_10,a_ti,b_ti_]
        *arg: [b, a_t1, b_t1, M, x, PL] PL is added by CIHAN

    Returns:
        NDArray: error vector
    """

    # m = N_HELICOID_MOLECULES
    b = arg[0]
    a_0, b_0 = arg[1], arg[2]
    M = arg[3]
    x = arg[4]
    PL = arg[5]
    delta_c_i = X[:-2]
    a_i = X[-2]
    b_i = X[-1]

    computed_A = compute_delta_A(
        M=M,
        delta_c_i=delta_c_i,
        a_i=a_i,
        b_i=b_i,
        a_0=a_0,
        b_0=b_0,
        x=x,
        PL=PL
    )
    return computed_A - b


def compute_delta_A(M, delta_c_i, a_i, b_i, b_0, a_0, x, PL) -> NDArray:
    """Compute delta Attenuation (with scattering) for a single pixel.

    Compute: M∆c + S (cf. Kevin's thesis: section 4.1.1)

        based on:
            - M · ∆c  + S = ∆A / PL(λ)
            - ∆S = (a1 · (λ / 500nm) **(-b1) - a0 · (λ / 500nm) **(-b0)) / (1 - 0.9)

    Args:
        M (_type_): Absorption matrix
        delta_c_i (_type_): Change in concentration at time/ pixel i
        a_i (_type_): Scattering coefficient a at time/ pixel i
        b_i (_type_): Scattering power b at time/ pixel i
        a_0 (_type_): Scattering coefficient a of reference spectrum
        b_0 (_type_): Scattering power b of reference spectrum
        x (_type_): selected wavelengths (typically here creatis_wl in selected range or subset for wl optimization)

    Returns:
        NDArray: Computed delta attenuation
    """

    S = compute_delta_scattering(a_i=a_i, b_i=b_i, a_0=a_0, b_0=b_0, x=x)
    A = (M @ delta_c_i + S)*PL
    return A


def compute_delta_scattering(a_i, b_i, a_0, b_0, x) -> NDArray:
    """S: Calc scattering (Thesis: section 4.1.1): (a1 · (λ / 500nm) **(-b1) - a0 · (λ / 500nm) **(-b0)) / (1 - 0.9)

    ## Assumptions:
        unitary pathlength of PL(λ) = 1
        majority of changes in scattering of one timepoint/pixel can be described from one contributor in the tissue

    Args:
        a_i (_type_): scatter coefficient a at time/ pixel i
        b_i (_type_): scatter power b at time/ pixel i
        b_1 (_type_): scatter power b of reference spectrum
        a_1 (_type_): scatter coefficient a of reference spectrum
        x (_type_): selected wavelengths (typically here creatis_wl in selected range or subset for wl optimization)

    Returns:
        NDArray: S
    """
    # delta_S = ((((x / 500) ** (-b_i)) * a_i) - (((x / 500) ** (-b_0)) * a_0)) / (
    #     1 - 0.9
    # )
    S_i = a_i * ((x / 500) ** (-b_i))
    S_0 = a_0 * ((x / 500) ** (-b_0))
    delta_S = (S_i - S_0) * 10  # scaling equivalent to 1 / (1 - 0.9)
    return delta_S


def compute_delta_A_img(M, delta_c, a, b, b_0, a_0, x) -> NDArray:
    """
    Compute the relative attenuation (delta A) with scattering for each pixel in the image.

    Args:
        M (NDArray): Absorption matrix
        delta_c (NDArray): Change in concentration for each pixel
        a (NDArray): Scattering coefficient a for each pixel
        b (NDArray): Scattering power b for each pixel
        a_0 (float): Scattering coefficient a of reference spectrum
        b_0 (float): Scattering power b of reference spectrum
        x (NDArray): selected wavelengths (typically here creatis_wl in selected range or subset for wl optimization)

    Returns:
        NDArray: _description_
    """

    assert delta_c.shape[:2] == a.shape[:2] == b.shape[:2]

    # img dim x wavelengths
    img_shape = delta_c.shape[:2]
    delta_A_img = np.zeros(shape=(*img_shape, x.shape[0]))

    # compute delta A for each pixel
    for i in range(delta_c.shape[0]):
        for j in range(delta_c.shape[1]):
            delta_c_i = delta_c[i, j, :]
            a_i = a[i, j]
            b_i = b[i, j]

            delta_A_img[i, j] = compute_delta_A(
                M=M, delta_c_i=delta_c_i, a_i=a_i, b_i=b_i, b_0=b_0, a_0=a_0, x=x
            )
    return delta_A_img
