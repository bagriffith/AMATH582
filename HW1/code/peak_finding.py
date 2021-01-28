import numpy as np
import scipy.optimize


def gaussian_model(u, a, c, width, bg):
    """A 1D gaussian added to a background to be used
    for fitting in the peak identification.

    Args:
        u: The independent variable
        a (float): The scaling factor of the gaussian
        c (tuple): 3 long tuple of the center x, y,
            and z of the gaussian
        width (float): The standard deviation of the gaussian
        bg (float): The constant value added to the gaussian
    
    Returns:
        The model with provided parameters at u
    """
    return a * np.exp(-(u-c)**2/(2*(width**2))) \
                / (np.sqrt(2*np.pi)*width) + bg


def fit_axis(u, data):
    """Finds the optimal parameters of to fit gaussian_model
    to the data over the axis u.

    Args:
        u (np.ndarray): The axis to fit over
        data (np.ndarray): The 1D data det evaluated over u to fit
    
    Returns:
        popt (tuple): The optimal parameters
        pcov (np.ndarray): The covariance matrix of
            the parameter fit
    """
    du = u[1] - u[0]

    # Choose the initial guess. A sensible choice
    # is important to reliable convergence
    bg0 = np.min(data)
    a0 = np.sum(data - bg0)*du
    c0 = u[np.argmax(data)]
    popt, pcov = scipy.optimize.curve_fit(gaussian_model,
                                          u,
                                          data,
                                          p0=(a0,
                                              c0,
                                              8*du,
                                              bg0))
    return popt, pcov


def fit_center(data_psd, u):
    """Averages the psd along each axis and then returns the
    best fit parameters and covariance matrix for each
    """
    over = [(0, 2), (1, 2), (0, 1)]
    
    comp_opt = []
    comp_cov = []

    for axis in over:
        comp_avg = np.mean(data_psd, axis=axis)
        popt, pcov = fit_axis(u, comp_avg)
        comp_opt.append(popt)
        comp_cov.append(pcov)

    return comp_opt, comp_cov


def get_center(comp_opt):
    return [popt[1] for popt in comp_opt]


def get_center_unc(comp_cov):
    return [np.sqrt(np.diag(pcov)[1]) for pcov in comp_cov]


def get_signal_width(comp_opt):
    return [popt[2] for popt in comp_opt]


def find_peaks_over_t(data, u):
    centers = np.zeros((3, np.shape(data)[-1]))

    for i in range(np.shape(data)[-1]):
        comp_opt, comp_cov = fit_center(data[:, :, :, i], u)
        centers[:, i] = get_center(comp_opt)

    return centers
