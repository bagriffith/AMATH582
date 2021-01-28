import numpy as np


def fourier_transform(data_space):
    """Uses np.nfft to calculate the 3D Fourier
    transform of each spacial recording.

    Args:
        data_space (np.ndarray): 4D array arranged with
        axes X,Y,Z,T
    
    Returns:
        np.ndarray: The fourier transformed data. Arranged
            with axes Kx,Ky,Kz,T
    """
    return np.fft.fftn(data_space, axes=[0, 1, 2])


def inv_fourier_transform(data_spec):
    """Uses np.infft to calculate the 3D Inverse
    Fourier transform of each frequency recording.

    Args:
        data_spec (np.ndarray): 4D array arranged with
        axes Kx,Ky,Kz,T
    
    Returns:
        np.ndarray: The inverse fourier transformed data.
            Arranged with axes X,Y,Z,T
    """
    return np.fft.ifftn(data_spec, axes=[0, 1, 2])


def psd_averaged(data_spec, dk):
    """Averages the provided spectral data over all recordings
    and then returns its PSD.

    Args:
        data_spec (np.ndarray): 4D array arranged with
        axes Kx,Ky,Kz,T
        dk: The frequency step size
    
    Returns:
        np.ndarray: PSD from the averaged frequency. Reduced 1
        dimension from data_spec
    """
    data_avg_spec = np.apply_along_axis(np.mean, 3, data_spec)
    psd = .5 * np.abs(data_avg_spec)**2 / (dk**3)
    return psd


def filter_around(data, center, width, axes):
    """Applies a symmetric 3D gaussian filter to each recording
    in a provided frequency array.

    Args:
        data (np.ndarray): 4D array arranged with axes Kx,Ky,Kz,T
        center (tuple): 
    
    Returns:
        np.ndarray: Data with the filter applied
    """
    u, v, w = axes
    U, V, W = np.meshgrid(u, v, w)
    R2 = sum([(U_i - c_i)**2 for U_i, c_i in zip([U, V, W], center)])
    filter_weights = np.exp(-R2/(2*width**2)) / (2 * np.pi * width**2)**(3/2)
    return data * np.tile(filter_weights[:, :, :, np.newaxis],
                          np.shape(data)[-1])
