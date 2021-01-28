import numpy as np
import scipy.io

import fourier
import peak_finding
import visualize


def run_all():
    # Shape of the submarine data
    n_w = 64  # Space points
    n_t = 49  # Time points
    L = 10    # Width of the cube of space

    # Establish axes in space
    x = np.linspace(-L, L, n_w + 1)[:-1]
    y = x.copy()
    z = x.copy()

    # And in wavenumber
    k = (2*np.pi/(2*L))*np.fft.fftfreq(n_w)
    ks = np.fft.fftshift(k)
    dk = k[1] - k[0]

    # Load the data file
    sub_data_file = 'data/subdata.mat'
    sub_data_raw = scipy.io.loadmat(sub_data_file)['subdata']

    # Reshape the data to be shaped (x, y, z, t)
    sub_data = np.reshape(sub_data_raw, (n_w, n_w, n_w, n_t))

    # Move into wavenumber space
    sub_spectrum = fourier.fourier_transform(sub_data)
    psd = fourier.psd_averaged(sub_spectrum, dk)

    # Identify the characteristic frequency
    comp_opt, comp_cov = peak_finding.fit_center(psd, ks)
    visualize.plot_xyz_psd(psd, ks, comp_opt, comp_cov)
    center_f = peak_finding.get_center(comp_opt)
    signal_width = peak_finding.get_signal_width(comp_opt)
    visualize.plot_projections(psd, ks, center_f, 2.5*max(signal_width))

    # Filter the signal and return to position space
    filer_width = 2*max(signal_width)
    sub_spec_filtered = \
        fourier.filter_around(sub_spectrum,
                              center_f,
                              filer_width,
                              [ks]*3)

    sub_data_filtered = fourier.inv_fourier_transform(sub_spec_filtered)
    sub_data_psd_f = np.abs(sub_data_filtered)**2

    # Find that sub
    sub_loc = peak_finding.find_peaks_over_t(sub_data_psd_f, x)
    visualize.record_path(sub_loc)


if __name__ == '__main__':
    run_all()
