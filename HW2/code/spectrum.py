import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import signal
from sksound import sounds
import cm_xml_to_matplotlib as cm


def plot_spectrum(f, t, S, name, freq_range=(50, 2000)):
    """Takes a provided rectangular array of spectrum vs time and creates a
    meshplot with the markings for notes of the equal tempered scale.

    Args:
        f (array_like): A 1D array of frequencies for the spectrum in P
        t (array_like): Time samples of the spectrum.
        S (array_like): An array of spectra at different times. Should be of
            shape (len(t), len(f)).
        name (str): The output file same. The plot will be saved in
            "HW2/figures/name.png".
        freq_range (tuple): The ends of the frequency range to plot. Defaults
            to (50 Hz, 2000 Hz).
    """
    if np.shape(S) != (len(f), len(t)):
        raise ValueError('S provided is of incorrect shape')

    # Load in the note names/frequencies
    notes = pd.read_csv('HW2/data/notes.csv')
    in_range = (notes['hertz'] > freq_range[0]) & \
               (notes['hertz'] < freq_range[1])
    notes = notes[in_range]
    natural = notes['note_flat'] == notes['note_sharp']
    notes['note_flat'] = notes['note_flat'] + notes['octave'].astype(str)

    # Start Plotting
    fig, ax_freq = plt.subplots(figsize=(10, 5))
    ax_note = ax_freq.twinx()  # Axis to mark notes on

    for ax in [ax_freq, ax_note]:
        ax.set_ylim(*freq_range)
        ax.set_yscale('log')

    # Don't mark with scientific notation
    ax_freq.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_freq.get_yaxis().set_minor_formatter(ticker.ScalarFormatter())

    # Label only C in each octave, but mark all natural notes
    ax_note.set_yticks(notes['hertz'][natural], minor=False)
    labels = [n if n[0] == 'C' else '' for n in notes['note_flat'][natural]]
    ax_note.set_yticklabels(labels, fontsize=18)

    # Only mark sharps/flats as minor ticks without labeling
    ax_note.set_yticks(notes['hertz'][~natural], minor=True)
    ax_note.get_yaxis().set_minor_formatter(ticker.NullFormatter())

    # Add lines across to mark the notes
    ax_note.grid(which='major', ls='-', c='k', lw=.5, alpha=.5)
    ax_note.grid(which='minor', ls=':', c='k', lw=.5, alpha=.4)

    ax_freq.set_xlabel('Time [s]')
    ax_freq.set_ylabel('Frequency [Hz]')

    # Proportional to PSD in db, but magnitude isn't displayed
    psd = np.log(np.abs(S)**2)

    vmin = np.percentile(psd, 50)

    # wavecmap = cm.make_cmap('HW2/code/12-4w_heir1_ccc1.xml')
    wavecmap = cm.make_cmap('HW2/code/3-3w_grblrd_ccc1.xml')
    ax_freq.pcolormesh(t, f, psd, cmap=wavecmap, shading='gouraud', vmin=vmin)
    fig.savefig(f'HW2/figures/{name}.png', dpi=400)


def get_spectrum(data, rate, width):
    """Produces the Gabor transform of the data provided using a gaussian
    window with the width provided. The transform is given in timesteps of
    1/4 the width.

    Args:
        data (array_like): The time series to be transformed.
        rate (float): The sampling rate (#/s) used for data.
        width (float): The width of the gaussian window in seconds.

    Returns:
        f (ndarray): Array of sample frequencies.
        t (ndarray): Array of sample times.
        S (ndarray): The Gabor transform of data.

    """
    window_width_pts = rate*width
    # Choose the size of the segment to be the nearest (in log space) power of
    # two to 4x the width of the filter.
    segment_width = pow(2, round(np.log2(5*window_width_pts)))
    overlap = segment_width - window_width_pts//4

    return signal.stft(data,
                       rate,
                       ('gaussian', window_width_pts),
                       nperseg=segment_width,
                       noverlap=overlap)


def low_pass(data, rate, cutoff):
    """Applies a 4th order low pass filter to a time series.

    Args:
        data (array_like): The time series to be filtered.
        rate (float): The sample rate of data.
        cutoff (float): Cutoff frequency for the filter in Hz.
    """
    S = np.fft.rfft(data)
    f = np.fft.rfftfreq(len(data), 1/rate)
    H = 1/(f/cutoff + 1)**4
    return np.fft.irfft(H*S)


def run_gnr():
    """Generates the spectrum of the Sweet Child O' Mine clip."""
    audio = sounds.Sound('HW2/data/GNR.m4a')
    f, t, S = get_spectrum(audio.data, audio.rate, .05)
    plot_spectrum(f, t, S, 'GNR-Spectrum', (250, 800))


def run_floyd_guitar():
    """Generates the spectrum of the guitar in the Comfortably Numb clip."""
    audio = sounds.Sound('HW2/data/Floyd.m4a')
    data = audio.data
    split_pts = [0, 20*audio.rate, 40*audio.rate, len(data)]

    for n, (left, right) in enumerate(zip(split_pts[:-1], split_pts[1:])):
        f, t, S = get_spectrum(data[left:right], audio.rate, .04)
        plot_spectrum(f, t, S, f'Floyd-Guitar-Spectrum-{n}', (110, 1000))


def run_floyd_bass():
    """Generates the spectrum of the bass in the Comfortably Numb clip, and
    produces a filtered clip of just the bass.
    """
    audio = sounds.Sound('HW2/data/Floyd.m4a')
    data = audio.data
    split_pts = [0, 20*audio.rate, 40*audio.rate, len(data)]

    for n, (left, right) in enumerate(zip(split_pts[:-1], split_pts[1:])):
        f, t, S = get_spectrum(data[left:right], audio.rate, .1)
        plot_spectrum(f, t, S, f'Floyd-Bass-Spectrum-{n}', (50, 140))

    bass_audio_data = low_pass(data, audio.rate, 140)
    bass_audio = sounds.Sound(inData=bass_audio_data, inRate=audio.rate)
    bass_audio.write_wav('HW2/output/Floyd-Bass.wav')


if __name__ == '__main__':
    run_gnr()
    run_floyd_bass()
    run_floyd_guitar()
