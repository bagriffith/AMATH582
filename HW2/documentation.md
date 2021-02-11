<a name="spectrum"></a>
# spectrum

<a name="spectrum.plot_spectrum"></a>
#### plot\_spectrum

```python
plot_spectrum(f, t, S, name, freq_range=(50, 2000))
```

Takes a provided rectangular array of spectrum vs time and creates a
meshplot with the markings for notes of the equal tempered scale.

**Arguments**:

- `f` _array_like_ - A 1D array of frequencies for the spectrum in P
- `t` _array_like_ - Time samples of the spectrum.
- `S` _array_like_ - An array of spectra at different times. Should be of
  shape (len(t), len(f)).
- `name` _str_ - The output file same. The plot will be saved in
  "HW2/figures/name.png".
- `freq_range` _tuple_ - The ends of the frequency range to plot. Defaults
  to (50 Hz, 2000 Hz).

<a name="spectrum.get_spectrum"></a>
#### get\_spectrum

```python
get_spectrum(data, rate, width)
```

Produces the Gabor transform of the data provided using a gaussian
window with the width provided. The transform is given in timesteps of
1/4 the width.

**Arguments**:

- `data` _array_like_ - The time series to be transformed.
- `rate` _float_ - The sampling rate (#/s) used for data.
- `width` _float_ - The width of the gaussian window in seconds.
  

**Returns**:

- `f` _ndarray_ - Array of sample frequencies.
- `t` _ndarray_ - Array of sample times.
- `S` _ndarray_ - The Gabor transform of data.

<a name="spectrum.low_pass"></a>
#### low\_pass

```python
low_pass(data, rate, cutoff)
```

Applies a 4th order low pass filter to a time series.

**Arguments**:

- `data` _array_like_ - The time series to be filtered.
- `rate` _float_ - The sample rate of data.
- `cutoff` _float_ - Cutoff frequency for the filter in Hz.

<a name="spectrum.run_gnr"></a>
#### run\_gnr

```python
run_gnr()
```

Generates the spectrum of the Sweet Child O' Mine clip.

<a name="spectrum.run_floyd_guitar"></a>
#### run\_floyd\_guitar

```python
run_floyd_guitar()
```

Generates the spectrum of the guitar in the Comfortably Numb clip.

<a name="spectrum.run_floyd_bass"></a>
#### run\_floyd\_bass

```python
run_floyd_bass()
```

Generates the spectrum of the bass in the Comfortably Numb clip, and
produces a filtered clip of just the bass.

