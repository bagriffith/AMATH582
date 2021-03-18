<a name="dmd"></a>
# dmd

<a name="dmd.dmd"></a>
#### dmd

```python
dmd(X2, u, s, vh)
```

Returns the DMD modes and complex frequencies for the system.

**Arguments**:

- `X2` _array-like_ - The X_2^M matrix
- `u` _array-like_ - The U matrix of the X_1^M-1 SVD
- `s` _array-like_ - The s array of the X_1^M-1 SVD
- `vh` _array-like_ - The vh matrix of the X_1^M-1 SVD

**Returns**:

- `ndarray` - Array of complex frequencies for DMD modes
- `ndarray` - Matrix with rows of the DMD modes

<a name="dmd.x_dmd"></a>
#### x\_dmd

```python
x_dmd(t, psi, w, b)
```

The DMD approximation of x(t).

**Arguments**:

- `t` _float_ - The time in frames
- `psi` _array-like_ - Matrix with rows of the DMD modes
- `w` _array-like_ - Array of complex frequencies for DMD modes
- `b` _array-like_ - Array of initial values of the DMD modes

**Returns**:

- `ndarray` - DMD approximation of pixels at t

<a name="dmd.frame_bg_sep"></a>
#### frame\_bg\_sep

```python
frame_bg_sep(t, X, psi, w, b)
```

Separate the forground and background of frame t

**Arguments**:

- `t` _int_ - The time in frames
- `psi` _array-like_ - Matrix with rows of the DMD modes
- `w` _array-like_ - Array of complex frequencies for DMD modes
- `b` _array-like_ - Array of initial values of the DMD modes

**Returns**:

- `ndarray` - Foreground array
- `ndarray` - Background array

<a name="dmd.show_frame"></a>
#### show\_frame

```python
show_frame(frame, shape, path_out)
```

Plot the frame provided.

**Arguments**:

- `frame` _array-like_ - 1 D array of pixels
- `shape` _tuple_ - The shape of the image (pixels_y, pixels_x)
- `path_out` _str_ - Path to save figure to

<a name="svd"></a>
# svd

<a name="svd.plot_n_modes"></a>
#### plot\_n\_modes

```python
plot_n_modes(X, V, n, shape, path_out)
```

Shows the numbers represented with the selected number of SVD modes

**Arguments**:

- `X` _array_like_ - Data matrix with rows of images
- `V` _array_like_ - Matrix with mode vectors as columns
- `n` _int_ - Number of modes to use in the representation
- `shape` _tuple_ - The shape of the image (pixels_y, pixels_x)
- `path_out` _str_ - Path to save figure to

<a name="svd.plot_mode_fraction"></a>
#### plot\_mode\_fraction

```python
plot_mode_fraction(s, path_out)
```

Plots the fraction of power represented with n modes

**Arguments**:

- `s` _array-like_ - 1D arrray of the variances of the principal components.
- `path_out` _str_ - Path to save figure to

<a name="loadVid"></a>
# loadVid

<a name="loadVid.open_video"></a>
#### open\_video

```python
open_video(vid_path)
```

Loads the video matrices

**Arguments**:

- `vid_path` _str_ - Path to the video file
  

**Returns**:

- `ndarray` - X_1^M-1
- `ndarray` - X_2^M

