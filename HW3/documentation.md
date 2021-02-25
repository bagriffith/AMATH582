<a name="pca"></a>
# pca

<a name="pca.Video"></a>
## Video Objects

```python
class Video()
```

Parameters of to load a single video matrix.

This is to load the video form the matlab matrix file and then crop it
appropriately.

**Attributes**:

- `filename` _str_ - The path to the matlab matrix file for the video.
- `start` _int_ - The frame to start loading from.
- `end` _int_ - Load frames before this number.
- `left` _int_ - The left edge to crop from.
- `right` _int_ - The right edge to crop from.
- `top` _int_ - The top edge to crop from.
- `bottom` _int_ - The bottom edge to crop from.

<a name="pca.Video.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(filename, start, end, left, right, top, bottom)
```

Initialize the Video class.

args:
    filename (str): The path to the matlab matrix file for the video.
    start (int): The frame to start loading from.
    end (int): Load frames before this number.
    left (int): The left edge to crop from.
    right (int): The right edge to crop from.
    top (int): The top edge to crop from.
    bottom (int): The bottom edge to crop from.

<a name="pca.Video.read"></a>
#### read

```python
 | read()
```

Retrieve the matrix of the video, cropped as specified.

**Returns**:

- `ndarray` - The matrix of the video. The shape is (vertical pixels,
  horizontal pixels, frames).

<a name="pca.Video.from_text"></a>
#### from\_text

```python
 | @staticmethod
 | from_text(text)
```

Creates a video class from the line of a CSV

**Arguments**:

- `text` _str_ - Line from a csv "path, start, end, left, right, top,
  bottom"
  

**Returns**:

- `Video` - A `Video` class for the line provided.

<a name="pca.make_vid_list"></a>
#### make\_vid\_list

```python
make_vid_list()
```

Create lists of `Video` for all 4 tests.txt

Creates the videos with the properties defined in the vid_props files.

**Returns**:

- `list` - List of each test's list of `Video` objects.

<a name="pca.read_meas_matrix"></a>
#### read\_meas\_matrix

```python
read_meas_matrix(vid_list)
```

Loads the matrix with rows being the separate time measurements.

This is the X matrix expected for PCA.

**Arguments**:

- `vid_list` _list_ - A list of `Video` class objects to load the
  measurements from. All of the should be the same length.
  

**Returns**:

- `ndarray` - The measurements matrix X for PCA. Shape (Number of Pixels,
  Number of Frames)

<a name="pca.pca"></a>
#### pca

```python
pca(M)
```

Preforms PCA on the matrix provided.

**Arguments**:

- `M` _array-like_ - The matrix to preform PCA on.
  

**Returns**:

- `ndarray` - The variances of the principal components.
- `ndarray` - U_T matrix to project M into principal components.

<a name="pca.plot_dominant_mode"></a>
#### plot\_dominant\_mode

```python
plot_dominant_mode(s, U_T, X, fig_path)
```

Plots the variances and the fist 4 PCA modes.

**Arguments**:

- `s` _array-like_ - 1D arrray of the variances of the principal components.
- `U_T` _array-like_ - The matrix to transform X into the principal
  components.
- `X` _array-like_ - The measurement matrix
- `fig_path` _str_ - The path to save the plot.
- `comps` _int_ - Number of modes to plot

