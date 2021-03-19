<a name="loadCurves"></a>
# loadCurves

<a name="loadCurves.create_nasa_curves"></a>
#### create\_nasa\_curves

```python
create_nasa_curves()
```

Creates a matrix of all NASA battery curves, and its labels

Creates each curve as a row in in a large X matrix. Each curve is voltage
interpolated to be in 128 steps as a function of the power delivered from
2% to 98%.

<a name="loadCurves.load_nasa_curves"></a>
#### load\_nasa\_curves

```python
load_nasa_curves(to_load)
```

Loads the curves cached to disk

**Returns**:

- `X` _ndarray_ - Matrix of curves
- `p` _ndarray_ - The discharge percentage that X rows are a function of
- `labels` _ndarray_ - Whuch battery each curve is
- `capacity` _ndarray_ - The capacity of each battery during each curve

<a name="main"></a>
# main

<a name="main.plot_aging"></a>
#### plot\_aging

```python
plot_aging(p, X, labels, capacity)
```

Plot examples of new and old battery curves.

**Arguments**:

- `X` _ndarray_ - Matrix of curves
- `p` _ndarray_ - The discharge percentage that X rows are a function of
- `labels` _ndarray_ - Whuch battery each curve is
- `capacity` _ndarray_ - The capacity of each battery during each curve

<a name="svd"></a>
# svd

<a name="svd.plot_mode_fraction"></a>
#### plot\_mode\_fraction

```python
plot_mode_fraction(s)
```

Plots the fraction of power represented with n modes.

**Arguments**:

- `s` _array-like_ - 1D arrray of the variances of the principal components.

<a name="svd.plot_n_modes"></a>
#### plot\_n\_modes

```python
plot_n_modes(p, X, V, n)
```

Shows the numbers represented with the selected number of SVD modes.

**Arguments**:

- `p` _array_like_ - The discharge percentage that X rows are a function of.
- `X` _array_like_ - Data matrix with rows of images.
- `V` _array_like_ - Matrix with mode vectors as columns.
- `n` _int_ - Number of modes to use in the representation.

<a name="svd.plot_mode_vs_life"></a>
#### plot\_mode\_vs\_life

```python
plot_mode_vs_life(capacity, X, V, n)
```

Plots how the capacity changes for values of each SVD mode.

**Arguments**:

- `capacity` _ndarray_ - The capacity of each battery during each curve.
- `X` _ndarray_ - Matrix of curves.
- `V` _array_like_ - Matrix with mode vectors as columns.
- `n` _int_ - Number of modes to plot.

<a name="predict"></a>
# predict

<a name="predict.find_weights"></a>
#### find\_weights

```python
find_weights(X, b)
```

Calculate the weights of the linear regression.

**Arguments**:

- `X` _array-like_ - Matrix with rows of curves.
- `b` _array-like_ - The capacity for each curve.

**Returns**:

- `ndarray` - Weights for the model.

<a name="predict.predict"></a>
#### predict

```python
predict(X, w)
```

Using previously calculated weights, predict the capacity of each curve.

**Arguments**:

- `X` _array-like_ - Matrix with rows of curves.
- `w` _array-like_ - Weights for the model.
  

**Returns**:

- `ndarray` - Predicted Capacities

<a name="predict.accuracy_dist"></a>
#### accuracy\_dist

```python
accuracy_dist(b_real, b_model, path_out)
```

Plots the error for two sets of capacties

**Arguments**:

- `b_real` _array-like_ - The capacity actually measured.
- `b_model` _array-like_ - Predicted capacities.

