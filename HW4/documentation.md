<a name="lda"></a>
# lda

<a name="lda.LDA"></a>
## LDA Objects

```python
class LDA()
```

Linear Discrimination Analysis model for classifying in up to 3 groups.

<a name="lda.LDA.fit"></a>
#### fit

```python
 | fit(X, y)
```

Trains the model.

**Arguments**:

- `X` _array-like_ - Contains rows of the training data examples.
- `y` _array-like_ - Contains labels for the training data rows.

<a name="lda.LDA.predict"></a>
#### predict

```python
 | predict(X)
```

Predicts the category of rows of X.

**Arguments**:

- `X` _array-like_ - Contains rows of the data to categorize.

<a name="evaluation"></a>
# evaluation

<a name="evaluation.NaiveClassifier"></a>
## NaiveClassifier Objects

```python
class NaiveClassifier()
```

A model classifier that randomly guesses a digit.

This was created as a simple test article to make sure the number_confusion
code worked indpendent of any model used.

<a name="evaluation.number_confusion"></a>
#### number\_confusion

```python
number_confusion(model, train_n, V)
```

Plots a how the model preforms at distinguising pairs of digits.

**Arguments**:

- `model` - The model class. Should have functions fit(X, y) that trains the
  model to identify labels y using data X and predict(X) that will
  label data in the matrix X.
- `train_n` _int_ - The number of example digits to train on.
- `V` _array-like_ - A matrix to transform the data into the basis for
  predictions.

<a name="evaluation.full_classification"></a>
#### full\_classification

```python
full_classification(model, train_n, V)
```

Plots a how the model preforms at identifying digits.

**Arguments**:

- `model` - The model class. Should have functions fit(X, y) that trains the
  model to identify labels y using data X and predict(X) that will
  label data in the matrix X.
- `train_n` _int_ - The number of example digits to train on.
- `V` _array-like_ - A matrix to transform the data into the basis for
  predictions.

<a name="evaluation.digit_performance"></a>
#### digit\_performance

```python
digit_performance(model, train_n, V, digits)
```

Plots a how the model preforms at identifying digits.

**Arguments**:

- `model` - The model class. Should have functions fit(X, y) that trains the
  model to identify labels y using data X and predict(X) that will
  label data in the matrix X.
- `train_n` _int_ - The number of example digits to train on.
- `V` _array-like_ - A matrix to transform the data into the basis for
  predictions.
- `digits` _list_ - List of digits to test on

<a name="main"></a>
# main

<a name="main.run_analysis"></a>
#### run\_analysis

```python
run_analysis()
```

Runs the full analysis for the MNIST handwritting project

<a name="svd"></a>
# svd

<a name="svd.plot_mode_proj"></a>
#### plot\_mode\_proj

```python
plot_mode_proj(X, V, labels, modes)
```

Creates a 3D projection of X into the 3 selected SVD modes

**Arguments**:

- `X` _array_like_ - Data matrix with rows of images
- `V` _array_like_ - Matrix with mode vectors as columns
- `modes` _list_ - List of 3 mode indexes to project on

<a name="svd.plot_n_modes"></a>
#### plot\_n\_modes

```python
plot_n_modes(X, V, n)
```

Shows the numbers represented with the selected number of SVD modes

**Arguments**:

- `X` _array_like_ - Data matrix with rows of images
- `V` _array_like_ - Matrix with mode vectors as columns
- `n` _int_ - Number of modes to use in the representation

<a name="svd.plot_svd_spectrum"></a>
#### plot\_svd\_spectrum

```python
plot_svd_spectrum(X, V)
```

Plots the svd spectrum of 4 random images.

**Arguments**:

- `X` _array_like_ - Data matrix with rows of images
- `V` _array_like_ - Matrix with mode vectors as columns

<a name="svd.plot_mode_fraction"></a>
#### plot\_mode\_fraction

```python
plot_mode_fraction(s)
```

Plots the fraction of power represented with n modes

**Arguments**:

- `s` _array-like_ - 1D arrray of the variances of the principal components.

<a name="loadmnist"></a>
# loadmnist

<a name="loadmnist.load_data"></a>
#### load\_data

```python
load_data(numbers=None, size=None)
```

Loads a matrix of selected numbers.

Creates a matrix of shape (nsamples, npixels) where nsamples is the number
of occurrences of the selected numbers.

**Arguments**:

- `numbers` _list_ - A list of digits to load. If None or empty, all digits
  will by loaded. Defaults to None.
- `size` _int_ - The max number of images to load. If None, all images will
  be loaded. Defaults to None.
  

**Returns**:

- `np.int32` - Matrix with rows of images
- `np.int8` - Array of digit labels for rows of images

