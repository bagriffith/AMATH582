import numpy as np
from mnist import MNIST

_mnist_path = '/home/brady/Documents/class/2021w/AMATH582/HW4/data'
images_raw = None
labels_raw = None


def load_data(numbers=None, size=None):
    """Loads a matrix of selected numbers.

    Creates a matrix of shape (nsamples, npixels) where nsamples is the number
    of occurrences of the selected numbers.

    Args:
        numbers (list): A list of digits to load. If None or empty, all digits
            will by loaded. Defaults to None.
        size (int): The max number of images to load. If None, all images will
            be loaded. Defaults to None.
    
    Returns:
        np.int32: Matrix with rows of images
        np.int8: Array of digit labels for rows of images
    """

    global images_raw, labels_raw
    if images_raw is None or labels_raw is None:
        mndata = MNIST(_mnist_path)
        images_raw, labels_raw = mndata.load_training()

        images_raw = np.float64(images_raw)
        labels_raw = np.int8(labels_raw)

    images, labels = images_raw.copy(), labels_raw.copy()

    if numbers:
        # Select numbers if numbers isn't None or empty
        mask = np.isin(labels, numbers)
        images = images[mask]
        labels = labels[mask]

    if size:
        if len(labels) > size:
            mask = np.random.choice(len(labels), size=size, replace=False)
            images = images[mask]
            labels = labels[mask]

    return images, labels
