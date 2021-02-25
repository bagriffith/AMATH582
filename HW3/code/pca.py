import numpy as np
import matplotlib.pyplot as plt
from scipy import io


class Video:
    """Parameters of to load a single video matrix.

    This is to load the video form the matlab matrix file and then crop it
    appropriately.

    Attributes:
        filename (str): The path to the matlab matrix file for the video.
        start (int): The frame to start loading from.
        end (int): Load frames before this number.
        left (int): The left edge to crop from.
        right (int): The right edge to crop from.
        top (int): The top edge to crop from.
        bottom (int): The bottom edge to crop from.
    """
    def __init__(self, filename, start, end, left, right, top, bottom):
        """Initialize the Video class.

        args:
            filename (str): The path to the matlab matrix file for the video.
            start (int): The frame to start loading from.
            end (int): Load frames before this number.
            left (int): The left edge to crop from.
            right (int): The right edge to crop from.
            top (int): The top edge to crop from.
            bottom (int): The bottom edge to crop from.
        """
        self.filename = filename

        if start > end:
            raise ValueError('`start` must be lower than `end`')
        self.start = start
        self.end = end

        if left > right:
            raise ValueError('`left` must be lower than `right`')
        self.left = left
        self.right = right

        if top > bottom:
            raise ValueError('`top` must be lower than `bottom`')
        self.top = top
        self.bottom = bottom

    def read(self):
        """Retrieve the matrix of the video, cropped as specified.

        Returns:
            ndarray: The matrix of the video. The shape is (vertical pixels,
                horizontal pixels, frames).
        """
        ml_file = io.loadmat(self.filename)

        # The file should only contain one matrix. If not, this will break
        key = list(ml_file.keys())[-1]

        ds = 3  # Down sample by this factor
        M = ml_file[key][self.top:self.bottom,  # Crop Horizontal
                         self.left:self.right,  # Crop Vertical
                         :,                     # Leave all color
                         self.start:self.end]   # Crop to start

        # Throw out anything after the last down sampled point on the
        # right and bottom
        M = M[:ds*(M.shape[0]//ds), :ds*(M.shape[1]//ds), :, :]

        # Average over the down sampled region and over all color channels
        M = M.reshape(M.shape[0]//ds,
                      ds,
                      M.shape[1]//ds,
                      ds, M.shape[-2],
                      M.shape[-1]).mean(3).mean(1).mean(2)

        return M

    @staticmethod
    def from_text(text):
        """Creates a video class from the line of a CSV

        Args:
            text (str): Line from a csv "path, start, end, left, right, top,
                bottom"

        Returns:
            Video: A `Video` class for the line provided.
        """
        elements = text.strip().split(',')

        if len(elements) != 7:
            raise ValueError('Line provided cannot form Video object')

        return Video(elements[0].strip('"'),
                     *(int(x) for x in elements[1:]))


def make_vid_list():
    """Create lists of `Video` for all 4 tests.txt

    Creates the videos with the properties defined in the vid_props files.

    Returns:
        list: List of each test's list of `Video` objects.
    """
    tests = []

    for i in range(1, 5):
        with open(f'data/vid_props/test{i}.csv') as f:
            test = [Video.from_text(l) for l in f.readlines()[1:]]
        tests.append(test)

    return tests


def read_meas_matrix(vid_list):
    """Loads the matrix with rows being the separate time measurements.

    This is the X matrix expected for PCA.

    Args:
        vid_list (list): A list of `Video` class objects to load the
            measurements from. All of the should be the same length.

    Returns:
        ndarray: The measurements matrix X for PCA. Shape (Number of Pixels,
        Number of Frames)
    """
    X = None
    for vid in vid_list:
        A = vid.read()
        B = A.reshape((-1, A.shape[-1]))
        B -= np.mean(B)
        X = B if X is None else np.append(X, B, axis=0)
    return X


def pca(M):
    """Preforms PCA on the matrix provided.

    Args:
        M (array-like): The matrix to preform PCA on.
    
    Returns:
        ndarray: The variances of the principal components.
        ndarray: U_T matrix to project M into principal components.

    """
    C = M / np.sqrt(np.shape(M)[-1] - 1)  # Normalize
    U, s, _ = np.linalg.svd(C)
    return s**2, U.T


def plot_dominant_mode(s, U_T, X, fig_path):
    """Plots the variances and the fist 4 PCA modes.

    Args:
        s (array-like): 1D arrray of the variances of the principal components.
        U_T (array-like): The matrix to transform X into the principal
            components.
        X (array-like): The measurement matrix
        fig_path (str): The path to save the plot.
        comps (int): Number of modes to plot
    """
    plt.rcParams.update({"text.usetex": True})
    fig = plt.figure(figsize=(9, 6))

    ax_g = fig.add_subplot(3, 3, (2, 3))

    ax_g.set_yscale('log')

    ax_g.set_xlabel('Component')
    ax_g.set_ylabel('Variance')

    ax_g.set_xlim(-1, 100)

    ax_g.plot(s[:100], c='k', marker='o', markersize=2)

    for i in range(6):
        ax = fig.add_subplot(3, 3, 4+i)
        ax.set_title(f'Component {i}')

        if i > 2:
            ax.set_xlabel('Frame')

        ax.set_xlim(0, X.shape[-1])
        ax.plot(np.matmul(U_T[i], X), c='k')

    fig.tight_layout(pad=0.05)
    # plt.show()
    fig.savefig(fig_path, bbox_inches='tight')


if __name__ == '__main__':
    for i, vid_list in enumerate(make_vid_list()):
        print(f'Loading test {i+1}')
        X = read_meas_matrix(vid_list)
        g, U_T = pca(X)
        
        # Save and load the PCA so that the computation
        # can be skipped when replotting
        np.save(f'output/g_{i+1}.npy', g)
        np.save(f'output/S_T_{i+1}.npy', U_T)

        g = np.load(f'output/g_{i+1}.npy')

        U_T = np.load(f'output/S_T_{i+1}.npy')
        plot_dominant_mode(g, U_T, X, f'figures/PCA_{i+1}.pdf')
