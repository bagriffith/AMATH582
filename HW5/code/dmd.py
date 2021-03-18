import numpy as np
import matplotlib.pyplot as plt
import cv2

def dmd(X2, u, s, vh):
    """Returns the DMD modes and complex frequencies for the system.

    Args:
        X2 (array-like): The X_2^M matrix
        u (array-like): The U matrix of the X_1^M-1 SVD
        s (array-like): The s array of the X_1^M-1 SVD
        vh (array-like): The vh matrix of the X_1^M-1 SVD
    Returns:
        ndarray: Array of complex frequencies for DMD modes
        ndarray: Matrix with rows of the DMD modes
    """
    s_i = np.diagflat(1/s)
    S = np.dot(u.T, np.dot(X2, np.dot(vh.T, s_i)))
    mu, y = np.linalg.eig(S)
    psi = np.dot(u, y)
    w = np.log(mu)
    return w, psi


def x_dmd(t, psi, w, b):
    """The DMD approximation of x(t).

    Args:
        t (float): The time in frames
        psi (array-like): Matrix with rows of the DMD modes
        w (array-like): Array of complex frequencies for DMD modes
        b (array-like): Array of initial values of the DMD modes
    Returns:
        ndarray: DMD approximation of pixels at t
    """
    return np.dot(psi, np.exp(w*t)*b)


def frame_bg_sep(t, X, psi, w, b):
    """Separate the forground and background of frame t

    Args:
        t (int): The time in frames
        psi (array-like): Matrix with rows of the DMD modes
        w (array-like): Array of complex frequencies for DMD modes
        b (array-like): Array of initial values of the DMD modes
    Returns:
        ndarray: Foreground array
        ndarray: Background array
    """
    k = np.argmin(np.abs(w))
    bg = x_dmd(t, psi[:, k], w[k], b[k])
    bg = np.abs(bg)
    fg = X[:, t] - bg
    # r = np.where(fg < 0, fg, np.zeros_like(fg))
    # fg -= r
    # bg += r
    return fg, bg


def show_frame(frame, shape, path_out):
    """Plot the frame provided.

    Args:
        frame (array-like): 1 D array of pixels
        shape (tuple): The shape of the image (pixels_y, pixels_x)
        path_out (str): Path to save figure to
    """
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(np.reshape(frame, shape), cmap='Greys_r')
    fig.savefig(path_out,  bbox_inches='tight')


def save_fgbg_videos(X, psi, w, b, shape, path_out):

    fg_out = cv2.VideoWriter(path_out[:-4] + '-fg' + path_out[-4:], 
                cv2.VideoWriter_fourcc(*'DIVX'), 30, shape)

    bg_out = cv2.VideoWriter(path_out[:-4] + '-bg' + path_out[-4:], 
                cv2.VideoWriter_fourcc(*'DIVX'), 30, shape)

    for i in range(0, X.shape[1]):
        fg, bg = frame_bg_sep(t, X, psi, w, b)
        fg_out.write(np.uint8(fg))
        bg_out.write(np.uint8(bg))

    fg_out.release()
    bg_out.release()
    
