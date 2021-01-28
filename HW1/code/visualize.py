import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import peak_finding


def plot_xyz_psd(data, u, comp_opt, comp_cov):
    """Plots the total of slices perpendicular to x, y and z, and
    then overlays the gaussian model with the provided paramters.

    Args:
        data (np.ndarray): The signal power density. Should be
            3D of shape (nx, ny, xz).
        
        u (np.ndarray): An array of values for the x, y and z axes.

        comp_opt (list): 3 long list of lists of parameters for
            gaussian_model to be plotted.

        comp_cov (list): 3 long list of covariance matrices for
            parameters for gaussian_model to be plotted.
    """
    # High resolution span over u
    u_hr = np.linspace(u[0], u[-1], 256)

    over = [(0, 2), (1, 2), (0, 1)]
    labels = ['$k_x$', '$k_y$', '$k_z$']

    center = peak_finding.get_center(comp_opt)
    center_unc = peak_finding.get_center_unc(comp_cov)

    fig, axs = plt.subplots(3, figsize=(8, 6))
    fig.tight_layout()

    for n, ax in enumerate(axs):
        ax.set_xlabel(labels[n])
        ax.set_ylabel('PSD')
        ax.set_xlim(u[0], u[-1])
        ax.plot(u, np.mean(data, axis=over[n]))
        ax.plot(u_hr, peak_finding.gaussian_model(u_hr, *comp_opt[n]))

        center_label = r'$k_{0i} = ' + \
            r'{:.3f} \pm {:.3f} $'.format(center[n],
                                          center_unc[n])

        ax.annotate(center_label,
                    (center[n],
                     peak_finding.gaussian_model(center[n],
                                                 *comp_opt[n])),
                    (30, 0),
                    arrowprops={'arrowstyle': '->'},
                    textcoords='offset pixels')

    fig.savefig('figures/axes_fit.pdf', bbox_inches="tight")


def plot_projections(data, u, center, width):
    """Plot the projection onto the XY, XZ, and YZ planes by summing the
    perpendicular PSD.

    Args:
        data (np.ndarray): The signal power density. Should be
            3D of shape (nx, ny, xz).
        
        u (np.ndarray): An array of values for the x, y and z axes.

        center (tuple): Tuple of the center frequency (x, y, z)

        width (float): Width to show around that center frequency.
    """
    fig, ax_all = plt.subplots(2, 2, figsize=(6, 6))

    fig.tight_layout()

    fig.delaxes(ax_all[0][1])  # Get rid of the unused 3rd axis

    axs = [ax_all[0][0], ax_all[1][1], ax_all[1][0]]
    n_x_list = [0, 2, 0]
    n_y_list = [2, 1, 1]
    labels = ['$k_x$', '$k_y$', '$k_z$']

    proj_list = [np.sum(data, axis=n) for n in range(3)]
    high = np.max(proj_list)
    low = np.min(proj_list)

    for n, (n_x, n_y, ax) in enumerate(zip(n_x_list, n_y_list, axs)):
        # ax.set_aspect()
        ax.set_xlim(center[n_x]-width, center[n_x]+width)
        ax.set_ylim(center[n_y]-width, center[n_y]+width)

        ax.set_xlabel(labels[n_x])
        ax.set_ylabel(labels[n_y])

        if n == 0:
            proj_list[n] = proj_list[n].transpose()  # Preserve handedness
        ax.pcolormesh(u, u, proj_list[n], cmap='hot', vmin=low, vmax=high)
  
    plt.savefig('figures/projected_freq.pdf', bbox_inches="tight")


def record_path(loc_array):
    """Produces a csv and 3D rendering of a path from an
    array of positions.

    Args:
        loc_array (np,ndarray): 3xN array of points in the path
    """
    # Write a csv for the table
    np.savetxt('figures/position.csv',
               loc_array.transpose(),
               fmt='%.3f',
               delimiter=',',
               comments='',
               header="X, Y, Z")

    # Generate 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 35)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot(*(loc_array[i, :] for i in range(3)), marker='o')
    fig.savefig('figures/path.pdf', bbox_inches="tight")
